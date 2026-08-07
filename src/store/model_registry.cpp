#include "store/model_registry.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unistd.h>

namespace {

int64_t NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

bool LowerSha256(const rl::common::v1::ContentDigest& digest) {
    if (digest.algorithm() != rl::common::v1::DIGEST_ALGORITHM_SHA256 ||
        digest.hex().size() != 64) {
        return false;
    }
    for (char character : digest.hex()) {
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f'))) {
            return false;
        }
    }
    return true;
}

bool ValidShape(const google::protobuf::RepeatedField<int64_t>& shape) {
    return !shape.empty() &&
           std::all_of(shape.begin(), shape.end(),
                       [](int64_t value) { return value > 0; });
}

bool ValidSchema(const rl::common::v1::SchemaIdentity& schema) {
    return !schema.schema_id().empty() && schema.schema_version() > 0 &&
           LowerSha256(schema.canonical_digest());
}

bool ValidSemantics(
    const rl::training::v1::TrainingSemanticsIdentity& semantics) {
    return !semantics.training_contract_id().empty() &&
           ValidSchema(semantics.observation_schema()) &&
           ValidSchema(semantics.action_schema()) &&
           ValidSchema(semantics.reward_schema()) &&
           !semantics.policy_distribution_schema_id().empty() &&
           !semantics.model_architecture_id().empty() &&
           LowerSha256(semantics.semantics_digest());
}

std::filesystem::path FileUriPath(const std::string& uri) {
    constexpr const char* prefix = "file://";
    if (uri.rfind(prefix, 0) != 0) return {};
    return std::filesystem::path(uri.substr(7));
}

bool IsWithin(const std::filesystem::path& child,
              const std::filesystem::path& parent) {
    auto child_it = child.begin();
    auto parent_it = parent.begin();
    for (; parent_it != parent.end(); ++parent_it, ++child_it) {
        if (child_it == child.end() || *child_it != *parent_it) return false;
    }
    return true;
}

bool Sha256Bytes(const void* data,
                 size_t size,
                 std::string* checksum,
                 std::string* error) {
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (context == nullptr) {
        *error = "cannot allocate SHA-256 context";
        return false;
    }
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int digest_size = 0;
    const bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
                    EVP_DigestUpdate(context, data, size) == 1 &&
                    EVP_DigestFinal_ex(context, digest.data(), &digest_size) ==
                        1;
    EVP_MD_CTX_free(context);
    if (!ok || digest_size != 32) {
        *error = "failed to calculate SHA-256";
        return false;
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < digest_size; ++index) {
        output << std::setw(2) << static_cast<unsigned int>(digest[index]);
    }
    *checksum = output.str();
    return true;
}

}  // namespace

bool ComputeSha256(const std::filesystem::path& path,
                   std::string* checksum,
                   std::string* error) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        *error = "cannot open model file: " + path.string();
        return false;
    }
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (context == nullptr) {
        *error = "cannot allocate SHA-256 context";
        return false;
    }
    bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1;
    std::array<char, 64 * 1024> buffer{};
    while (ok && stream.good()) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = stream.gcount();
        if (count > 0) {
            ok = EVP_DigestUpdate(context, buffer.data(),
                                  static_cast<size_t>(count)) == 1;
        }
    }
    if (stream.bad()) ok = false;
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int digest_size = 0;
    if (ok) {
        ok = EVP_DigestFinal_ex(context, digest.data(), &digest_size) == 1;
    }
    EVP_MD_CTX_free(context);
    if (!ok || digest_size != 32) {
        *error = "failed to calculate SHA-256: " + path.string();
        return false;
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < digest_size; ++index) {
        output << std::setw(2) << static_cast<unsigned int>(digest[index]);
    }
    *checksum = output.str();
    return true;
}

bool ComputeManifestDigest(
    const rl::training::v1::ModelArtifactManifest& manifest,
    std::string* digest,
    std::string* error) {
    rl::training::v1::ModelArtifactManifest copy(manifest);
    copy.mutable_identity()->clear_manifest_digest();
    std::string serialized;
    google::protobuf::io::StringOutputStream output(&serialized);
    google::protobuf::io::CodedOutputStream coded(&output);
    coded.SetSerializationDeterministic(true);
    if (!copy.SerializeToCodedStream(&coded) || coded.HadError()) {
        *error = "failed to deterministically serialize model manifest";
        return false;
    }
    coded.Trim();
    return Sha256Bytes(serialized.data(), serialized.size(), digest, error);
}

ModelRegistry::ModelRegistry(ModelDistributorConfig config)
    : config_(std::move(config)) {
    std::error_code error;
    artifact_root_ = std::filesystem::weakly_canonical(
        std::filesystem::absolute(config_.artifact_root), error);
    if (error) artifact_root_ = std::filesystem::absolute(config_.artifact_root);
    std::ostringstream identity;
    identity << "model-distributor-" << NowMs() << "-" << getpid();
    instance_id_ = identity.str();
}

bool ModelRegistry::ServiceIdentityValid(
    const rl::common::v1::ServiceInstanceIdentity& identity) {
    return !identity.component().empty() && !identity.instance_id().empty() &&
           identity.lifecycle_epoch() > 0;
}

bool ModelRegistry::ModelIdentityValid(
    const rl::training::v1::ModelIdentity& identity) {
    return !identity.model_lineage_id().empty() &&
           LowerSha256(identity.artifact_digest()) &&
           LowerSha256(identity.manifest_digest());
}

bool ModelRegistry::ContractMatches(
    const rl::common::v1::ContractIdentity& contract) const {
    return contract.package_name() == config_.contract.package_name &&
           contract.package_version() == config_.contract.package_version &&
           LowerSha256(contract.source_digest()) &&
           contract.source_digest().hex() == config_.contract.source_digest &&
           LowerSha256(contract.artifact_digest()) &&
           contract.artifact_digest().hex() == config_.contract.artifact_digest &&
           contract.platform() == config_.contract.platform &&
           contract.generator_identity() == config_.contract.generator_identity;
}

void ModelRegistry::FillContractIdentity(
    rl::common::v1::ContractIdentity* contract) const {
    contract->set_package_name(config_.contract.package_name);
    contract->set_package_version(config_.contract.package_version);
    contract->mutable_source_digest()->set_algorithm(
        rl::common::v1::DIGEST_ALGORITHM_SHA256);
    contract->mutable_source_digest()->set_hex(config_.contract.source_digest);
    contract->mutable_artifact_digest()->set_algorithm(
        rl::common::v1::DIGEST_ALGORITHM_SHA256);
    contract->mutable_artifact_digest()->set_hex(
        config_.contract.artifact_digest);
    contract->set_platform(config_.contract.platform);
    contract->set_generator_identity(config_.contract.generator_identity);
}

void ModelRegistry::FillDistributorIdentity(
    rl::common::v1::ServiceInstanceIdentity* identity) const {
    identity->set_component("model-distributor");
    identity->set_instance_id(instance_id_);
    identity->set_lifecycle_epoch(1);
}

bool ModelRegistry::ValidateManifest(
    const rl::training::v1::ModelArtifactManifest& manifest,
    std::filesystem::path* model_path,
    std::string* error) const {
    if (manifest.manifest_schema_version() != 1) {
        *error = "unsupported manifest schema version";
        return false;
    }
    if (!ContractMatches(manifest.contract())) {
        *error = "manifest contract identity does not match service artifact";
        return false;
    }
    if (!ModelIdentityValid(manifest.identity()) || !manifest.ready()) {
        *error = "model identity or ready state is invalid";
        return false;
    }
    if (!ValidSchema(manifest.observation_schema()) ||
        !ValidSchema(manifest.action_schema()) ||
        manifest.model_architecture_id().empty() ||
        manifest.tensor_dtype().empty() ||
        !ValidShape(manifest.input_shape()) ||
        !ValidShape(manifest.action_shape()) ||
        !ValidShape(manifest.value_shape()) ||
        !LowerSha256(manifest.training_config_digest()) ||
        !ValidSemantics(manifest.training_semantics()) ||
        manifest.training_semantics().observation_schema().SerializeAsString() !=
            manifest.observation_schema().SerializeAsString() ||
        manifest.training_semantics().action_schema().SerializeAsString() !=
            manifest.action_schema().SerializeAsString() ||
        manifest.training_semantics().model_architecture_id() !=
            manifest.model_architecture_id() ||
        manifest.train_updates() < 0 || manifest.trained_samples() < 0 ||
        manifest.published_at_unix_ms() <= 0) {
        *error = "manifest schema, shape, or training identity is invalid";
        return false;
    }
    if (manifest.model_file().empty() ||
        std::filesystem::path(manifest.model_file()).filename() !=
            std::filesystem::path(manifest.model_file()) ||
        manifest.size_bytes() <= 0) {
        *error = "manifest model file identity is invalid";
        return false;
    }
    std::string computed_manifest_digest;
    if (!ComputeManifestDigest(manifest, &computed_manifest_digest, error) ||
        computed_manifest_digest != manifest.identity().manifest_digest().hex()) {
        if (error->empty()) *error = "manifest digest does not match canonical bytes";
        return false;
    }

    const std::filesystem::path requested =
        FileUriPath(manifest.artifact_uri());
    if (requested.empty()) {
        *error = "artifact_uri must use file://";
        return false;
    }
    std::error_code canonical_error;
    const std::filesystem::path canonical =
        std::filesystem::weakly_canonical(requested, canonical_error);
    if (canonical_error || !IsWithin(canonical, artifact_root_)) {
        *error = "artifact path is outside configured root";
        return false;
    }
    if (canonical.filename() != manifest.model_file() ||
        !std::filesystem::is_regular_file(canonical)) {
        *error = "artifact file does not exist or has a mismatched name";
        return false;
    }
    const uintmax_t size = std::filesystem::file_size(canonical, canonical_error);
    if (canonical_error ||
        size > static_cast<uintmax_t>(std::numeric_limits<int64_t>::max()) ||
        static_cast<int64_t>(size) != manifest.size_bytes()) {
        *error = "artifact size does not match manifest";
        return false;
    }
    std::string artifact_digest;
    if (!ComputeSha256(canonical, &artifact_digest, error)) return false;
    if (artifact_digest != manifest.identity().artifact_digest().hex()) {
        *error = "artifact digest does not match model bytes";
        return false;
    }
    *model_path = canonical;
    return true;
}

rl::training::v1::ModelRegisterResult ModelRegistry::Register(
    const rl::training::v1::ModelArtifactManifest& manifest,
    RegisteredModel* registered,
    std::string* message) {
    std::filesystem::path path;
    std::string error;
    if (!ValidateManifest(manifest, &path, &error)) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++register_attempt_count_;
        ++rejected_register_count_;
        last_error_ = error;
        *message = error;
        return rl::training::v1::MODEL_REGISTER_RESULT_REJECTED_INVALID;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    ++register_attempt_count_;
    const ModelKey key{manifest.identity().model_lineage_id(),
                       manifest.identity().model_version()};
    const auto existing = models_.find(key);
    if (existing != models_.end()) {
        if (existing->second.manifest.SerializeAsString() ==
            manifest.SerializeAsString()) {
            ++duplicate_register_count_;
            *registered = existing->second;
            *message = "model already registered";
            return rl::training::v1::MODEL_REGISTER_RESULT_ALREADY_REGISTERED;
        }
        ++rejected_register_count_;
        last_error_ = "model lineage/version conflicts with registered identity";
        *message = last_error_;
        return rl::training::v1::MODEL_REGISTER_RESULT_REJECTED_CONFLICT;
    }
    const auto latest = latest_version_seen_.find(key.lineage_id);
    if (latest != latest_version_seen_.end() &&
        key.version + config_.retention_versions <= latest->second) {
        ++rejected_register_count_;
        last_error_ = "model version is outside the serving retention window";
        *message = last_error_;
        return rl::training::v1::MODEL_REGISTER_RESULT_REJECTED_INVALID;
    }
    RegisteredModel record{manifest, path};
    models_.emplace(key, record);
    latest_version_seen_[key.lineage_id] =
        latest == latest_version_seen_.end()
            ? key.version
            : std::max(latest->second, key.version);
    latest_model_ = manifest.identity();
    has_latest_model_ = true;

    std::vector<ModelKey> lineage_keys;
    for (const auto& [candidate, value] : models_) {
        (void)value;
        if (candidate.lineage_id == key.lineage_id) {
            lineage_keys.push_back(candidate);
        }
    }
    while (lineage_keys.size() > config_.retention_versions) {
        const ModelKey removed = lineage_keys.front();
        lineage_keys.erase(lineage_keys.begin());
        models_.erase(removed);
    }
    *registered = record;
    *message = "model registered";
    return rl::training::v1::MODEL_REGISTER_RESULT_REGISTERED;
}

bool ModelRegistry::Get(const rl::training::v1::ModelIdentity& identity,
                        RegisteredModel* model,
                        std::string* message) {
    if (!ModelIdentityValid(identity)) {
        *message = "requested model identity is invalid";
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const ModelKey key{identity.model_lineage_id(), identity.model_version()};
    const auto found = models_.find(key);
    if (found == models_.end() ||
        found->second.manifest.identity().SerializeAsString() !=
            identity.SerializeAsString()) {
        last_error_ = "model identity not found";
        *message = last_error_;
        return false;
    }
    *model = found->second;
    *message = "model found";
    return true;
}

bool ModelRegistry::GetByVersion(const std::string& lineage_id,
                                 uint64_t model_version,
                                 RegisteredModel* model,
                                 std::string* message) {
    if (lineage_id.empty()) {
        *message = "model lineage is required";
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = models_.find(ModelKey{lineage_id, model_version});
    if (found == models_.end()) {
        last_error_ = "model version not found";
        *message = last_error_;
        return false;
    }
    *model = found->second;
    *message = "model version found";
    return true;
}

bool ModelRegistry::GetLatest(const std::string& lineage_id,
                              RegisteredModel* model,
                              std::string* message) {
    if (lineage_id.empty()) {
        *message = "model lineage is required";
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto latest = latest_version_seen_.find(lineage_id);
    if (latest == latest_version_seen_.end()) {
        last_error_ = "model lineage not found";
        *message = last_error_;
        return false;
    }
    const auto found = models_.find(ModelKey{lineage_id, latest->second});
    if (found == models_.end()) {
        last_error_ = "latest model is outside retention";
        *message = last_error_;
        return false;
    }
    *model = found->second;
    *message = "latest model found";
    return true;
}

std::string ModelRegistry::AckKey(
    const rl::training::v1::AckModelReq& request) {
    return ServiceKey(request.aiserver()) + "\x1f" +
           request.load_instance_id();
}

std::string ModelRegistry::ServiceKey(
    const rl::common::v1::ServiceInstanceIdentity& service) {
    return service.component() + "\x1f" + service.instance_id() + "\x1f" +
           std::to_string(service.lifecycle_epoch());
}

bool ModelRegistry::AckEquivalent(
    const rl::training::v1::AckModelReq& left,
    const rl::training::v1::AckModelReq& right) {
    return left.model().SerializeAsString() == right.model().SerializeAsString() &&
           left.load_status() == right.load_status();
}

rl::training::v1::ModelAckResult ModelRegistry::Ack(
    const rl::training::v1::AckModelReq& request,
    std::string* message) {
    if (!ServiceIdentityValid(request.aiserver()) ||
        !ModelIdentityValid(request.model()) ||
        request.load_instance_id().empty() ||
        request.load_status() ==
            rl::training::v1::MODEL_LOAD_STATUS_UNSPECIFIED) {
        *message = "model ACK identity is invalid";
        return rl::training::v1::MODEL_ACK_RESULT_REJECTED;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string key = AckKey(request);
    const auto existing = acknowledgements_.find(key);
    if (existing != acknowledgements_.end()) {
        if (AckEquivalent(existing->second, request)) {
            *message = "model ACK already applied";
            return rl::training::v1::MODEL_ACK_RESULT_ALREADY_APPLIED;
        }
        last_error_ = "model ACK idempotency key conflicts with existing result";
        *message = last_error_;
        return rl::training::v1::MODEL_ACK_RESULT_REJECTED;
    }
    const ModelKey model_key{request.model().model_lineage_id(),
                             request.model().model_version()};
    const auto model = models_.find(model_key);
    const auto completed = completed_downloads_.find(
        ServiceKey(request.aiserver()));
    const bool registered_identity =
        model != models_.end() &&
        model->second.manifest.identity().SerializeAsString() ==
            request.model().SerializeAsString();
    const bool completed_download_identity =
        completed != completed_downloads_.end() &&
        completed->second.SerializeAsString() ==
            request.model().SerializeAsString();
    if (!registered_identity && !completed_download_identity) {
        last_error_ = "model identity not found";
        *message = last_error_;
        return rl::training::v1::MODEL_ACK_RESULT_NOT_FOUND;
    }
    acknowledgements_.emplace(key, request);
    if (completed_download_identity) completed_downloads_.erase(completed);
    latest_ack_aiserver_ = request.aiserver();
    latest_ack_model_ = request.model();
    latest_ack_status_ = request.load_status();
    has_latest_ack_ = true;
    if (request.load_status() == rl::training::v1::MODEL_LOAD_STATUS_LOADED) {
        ++loaded_ack_count_;
    } else {
        ++failed_ack_count_;
        last_error_ = request.message();
    }
    *message = "model ACK applied";
    return rl::training::v1::MODEL_ACK_RESULT_APPLIED;
}

void ModelRegistry::RecordDownloadAttempt() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++download_request_count_;
}

void ModelRegistry::RecordDownloadSuccess(
    const rl::common::v1::ServiceInstanceIdentity& requester,
    const rl::training::v1::ModelIdentity& model,
    int64_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++download_success_count_;
    bytes_served_ += bytes;
    completed_downloads_[ServiceKey(requester)] = model;
}

void ModelRegistry::RecordDownloadFailure(const std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++download_failure_count_;
    last_error_ = error;
}

void ModelRegistry::GetStatus(
    rl::training::v1::ModelDistributorStatusRsp* response) const {
    std::lock_guard<std::mutex> lock(mutex_);
    FillContractIdentity(response->mutable_contract());
    FillDistributorIdentity(response->mutable_distributor());
    response->set_ready(true);
    response->set_registered_model_count(static_cast<int64_t>(models_.size()));
    if (has_latest_model_) *response->mutable_latest_model() = latest_model_;
    response->set_register_attempt_count(register_attempt_count_);
    response->set_duplicate_register_count(duplicate_register_count_);
    response->set_rejected_register_count(rejected_register_count_);
    response->set_download_request_count(download_request_count_);
    response->set_download_success_count(download_success_count_);
    response->set_download_failure_count(download_failure_count_);
    response->set_bytes_served(bytes_served_);
    response->set_loaded_ack_count(loaded_ack_count_);
    response->set_failed_ack_count(failed_ack_count_);
    if (has_latest_ack_) {
        *response->mutable_latest_ack_aiserver() = latest_ack_aiserver_;
        *response->mutable_latest_ack_model() = latest_ack_model_;
        response->set_latest_ack_status(latest_ack_status_);
    }
    response->set_last_error(last_error_);
    response->set_timestamp_unix_ms(NowMs());
}

const std::string& ModelRegistry::instance_id() const {
    return instance_id_;
}

std::size_t ModelRegistry::chunk_bytes() const {
    return config_.chunk_bytes;
}
