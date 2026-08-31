#include "store/model_registry.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <sys/stat.h>
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

bool Sha256Bytes(const void* data,
                 size_t size,
                 std::string* checksum,
                 std::string* error);

bool ValidRolloutEstimatorProfile(
    const rl::training::v1::RolloutEstimatorProfile& profile) {
    if (!std::isfinite(profile.gamma()) || profile.gamma() < 0.0 ||
        profile.gamma() > 1.0 ||
        !std::isfinite(profile.gae_lambda()) ||
        profile.gae_lambda() < 0.0 || profile.gae_lambda() > 1.0 ||
        profile.tmax() == 0 ||
        !LowerSha256(profile.profile_digest())) {
        return false;
    }
    rl::training::v1::RolloutEstimatorProfile canonical(profile);
    canonical.clear_profile_digest();
    std::string payload;
    google::protobuf::io::StringOutputStream output(&payload);
    google::protobuf::io::CodedOutputStream coded(&output);
    coded.SetSerializationDeterministic(true);
    if (!canonical.SerializeToCodedStream(&coded) || coded.HadError()) {
        return false;
    }
    coded.Trim();
    std::string digest;
    std::string error;
    return Sha256Bytes(payload.data(), payload.size(), &digest, &error) &&
           digest == profile.profile_digest().hex();
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

bool ComputeSha256Descriptor(int descriptor,
                             int64_t size,
                             std::string* checksum,
                             std::string* error) {
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (context == nullptr) {
        *error = "cannot allocate SHA-256 context";
        return false;
    }
    bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1;
    std::array<char, 64 * 1024> buffer{};
    int64_t offset = 0;
    while (ok && offset < size) {
        const size_t requested = static_cast<size_t>(
            std::min<int64_t>(size - offset, buffer.size()));
        const ssize_t count = ::pread(
            descriptor, buffer.data(), requested, static_cast<off_t>(offset));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            ok = false;
            break;
        }
        ok = EVP_DigestUpdate(
                 context, buffer.data(), static_cast<size_t>(count)) == 1;
        offset += count;
    }
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int digest_size = 0;
    if (ok && offset == size) {
        ok = EVP_DigestFinal_ex(context, digest.data(), &digest_size) == 1;
    } else {
        ok = false;
    }
    EVP_MD_CTX_free(context);
    if (!ok || digest_size != 32) {
        *error = "failed to calculate SHA-256 from opened model artifact";
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

std::string StepDirectoryName(uint64_t step) {
    std::ostringstream output;
    output << std::setfill('0') << std::setw(7) << step;
    return output.str();
}

}  // namespace

ModelArtifactHandle::ModelArtifactHandle(
    int descriptor_value,
    int64_t size_value,
    std::filesystem::path path_value)
    : descriptor(descriptor_value),
      size(size_value),
      path(std::move(path_value)) {}

ModelArtifactHandle::~ModelArtifactHandle() {
    if (descriptor >= 0) ::close(descriptor);
}

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
           identity.has_model_step() &&
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
    const rl::common::v1::ContractIdentity& contract,
    const std::string& local_artifact_path,
    std::filesystem::path* model_path,
    std::string* error) const {
    if (!ContractMatches(contract)) {
        *error = "registration contract identity does not match service artifact";
        return false;
    }
    if (!ModelIdentityValid(manifest.identity())) {
        *error = "model identity is invalid";
        return false;
    }
    // The training-contract digest owns ABI and rollout semantics. AIServer
    // validates the actual ONNX dtype and shapes when it prepares the bytes.
    if (!LowerSha256(manifest.training_config_digest()) ||
        !LowerSha256(manifest.training_contract_digest()) ||
        manifest.training_contract_digest().hex() !=
            config_.training_contract_digest ||
        !ValidRolloutEstimatorProfile(
            manifest.rollout_estimator_profile()) ||
        manifest.size_bytes() <= 0 || manifest.published_at_unix_ms() <= 0) {
        *error = "manifest training identity or artifact metadata is invalid";
        return false;
    }
    std::string computed_manifest_digest;
    if (!ComputeManifestDigest(manifest, &computed_manifest_digest, error) ||
        computed_manifest_digest != manifest.identity().manifest_digest().hex()) {
        if (error->empty()) *error = "manifest digest does not match canonical bytes";
        return false;
    }

    const std::filesystem::path requested(local_artifact_path);
    if (requested.empty() || !requested.is_absolute()) {
        *error = "local_artifact_path must be absolute";
        return false;
    }
    std::error_code canonical_error;
    const auto requested_status =
        std::filesystem::symlink_status(requested, canonical_error);
    if (canonical_error ||
        requested_status.type() != std::filesystem::file_type::regular) {
        *error = "artifact must be a regular non-symbolic-link file";
        return false;
    }
    canonical_error.clear();
    const std::filesystem::path canonical =
        std::filesystem::weakly_canonical(requested, canonical_error);
    if (canonical_error || !IsWithin(canonical, artifact_root_)) {
        *error = "artifact path is outside configured root";
        return false;
    }
    if (canonical.filename() != "SaveModel.onnx" ||
        canonical.parent_path().parent_path() != artifact_root_ ||
        canonical.parent_path().filename() !=
            StepDirectoryName(manifest.identity().model_step()) ||
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
    const rl::common::v1::ContractIdentity& contract,
    const std::string& local_artifact_path,
    RegisteredModel* registered,
    std::string* message) {
    std::filesystem::path path;
    std::string error;
    if (!ValidateManifest(manifest, contract, local_artifact_path,
                          &path, &error)) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++register_attempt_count_;
        ++rejected_register_count_;
        last_error_ = error;
        *message = error;
        return rl::training::v1::MODEL_REGISTER_RESULT_REJECTED_INVALID;
    }

    const int descriptor = ::open(
        path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    struct stat status {};
    std::string opened_digest;
    if (descriptor < 0 || ::fstat(descriptor, &status) != 0 ||
        !S_ISREG(status.st_mode) || status.st_size <= 0 ||
        status.st_size != manifest.size_bytes() ||
        !ComputeSha256Descriptor(descriptor, status.st_size,
                                 &opened_digest, &error) ||
        opened_digest != manifest.identity().artifact_digest().hex()) {
        if (descriptor >= 0) ::close(descriptor);
        if (error.empty()) {
            error = "opened model artifact changed after manifest validation";
        }
        std::lock_guard<std::mutex> lock(mutex_);
        ++register_attempt_count_;
        ++rejected_register_count_;
        last_error_ = error;
        *message = error;
        return rl::training::v1::MODEL_REGISTER_RESULT_REJECTED_INVALID;
    }
    auto artifact = std::make_shared<ModelArtifactHandle>(
        descriptor, static_cast<int64_t>(status.st_size), path);
    std::lock_guard<std::mutex> lock(mutex_);
    ++register_attempt_count_;
    const ModelKey key{manifest.identity().model_lineage_id(),
                       manifest.identity().model_step()};
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
        last_error_ = "model lineage/step conflicts with registered identity";
        *message = last_error_;
        return rl::training::v1::MODEL_REGISTER_RESULT_REJECTED_CONFLICT;
    }
    const auto latest = latest_step_seen_.find(key.lineage_id);
    if (latest == latest_step_seen_.end() && key.step != 0) {
        ++rejected_register_count_;
        last_error_ = "model lineage must begin at step 0";
        *message = last_error_;
        return rl::training::v1::MODEL_REGISTER_RESULT_REJECTED_INVALID;
    }
    if (latest != latest_step_seen_.end() &&
        (latest->second == std::numeric_limits<uint64_t>::max() ||
         key.step != latest->second + 1)) {
        ++rejected_register_count_;
        last_error_ = key.step <= latest->second
                          ? "model step is stale or no longer available"
                          : "model publication sequence contains a gap";
        *message = last_error_;
        return rl::training::v1::MODEL_REGISTER_RESULT_REJECTED_INVALID;
    }
    RegisteredModel record{manifest, path, std::move(artifact)};
    models_.emplace(key, record);
    latest_step_seen_[key.lineage_id] = key.step;
    if (!has_latest_model_ ||
        latest_model_.model_lineage_id() != key.lineage_id ||
        key.step >= latest_model_.model_step()) {
        latest_model_ = manifest.identity();
        has_latest_model_ = true;
    }

    std::vector<ModelKey> lineage_keys;
    for (const auto& [candidate, value] : models_) {
        (void)value;
        if (candidate.lineage_id == key.lineage_id) {
            lineage_keys.push_back(candidate);
        }
    }
    while (lineage_keys.size() > config_.retention_steps) {
        const ModelKey removed = lineage_keys.front();
        lineage_keys.erase(lineage_keys.begin());
        models_.erase(removed);
    }
    PruneSettlementFactsLocked(key.lineage_id);
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
    const ModelKey key{identity.model_lineage_id(), identity.model_step()};
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

bool ModelRegistry::GetByStep(const std::string& lineage_id,
                              uint64_t model_step,
                              RegisteredModel* model,
                              std::string* message) {
    if (lineage_id.empty()) {
        *message = "model lineage is required";
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = models_.find(ModelKey{lineage_id, model_step});
    if (found == models_.end()) {
        last_error_ = "model step not found";
        *message = last_error_;
        return false;
    }
    *model = found->second;
    *message = "model step found";
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
    const auto latest = latest_step_seen_.find(lineage_id);
    if (latest == latest_step_seen_.end()) {
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

bool ModelRegistry::GetAvailableRange(const std::string& lineage_id,
                                      uint64_t* floor,
                                      uint64_t* latest,
                                      std::string* message) const {
    if (lineage_id.empty() || floor == nullptr || latest == nullptr) {
        if (message != nullptr) *message = "model lineage is required";
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return AvailableRangeLocked(lineage_id, floor, latest, message);
}

bool ModelRegistry::AvailableRangeLocked(const std::string& lineage_id,
                                         uint64_t* floor,
                                         uint64_t* latest,
                                         std::string* message) const {
    const auto latest_found = latest_step_seen_.find(lineage_id);
    if (latest_found == latest_step_seen_.end() ||
        models_.find(ModelKey{lineage_id, latest_found->second}) ==
            models_.end()) {
        if (message != nullptr) *message = "model lineage not found";
        return false;
    }
    *latest = latest_found->second;
    *floor = *latest;
    while (*floor > 0 &&
           models_.find(ModelKey{lineage_id, *floor - 1}) != models_.end()) {
        --(*floor);
    }
    if (message != nullptr) *message = "contiguous model range found";
    return true;
}

bool ModelRegistry::GetManifestSnapshot(
    const rl::training::v1::ModelIdentity& selector,
    bool latest_in_lineage,
    RegisteredModel* model,
    uint64_t* floor,
    uint64_t* latest,
    std::string* message) {
    if (selector.model_lineage_id().empty() || model == nullptr ||
        floor == nullptr || latest == nullptr || message == nullptr) {
        if (message != nullptr) *message = "model selector is invalid";
        return false;
    }
    const bool has_artifact_digest =
        selector.has_artifact_digest() &&
        !selector.artifact_digest().hex().empty();
    const bool has_manifest_digest =
        selector.has_manifest_digest() &&
        !selector.manifest_digest().hex().empty();
    if (!latest_in_lineage && has_artifact_digest != has_manifest_digest) {
        *message = "requested model selector has incomplete digest identity";
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (!latest_in_lineage && !selector.has_model_step()) {
        *message = "requested model selector is missing model step";
        return false;
    }
    ModelKey key{selector.model_lineage_id(), selector.model_step()};
    if (latest_in_lineage) {
        const auto latest_found = latest_step_seen_.find(key.lineage_id);
        if (latest_found == latest_step_seen_.end()) {
            *message = "model lineage not found";
            return false;
        }
        key.step = latest_found->second;
    }
    const auto found = models_.find(key);
    if (found == models_.end() ||
        (!latest_in_lineage && has_artifact_digest &&
         found->second.manifest.identity().SerializeAsString() !=
             selector.SerializeAsString())) {
        last_error_ = has_artifact_digest ? "model identity not found"
                                          : "model step not found";
        *message = last_error_;
        return false;
    }
    if (!AvailableRangeLocked(key.lineage_id, floor, latest, message)) {
        return false;
    }
    *model = found->second;
    *message = latest_in_lineage ? "latest model found" : "model found";
    return true;
}

std::string ModelRegistry::AckKey(
    const rl::training::v1::AckModelReq& request) {
    return DownloadKey(request.aiserver(), request.model());
}

std::string ModelRegistry::ServiceKey(
    const rl::common::v1::ServiceInstanceIdentity& service) {
    return service.component() + "\x1f" + service.instance_id() + "\x1f" +
           std::to_string(service.lifecycle_epoch());
}

std::string ModelRegistry::DownloadKey(
    const rl::common::v1::ServiceInstanceIdentity& service,
    const rl::training::v1::ModelIdentity& model) {
    return ServiceKey(service) + "\x1e" + model.SerializeAsString();
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
    const auto completed = completed_downloads_.find(
        DownloadKey(request.aiserver(), request.model()));
    const bool completed_download_identity =
        completed != completed_downloads_.end() &&
        completed->second.SerializeAsString() ==
            request.model().SerializeAsString();
    if (!completed_download_identity) {
        last_error_ = "completed model download fact not found";
        *message = last_error_;
        return rl::training::v1::MODEL_ACK_RESULT_NOT_FOUND;
    }
    acknowledgements_.emplace(key, request);
    acknowledgement_models_.emplace(
        key,
        ModelKey{request.model().model_lineage_id(),
                 request.model().model_step()});
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

void ModelRegistry::PruneSettlementFactsLocked(
    const std::string& lineage_id) {
    const auto latest = latest_step_seen_.find(lineage_id);
    if (latest == latest_step_seen_.end()) return;
    const uint64_t latest_step = latest->second;
    auto outside_retention = [&](const ModelKey& key) {
        return key.lineage_id == lineage_id &&
               key.step <= latest_step &&
               latest_step - key.step >= config_.retention_steps;
    };
    for (auto iterator = acknowledgement_models_.begin();
         iterator != acknowledgement_models_.end();) {
        if (!outside_retention(iterator->second)) {
            ++iterator;
            continue;
        }
        acknowledgements_.erase(iterator->first);
        iterator = acknowledgement_models_.erase(iterator);
    }
    for (auto iterator = completed_downloads_.begin();
         iterator != completed_downloads_.end();) {
        const auto& model = iterator->second;
        const ModelKey key{model.model_lineage_id(), model.model_step()};
        if (outside_retention(key)) {
            iterator = completed_downloads_.erase(iterator);
        } else {
            ++iterator;
        }
    }
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
    // A stream may finish after its registry entry was evicted. RegisterModel
    // prunes old settlement facts on every publication, while this exact fact
    // lets that already-open stream complete one idempotent model-level ACK.
    completed_downloads_[DownloadKey(requester, model)] = model;
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
    if (has_latest_model_) {
        *response->mutable_latest_model() = latest_model_;
        uint64_t floor = 0;
        uint64_t latest = 0;
        const auto latest_found = latest_step_seen_.find(
            latest_model_.model_lineage_id());
        if (latest_found != latest_step_seen_.end() &&
            models_.find(ModelKey{latest_model_.model_lineage_id(),
                                  latest_found->second}) != models_.end()) {
            latest = latest_found->second;
            floor = latest;
            while (floor > 0 &&
                   models_.find(ModelKey{
                       latest_model_.model_lineage_id(), floor - 1}) !=
                       models_.end()) {
                --floor;
            }
            response->set_available_floor_model_step(floor);
            response->set_latest_available_model_step(latest);
        }
    }
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
