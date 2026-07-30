#include "store/model_registry.h"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <unistd.h>

namespace {

int64_t NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

bool IsLowerHexSha256(const std::string& value) {
    if (value.size() != 64) return false;
    for (const char character : value) {
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f'))) {
            return false;
        }
    }
    return true;
}

bool ValidShape(const google::protobuf::RepeatedField<int64_t>& shape) {
    if (shape.empty()) return false;
    for (const int64_t dimension : shape) {
        if (dimension <= 0) return false;
    }
    return true;
}

bool ShapeIdentityEqual(
    const google::protobuf::RepeatedField<int64_t>& left,
    const google::protobuf::RepeatedField<int64_t>& right) {
    return left.size() == right.size() &&
           std::equal(left.begin(), left.end(), right.begin());
}

bool ManifestIdentityEqual(const maze::ModelArtifactManifest& left,
                           const maze::ModelArtifactManifest& right) {
    return left.schema_version() == right.schema_version() &&
           left.contract_version() == right.contract_version() &&
           left.model_version() == right.model_version() &&
           left.model_file() == right.model_file() &&
           left.size_bytes() == right.size_bytes() &&
           left.sha256() == right.sha256() &&
           ShapeIdentityEqual(left.input_shape(), right.input_shape()) &&
           ShapeIdentityEqual(left.action_shape(), right.action_shape()) &&
           ShapeIdentityEqual(left.value_shape(), right.value_shape()) &&
           left.seed() == right.seed() &&
           left.ready() == right.ready();
}

bool AckEqual(const maze::AckModelReq& left,
              const maze::AckModelReq& right) {
    return left.sha256() == right.sha256() &&
           left.load_status() == right.load_status();
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
    if (!context) {
        *error = "cannot allocate SHA-256 context";
        return false;
    }
    bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1;
    std::array<char, 64 * 1024> buffer{};
    while (ok && stream.good()) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = stream.gcount();
        if (count > 0) {
            ok = EVP_DigestUpdate(
                     context, buffer.data(), static_cast<std::size_t>(count)) ==
                 1;
        }
    }
    if (stream.bad()) ok = false;
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int digest_size = 0;
    if (ok) {
        ok = EVP_DigestFinal_ex(context, digest.data(), &digest_size) == 1;
    }
    EVP_MD_CTX_free(context);
    if (!ok) {
        *error = "failed to calculate SHA-256: " + path.string();
        return false;
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < digest_size; ++index) {
        output << std::setw(2)
               << static_cast<unsigned int>(digest[index]);
    }
    *checksum = output.str();
    return true;
}

ModelRegistry::ModelRegistry(ModelDistributorConfig config)
    : config_(std::move(config)) {
    std::error_code error;
    artifact_root_ = std::filesystem::weakly_canonical(
        std::filesystem::absolute(config_.artifact_root), error);
    if (error) {
        artifact_root_ = std::filesystem::absolute(config_.artifact_root);
    }
    std::ostringstream identity;
    identity << "model-distributor-" << NowMs() << "-" << getpid();
    instance_id_ = identity.str();
}

bool ModelRegistry::ValidateManifest(
    const maze::ModelArtifactManifest& manifest,
    std::filesystem::path* model_path,
    std::string* error) const {
    if (manifest.schema_version() != 1) {
        *error = "unsupported manifest schema_version";
        return false;
    }
    if (manifest.contract_version() != config_.contract_version) {
        *error = "manifest contract_version does not match service";
        return false;
    }
    if (manifest.model_version() < 0 || !manifest.ready()) {
        *error = "manifest model_version or ready state is invalid";
        return false;
    }
    if (manifest.model_file().empty() ||
        std::filesystem::path(manifest.model_file()).filename() !=
            std::filesystem::path(manifest.model_file())) {
        *error = "model_file must be a file name";
        return false;
    }
    if (manifest.size_bytes() <= 0 ||
        !IsLowerHexSha256(manifest.sha256()) ||
        !ValidShape(manifest.input_shape()) ||
        !ValidShape(manifest.action_shape()) ||
        !ValidShape(manifest.value_shape())) {
        *error = "manifest size, checksum, or shape is invalid";
        return false;
    }

    std::filesystem::path requested = FileUriPath(manifest.artifact_uri());
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
    const auto size = std::filesystem::file_size(canonical, canonical_error);
    if (canonical_error ||
        size > static_cast<uintmax_t>(std::numeric_limits<int64_t>::max()) ||
        static_cast<int64_t>(size) != manifest.size_bytes()) {
        *error = "artifact size does not match manifest";
        return false;
    }
    std::string actual_checksum;
    if (!ComputeSha256(canonical, &actual_checksum, error)) return false;
    if (actual_checksum != manifest.sha256()) {
        *error = "artifact checksum does not match manifest";
        return false;
    }
    *model_path = canonical;
    return true;
}

maze::ModelRegisterResult ModelRegistry::Register(
    const maze::ModelArtifactManifest& manifest,
    RegisteredModel* registered,
    std::string* message) {
    std::filesystem::path model_path;
    std::string validation_error;
    if (!ValidateManifest(manifest, &model_path, &validation_error)) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++register_attempt_count_;
        ++rejected_register_count_;
        last_error_ = validation_error;
        *message = validation_error;
        return maze::MODEL_REGISTER_RESULT_REJECTED_INVALID;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    ++register_attempt_count_;
    const int key = manifest.model_version();
    const auto existing = models_.find(key);
    if (existing != models_.end()) {
        if (ManifestIdentityEqual(existing->second.manifest, manifest)) {
            ++duplicate_register_count_;
            *registered = existing->second;
            *message = "model already registered";
            return maze::MODEL_REGISTER_RESULT_ALREADY_REGISTERED;
        }
        ++rejected_register_count_;
        last_error_ = "model identity conflicts with registered version";
        *message = last_error_;
        return maze::MODEL_REGISTER_RESULT_REJECTED_CONFLICT;
    }

    if (latest_version_seen_ >= 0 &&
        manifest.model_version() <=
            latest_version_seen_ -
                static_cast<int>(config_.retention_versions)) {
        ++rejected_register_count_;
        last_error_ = "model version is outside the serving retention window";
        *message = last_error_;
        return maze::MODEL_REGISTER_RESULT_REJECTED_INVALID;
    }

    RegisteredModel record{manifest, model_path};
    models_.emplace(key, record);
    latest_version_seen_ =
        std::max(latest_version_seen_, manifest.model_version());

    while (models_.size() > config_.retention_versions) {
        const int removed = models_.begin()->first;
        models_.erase(models_.begin());
        for (auto ack = acknowledgements_.begin();
             ack != acknowledgements_.end();) {
            if (ack->first.first == removed) {
                ack = acknowledgements_.erase(ack);
            } else {
                ++ack;
            }
        }
    }
    *registered = record;
    *message = "model registered";
    return maze::MODEL_REGISTER_RESULT_REGISTERED;
}

bool ModelRegistry::Get(int model_version,
                        RegisteredModel* model,
                        std::string* message) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = models_.find(model_version);
    if (found == models_.end()) {
        last_error_ = "model not found";
        *message = last_error_;
        return false;
    }
    *model = found->second;
    *message = "model found";
    return true;
}

bool ModelRegistry::GetLatest(RegisteredModel* model,
                              std::string* message) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (models_.empty()) {
        last_error_ = "model not found";
        *message = last_error_;
        return false;
    }
    const auto found = std::prev(models_.end());
    *model = found->second;
    *message = "latest model found";
    return true;
}

maze::ModelAckResult ModelRegistry::Ack(const maze::AckModelReq& request,
                                        std::string* message) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto model = models_.find(request.model_version());
    if (model == models_.end()) {
        *message = "model not found";
        last_error_ = *message;
        return maze::MODEL_ACK_RESULT_NOT_FOUND;
    }
    if (request.aiserver_id().empty() ||
        request.sha256() != model->second.manifest.sha256() ||
        request.load_status() == maze::MODEL_LOAD_STATUS_UNSPECIFIED) {
        *message = "model ACK identity is invalid";
        last_error_ = *message;
        return maze::MODEL_ACK_RESULT_REJECTED;
    }

    const AckKey key{request.model_version(), request.aiserver_id()};
    const auto existing = acknowledgements_.find(key);
    if (existing != acknowledgements_.end()) {
        if (AckEqual(existing->second, request)) {
            *message = "model ACK already applied";
            return maze::MODEL_ACK_RESULT_ALREADY_APPLIED;
        }
        *message = "model ACK conflicts with existing result";
        last_error_ = *message;
        return maze::MODEL_ACK_RESULT_REJECTED;
    }

    acknowledgements_.emplace(key, request);
    latest_ack_aiserver_id_ = request.aiserver_id();
    latest_ack_model_version_ = request.model_version();
    latest_ack_status_ = request.load_status();
    if (request.load_status() == maze::MODEL_LOAD_STATUS_LOADED) {
        ++loaded_ack_count_;
    } else {
        ++failed_ack_count_;
        last_error_ = request.message();
    }
    *message = "model ACK applied";
    return maze::MODEL_ACK_RESULT_APPLIED;
}

void ModelRegistry::RecordDownloadAttempt() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++download_request_count_;
}

void ModelRegistry::RecordDownloadSuccess(int64_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++download_success_count_;
    bytes_served_ += bytes;
}

void ModelRegistry::RecordDownloadFailure(const std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++download_failure_count_;
    last_error_ = error;
}

void ModelRegistry::GetStatus(
    maze::ModelDistributorStatusRsp* response) const {
    std::lock_guard<std::mutex> lock(mutex_);
    response->set_protocol_version(config_.protocol_version);
    response->set_distributor_instance_id(instance_id_);
    response->set_ready(true);
    response->set_registered_model_count(
        static_cast<int64_t>(models_.size()));
    int latest_version = -1;
    std::string latest_checksum;
    for (const auto& item : models_) {
        if (item.first > latest_version) {
            latest_version = item.first;
            latest_checksum = item.second.manifest.sha256();
        }
    }
    response->set_latest_model_version(latest_version);
    response->set_latest_model_checksum(latest_checksum);
    response->set_register_attempt_count(register_attempt_count_);
    response->set_duplicate_register_count(duplicate_register_count_);
    response->set_rejected_register_count(rejected_register_count_);
    response->set_download_request_count(download_request_count_);
    response->set_download_success_count(download_success_count_);
    response->set_download_failure_count(download_failure_count_);
    response->set_bytes_served(bytes_served_);
    response->set_loaded_ack_count(loaded_ack_count_);
    response->set_failed_ack_count(failed_ack_count_);
    response->set_latest_ack_aiserver_id(latest_ack_aiserver_id_);
    response->set_latest_ack_model_version(latest_ack_model_version_);
    response->set_latest_ack_status(latest_ack_status_);
    response->set_last_error(last_error_);
    response->set_timestamp_ms(NowMs());
}

const std::string& ModelRegistry::instance_id() const {
    return instance_id_;
}

std::size_t ModelRegistry::chunk_bytes() const {
    return config_.chunk_bytes;
}
