#include "store/model_registry.h"

#include <iterator>

#include <algorithm>
#include <chrono>
#include <fcntl.h>
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

ModelArtifactHandle::ModelArtifactHandle(
    int descriptor_value,
    int64_t size_value)
    : descriptor(descriptor_value),
      size(size_value) {}

ModelArtifactHandle::~ModelArtifactHandle() {
    if (descriptor >= 0) ::close(descriptor);
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
           identity.has_model_step();
}

void ModelRegistry::FillDistributorIdentity(
    rl::common::v1::ServiceInstanceIdentity* identity) const {
    identity->set_component("model-distributor");
    identity->set_instance_id(instance_id_);
    identity->set_lifecycle_epoch(1);
}

bool ModelRegistry::ValidateManifest(
    const rl::training::v1::ModelArtifactManifest& manifest,
    const std::string& local_artifact_path,
    std::filesystem::path* model_path,
    std::string* error) const {
    if (!ModelIdentityValid(manifest.identity())) {
        *error = "model identity is invalid";
        return false;
    }
    if (manifest.size_bytes() <= 0 || manifest.published_at_unix_ms() <= 0) {
        *error = "manifest artifact metadata is invalid";
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
    if (!std::filesystem::is_regular_file(canonical)) {
        *error = "artifact file does not exist";
        return false;
    }
    const uintmax_t size = std::filesystem::file_size(canonical, canonical_error);
    if (canonical_error ||
        size > static_cast<uintmax_t>(std::numeric_limits<int64_t>::max()) ||
        static_cast<int64_t>(size) != manifest.size_bytes()) {
        *error = "artifact size does not match manifest";
        return false;
    }
    *model_path = canonical;
    return true;
}

rl::training::v1::ModelRegisterResult ModelRegistry::Register(
    const rl::training::v1::ModelArtifactManifest& manifest,
    const std::string& local_artifact_path,
    RegisteredModel* registered,
    std::string* message) {
    std::filesystem::path path;
    std::string error;
    if (!ValidateManifest(manifest, local_artifact_path, &path, &error)) {
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
    if (descriptor < 0 || ::fstat(descriptor, &status) != 0 ||
        !S_ISREG(status.st_mode) || status.st_size <= 0 ||
        status.st_size != manifest.size_bytes()) {
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
        descriptor, static_cast<int64_t>(status.st_size));
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
    RegisteredModel record{manifest, std::move(artifact)};
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
    if (found == models_.end()) {
        last_error_ = "model step not found";
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
    return ServiceKey(request.aiserver()) + "\x1e" +
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
        if (AckEquivalent(*existing->second, request)) {
            *message = "model ACK already applied";
            return rl::training::v1::MODEL_ACK_RESULT_ALREADY_APPLIED;
        }
        last_error_ = "model ACK idempotency key conflicts with existing result";
        *message = last_error_;
        return rl::training::v1::MODEL_ACK_RESULT_REJECTED;
    }
    const ModelKey model_key{request.model().model_lineage_id(),
                             request.model().model_step()};
    const auto registered = models_.find(model_key);
    if (registered == models_.end() ||
        registered->second.manifest.identity().SerializeAsString() !=
            request.model().SerializeAsString()) {
        last_error_ = "registered model identity not found";
        *message = last_error_;
        return rl::training::v1::MODEL_ACK_RESULT_NOT_FOUND;
    }
    load_receipts_.push_back(request);
    acknowledgements_.emplace(key, std::prev(load_receipts_.end()));
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
    for (auto it = load_receipts_.begin(); it != load_receipts_.end();) {
        const auto& model = it->model();
        if (model.model_lineage_id() == lineage_id && model.model_step() <= latest_step &&
            latest_step - model.model_step() >= config_.retention_steps) {
            acknowledgements_.erase(AckKey(*it));
            it = load_receipts_.erase(it);
        } else {
            ++it;
        }
    }
}

void ModelRegistry::GetLoadReceipt(
    const rl::training::v1::GetModelLoadReceiptReq& request,
    rl::training::v1::GetModelLoadReceiptRsp* response) const {
    std::lock_guard<std::mutex> lock(mutex_);
    FillDistributorIdentity(response->mutable_distributor());
    const auto& model = request.model();
    const auto latest = latest_step_seen_.find(model.model_lineage_id());
    if (latest != latest_step_seen_.end() && model.model_step() <= latest->second &&
        latest->second - model.model_step() >= config_.retention_steps) {
        response->set_outside_retention(true);
        return;
    }
    const auto source = ServiceKey(request.aiserver());
    for (auto it = load_receipts_.rbegin(); it != load_receipts_.rend(); ++it) {
        if (it->model().model_step() == model.model_step() &&
            it->model().model_lineage_id() == model.model_lineage_id() &&
            ServiceKey(it->aiserver()) == source) {
            *response->mutable_receipt() = *it;
            return;
        }
    }
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
    rl::training::v1::ModelDistributorStatusRsp* response) const {
    std::lock_guard<std::mutex> lock(mutex_);
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
