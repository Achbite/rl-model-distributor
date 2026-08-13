#include "grpc/model_distributor_service.h"

#include <algorithm>
#include <cerrno>
#include <unistd.h>
#include <vector>

ModelDistributorServiceImpl::ModelDistributorServiceImpl(
    const ModelDistributorConfig& config)
    : registry_(config) {}

grpc::Status ModelDistributorServiceImpl::RegisterModel(
    grpc::ServerContext*,
    const rl::training::v1::RegisterModelReq* request,
    rl::training::v1::RegisterModelRsp* response) {
    registry_.FillDistributorIdentity(response->mutable_distributor());
    if (!request->has_manifest()) {
        response->set_ret_code(-1);
        response->set_result(
            rl::training::v1::MODEL_REGISTER_RESULT_REJECTED_INVALID);
        response->set_message("manifest is required");
        return grpc::Status::OK;
    }
    RegisteredModel record;
    std::string message;
    const auto result =
        registry_.Register(request->manifest(), &record, &message);
    response->set_ret_code(
        result == rl::training::v1::MODEL_REGISTER_RESULT_REGISTERED ||
                result ==
                    rl::training::v1::MODEL_REGISTER_RESULT_ALREADY_REGISTERED
            ? 0
            : -1);
    response->set_result(result);
    response->set_message(message);
    if (response->ret_code() == 0) {
        *response->mutable_manifest() = record.manifest;
    }
    return grpc::Status::OK;
}

grpc::Status ModelDistributorServiceImpl::GetModelManifest(
    grpc::ServerContext*,
    const rl::training::v1::GetModelManifestReq* request,
    rl::training::v1::GetModelManifestRsp* response) {
    registry_.FillDistributorIdentity(response->mutable_distributor());
    if (!ModelRegistry::ServiceIdentityValid(request->requester())) {
        response->set_ret_code(-1);
        response->set_message("requester service identity is invalid");
        return grpc::Status::OK;
    }
    RegisteredModel record;
    std::string message;
    uint64_t floor = 0;
    uint64_t latest = 0;
    const bool found = registry_.GetManifestSnapshot(
        request->requested_model(), request->latest_in_lineage(), &record,
        &floor, &latest, &message);
    if (!found) {
        response->set_ret_code(-1);
        response->set_message(message);
        return grpc::Status::OK;
    }
    response->set_ret_code(0);
    response->set_message(message);
    *response->mutable_manifest() = record.manifest;
    response->set_available_floor_model_version(floor);
    response->set_latest_available_model_version(latest);
    return grpc::Status::OK;
}

grpc::Status ModelDistributorServiceImpl::DownloadModel(
    grpc::ServerContext* context,
    const rl::training::v1::DownloadModelReq* request,
    grpc::ServerWriter<rl::training::v1::ModelChunk>* writer) {
    registry_.RecordDownloadAttempt();
    if (!ModelRegistry::ServiceIdentityValid(request->requester())) {
        registry_.RecordDownloadFailure("requester service identity invalid");
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            "requester service identity invalid");
    }
    RegisteredModel record;
    std::string message;
    if (!registry_.Get(request->requested_model(), &record, &message)) {
        registry_.RecordDownloadFailure(message);
        return grpc::Status(grpc::StatusCode::NOT_FOUND, message);
    }
    if (!record.artifact || record.artifact->descriptor < 0 ||
        record.artifact->size != record.manifest.size_bytes()) {
        registry_.RecordDownloadFailure("registered model handle is invalid");
        return grpc::Status(grpc::StatusCode::INTERNAL,
                            "registered model handle is invalid");
    }
    std::vector<char> buffer(registry_.chunk_bytes());
    int64_t offset = 0;
    while (offset < record.artifact->size) {
        const size_t requested = static_cast<size_t>(std::min<int64_t>(
            record.artifact->size - offset,
            static_cast<int64_t>(buffer.size())));
        ssize_t count = ::pread(record.artifact->descriptor, buffer.data(),
                                requested, static_cast<off_t>(offset));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            registry_.RecordDownloadFailure("model download read failed");
            return grpc::Status(grpc::StatusCode::DATA_LOSS,
                                "model download read failed");
        }
        rl::training::v1::ModelChunk chunk;
        *chunk.mutable_model() = record.manifest.identity();
        chunk.set_offset(offset);
        chunk.set_data(buffer.data(), static_cast<size_t>(count));
        if (context->IsCancelled() || !writer->Write(chunk)) {
            registry_.RecordDownloadFailure("model download cancelled");
            return grpc::Status(grpc::StatusCode::CANCELLED,
                                "model download cancelled");
        }
        offset += static_cast<int64_t>(count);
    }
    if (offset != record.manifest.size_bytes()) {
        registry_.RecordDownloadFailure("model download read failed");
        return grpc::Status(grpc::StatusCode::DATA_LOSS,
                            "model download read failed");
    }
    registry_.RecordDownloadSuccess(
        request->requester(), record.manifest.identity(), offset);
    return grpc::Status::OK;
}

grpc::Status ModelDistributorServiceImpl::AckModel(
    grpc::ServerContext*,
    const rl::training::v1::AckModelReq* request,
    rl::training::v1::AckModelRsp* response) {
    registry_.FillDistributorIdentity(response->mutable_distributor());
    std::string message;
    const auto result = registry_.Ack(*request, &message);
    response->set_ret_code(
        result == rl::training::v1::MODEL_ACK_RESULT_APPLIED ||
                result == rl::training::v1::MODEL_ACK_RESULT_ALREADY_APPLIED
            ? 0
            : -1);
    response->set_result(result);
    response->set_message(message);
    return grpc::Status::OK;
}

grpc::Status ModelDistributorServiceImpl::GetModelDistributorStatus(
    grpc::ServerContext*,
    const rl::training::v1::ModelDistributorStatusReq*,
    rl::training::v1::ModelDistributorStatusRsp* response) {
    registry_.GetStatus(response);
    return grpc::Status::OK;
}

const std::string& ModelDistributorServiceImpl::instance_id() const {
    return registry_.instance_id();
}
