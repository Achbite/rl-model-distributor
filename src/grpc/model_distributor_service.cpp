#include "grpc/model_distributor_service.h"

#include <fstream>
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
    const auto& requested = request->requested_model();
    const bool has_artifact_digest =
        requested.has_artifact_digest() &&
        !requested.artifact_digest().hex().empty();
    const bool has_manifest_digest =
        requested.has_manifest_digest() &&
        !requested.manifest_digest().hex().empty();
    bool found = false;
    if (request->latest_in_lineage()) {
        found = registry_.GetLatest(
            requested.model_lineage_id(), &record, &message);
    } else if (!has_artifact_digest && !has_manifest_digest) {
        found = registry_.GetByVersion(
            requested.model_lineage_id(), requested.model_version(),
            &record, &message);
    } else if (has_artifact_digest && has_manifest_digest) {
        found = registry_.Get(requested, &record, &message);
    } else {
        message = "requested model selector has incomplete digest identity";
    }
    if (!found) {
        response->set_ret_code(-1);
        response->set_message(message);
        return grpc::Status::OK;
    }
    response->set_ret_code(0);
    response->set_message(message);
    *response->mutable_manifest() = record.manifest;
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
    std::ifstream input(record.path, std::ios::binary);
    if (!input.is_open()) {
        registry_.RecordDownloadFailure("cannot open registered model");
        return grpc::Status(grpc::StatusCode::INTERNAL,
                            "cannot open registered model");
    }
    std::vector<char> buffer(registry_.chunk_bytes());
    int64_t offset = 0;
    while (input.good()) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count <= 0) break;
        rl::training::v1::ModelChunk chunk;
        *chunk.mutable_model() = record.manifest.identity();
        chunk.set_offset(offset);
        chunk.set_data(buffer.data(), static_cast<size_t>(count));
        if (context->IsCancelled() || !writer->Write(chunk)) {
            registry_.RecordDownloadFailure("model download cancelled");
            return grpc::Status(grpc::StatusCode::CANCELLED,
                                "model download cancelled");
        }
        offset += count;
    }
    if (input.bad() || offset != record.manifest.size_bytes()) {
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
