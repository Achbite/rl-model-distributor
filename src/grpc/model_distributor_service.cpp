#include "grpc/model_distributor_service.h"

#include <fstream>
#include <vector>

ModelDistributorServiceImpl::ModelDistributorServiceImpl(
    const ModelDistributorConfig& config)
    : registry_(config) {}

grpc::Status ModelDistributorServiceImpl::RegisterModel(
    grpc::ServerContext*,
    const maze::RegisterModelReq* request,
    maze::RegisterModelRsp* response) {
    if (!request->has_manifest()) {
        response->set_ret_code(-1);
        response->set_result(
            maze::MODEL_REGISTER_RESULT_REJECTED_INVALID);
        response->set_message("manifest is required");
        response->set_distributor_instance_id(registry_.instance_id());
        return grpc::Status::OK;
    }
    RegisteredModel record;
    std::string message;
    const auto result =
        registry_.Register(request->manifest(), &record, &message);
    response->set_ret_code(
        result == maze::MODEL_REGISTER_RESULT_REGISTERED ||
                result == maze::MODEL_REGISTER_RESULT_ALREADY_REGISTERED
            ? 0
            : -1);
    response->set_result(result);
    response->set_message(message);
    response->set_distributor_instance_id(registry_.instance_id());
    if (response->ret_code() == 0) {
        *response->mutable_manifest() = record.manifest;
    }
    return grpc::Status::OK;
}

grpc::Status ModelDistributorServiceImpl::GetModelManifest(
    grpc::ServerContext*,
    const maze::GetModelManifestReq* request,
    maze::GetModelManifestRsp* response) {
    RegisteredModel record;
    std::string message;
    const bool found = request->latest()
        ? registry_.GetLatest(request->run_id(), &record, &message)
        : registry_.Get(
              request->run_id(), request->model_version(), &record, &message);
    if (!found) {
        response->set_ret_code(-1);
        response->set_message(message);
        response->set_distributor_instance_id(registry_.instance_id());
        return grpc::Status::OK;
    }
    response->set_ret_code(0);
    response->set_message(message);
    *response->mutable_manifest() = record.manifest;
    response->set_distributor_instance_id(registry_.instance_id());
    return grpc::Status::OK;
}

grpc::Status ModelDistributorServiceImpl::DownloadModel(
    grpc::ServerContext* context,
    const maze::DownloadModelReq* request,
    grpc::ServerWriter<maze::ModelChunk>* writer) {
    registry_.RecordDownloadAttempt();
    RegisteredModel record;
    std::string message;
    if (!registry_.Get(
            request->run_id(), request->model_version(), &record, &message)) {
        registry_.RecordDownloadFailure(message);
        return grpc::Status(grpc::StatusCode::NOT_FOUND, message);
    }
    std::ifstream input(record.path, std::ios::binary);
    if (!input.is_open()) {
        registry_.RecordDownloadFailure("cannot open registered model");
        return grpc::Status(
            grpc::StatusCode::INTERNAL, "cannot open registered model");
    }

    std::vector<char> buffer(registry_.chunk_bytes());
    int64_t offset = 0;
    while (input.good()) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count <= 0) break;
        maze::ModelChunk chunk;
        chunk.set_run_id(request->run_id());
        chunk.set_model_version(request->model_version());
        chunk.set_offset(offset);
        chunk.set_data(buffer.data(), static_cast<std::size_t>(count));
        if (context->IsCancelled() || !writer->Write(chunk)) {
            registry_.RecordDownloadFailure("model download cancelled");
            return grpc::Status(
                grpc::StatusCode::CANCELLED, "model download cancelled");
        }
        offset += count;
    }
    if (input.bad() || offset != record.manifest.size_bytes()) {
        registry_.RecordDownloadFailure("model download read failed");
        return grpc::Status(
            grpc::StatusCode::DATA_LOSS, "model download read failed");
    }
    registry_.RecordDownloadSuccess(offset);
    return grpc::Status::OK;
}

grpc::Status ModelDistributorServiceImpl::AckModel(
    grpc::ServerContext*,
    const maze::AckModelReq* request,
    maze::AckModelRsp* response) {
    std::string message;
    const auto result = registry_.Ack(*request, &message);
    response->set_ret_code(
        result == maze::MODEL_ACK_RESULT_APPLIED ||
                result == maze::MODEL_ACK_RESULT_ALREADY_APPLIED
            ? 0
            : -1);
    response->set_result(result);
    response->set_message(message);
    response->set_distributor_instance_id(registry_.instance_id());
    return grpc::Status::OK;
}

grpc::Status ModelDistributorServiceImpl::GetModelDistributorStatus(
    grpc::ServerContext*,
    const maze::ModelDistributorStatusReq* request,
    maze::ModelDistributorStatusRsp* response) {
    registry_.GetStatus(request->run_id(), response);
    return grpc::Status::OK;
}

const std::string& ModelDistributorServiceImpl::instance_id() const {
    return registry_.instance_id();
}
