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
        response->set_result(
            rl::training::v1::MODEL_REGISTER_RESULT_REJECTED_INVALID);
        response->set_message("manifest is required");
        return grpc::Status::OK;
    }
    RegisteredModel record;
    std::string message;
    const auto result = registry_.Register(
        request->manifest(), request->local_artifact_path(), &record, &message);
    response->set_result(result);
    response->set_message(message);
    if (result == rl::training::v1::MODEL_REGISTER_RESULT_REGISTERED ||
        result ==
            rl::training::v1::MODEL_REGISTER_RESULT_ALREADY_REGISTERED) {
        *response->mutable_manifest() = record.manifest;
    }
    return grpc::Status::OK;
}

grpc::Status ModelDistributorServiceImpl::GetModelManifest(
    grpc::ServerContext*,
    const rl::training::v1::GetModelManifestReq* request,
    rl::training::v1::GetModelManifestRsp* response) {
    registry_.FillDistributorIdentity(response->mutable_distributor());
    const auto& selector = request->requested_model();
    if (!ModelRegistry::ServiceIdentityValid(request->requester()) ||
        selector.model_lineage_id().empty() ||
        (!request->latest_in_lineage() && !selector.has_model_step())) {
        response->set_message("model lookup identity is invalid");
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            response->message());
    }
    RegisteredModel record;
    std::string message;
    uint64_t floor = 0;
    uint64_t latest = 0;
    const bool found = registry_.GetManifestSnapshot(
        request->requested_model(), request->latest_in_lineage(), &record,
        &floor, &latest, &message);
    if (!found) {
        response->set_result(
            rl::training::v1::MODEL_LOOKUP_RESULT_NOT_FOUND);
        response->set_message(message);
        return grpc::Status::OK;
    }
    response->set_result(rl::training::v1::MODEL_LOOKUP_RESULT_FOUND);
    response->set_message(message);
    *response->mutable_manifest() = record.manifest;
    response->set_available_floor_model_step(floor);
    response->set_latest_available_model_step(latest);
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
    registry_.RecordDownloadSuccess(offset);
    return grpc::Status::OK;
}

grpc::Status ModelDistributorServiceImpl::AckModel(
    grpc::ServerContext*,
    const rl::training::v1::AckModelReq* request,
    rl::training::v1::AckModelRsp* response) {
    registry_.FillDistributorIdentity(response->mutable_distributor());
    std::string message;
    const auto result = registry_.Ack(*request, &message);
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

#include "rl_sdk/metric_catalog.h"
#include "proto/metrics/catalog.pb.h"

rl::training::v1::GetMetricCatalogRsp ModelDistributorServiceImpl::MetricCatalog() {
    namespace wire = rl::training::v1;
    wire::GetMetricCatalogRsp catalog;
    registry_.FillDistributorIdentity(catalog.mutable_source());
    const std::string method = "rl.training.v1.ModelDistributorService/GetModelDistributorStatus";
    rl_sdk::AddStatusMetric(catalog, "model.distributor.registered_model_count", "Registered Model Count", "model", "count", "distributor", method, "registered_model_count");
    rl_sdk::AddStatusMetric(catalog, "model.distributor.register_attempt_count", "Register Attempt Count", "model", "count", "distributor", method, "register_attempt_count");
    rl_sdk::AddStatusMetric(catalog, "model.distributor.duplicate_register_count", "Duplicate Register Count", "model", "count", "distributor", method, "duplicate_register_count");
    rl_sdk::AddStatusMetric(catalog, "model.distributor.rejected_register_count", "Rejected Register Count", "model", "count", "distributor", method, "rejected_register_count");
    rl_sdk::AddStatusMetric(catalog, "model.distributor.download_request_count", "Download Request Count", "model", "count", "distributor", method, "download_request_count");
    rl_sdk::AddStatusMetric(catalog, "model.distributor.download_success_count", "Download Success Count", "model", "count", "distributor", method, "download_success_count");
    rl_sdk::AddStatusMetric(catalog, "model.distributor.download_failure_count", "Download Failure Count", "model", "count", "distributor", method, "download_failure_count");
    rl_sdk::AddStatusMetric(catalog, "model.distributor.bytes_served", "Bytes Served", "model", "bytes", "distributor", method, "bytes_served");
    rl_sdk::AddStatusMetric(catalog, "model.distributor.loaded_ack_count", "Loaded Ack Count", "model", "count", "distributor", method, "loaded_ack_count");
    rl_sdk::AddStatusMetric(catalog, "model.distributor.failed_ack_count", "Failed Ack Count", "model", "count", "distributor", method, "failed_ack_count");
    return catalog;
}
