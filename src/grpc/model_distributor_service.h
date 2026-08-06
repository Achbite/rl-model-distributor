#pragma once

#include "config/config_loader.h"
#include "store/model_registry.h"
#include "training.grpc.pb.h"

class ModelDistributorServiceImpl final
    : public rl::training::v1::ModelDistributorService::Service {
public:
    explicit ModelDistributorServiceImpl(
        const ModelDistributorConfig& config);

    grpc::Status RegisterModel(
        grpc::ServerContext* context,
        const rl::training::v1::RegisterModelReq* request,
        rl::training::v1::RegisterModelRsp* response) override;
    grpc::Status GetModelManifest(
        grpc::ServerContext* context,
        const rl::training::v1::GetModelManifestReq* request,
        rl::training::v1::GetModelManifestRsp* response) override;
    grpc::Status DownloadModel(
        grpc::ServerContext* context,
        const rl::training::v1::DownloadModelReq* request,
        grpc::ServerWriter<rl::training::v1::ModelChunk>* writer) override;
    grpc::Status AckModel(
        grpc::ServerContext* context,
        const rl::training::v1::AckModelReq* request,
        rl::training::v1::AckModelRsp* response) override;
    grpc::Status GetModelDistributorStatus(
        grpc::ServerContext* context,
        const rl::training::v1::ModelDistributorStatusReq* request,
        rl::training::v1::ModelDistributorStatusRsp* response) override;

    const std::string& instance_id() const;

private:
    ModelRegistry registry_;
};
