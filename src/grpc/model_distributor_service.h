#pragma once

#include "config/config_loader.h"
#include "maze.grpc.pb.h"
#include "store/model_registry.h"

class ModelDistributorServiceImpl final
    : public maze::ModelDistributorService::Service {
public:
    explicit ModelDistributorServiceImpl(
        const ModelDistributorConfig& config);

    grpc::Status RegisterModel(grpc::ServerContext* context,
                               const maze::RegisterModelReq* request,
                               maze::RegisterModelRsp* response) override;

    grpc::Status GetModelManifest(
        grpc::ServerContext* context,
        const maze::GetModelManifestReq* request,
        maze::GetModelManifestRsp* response) override;

    grpc::Status DownloadModel(
        grpc::ServerContext* context,
        const maze::DownloadModelReq* request,
        grpc::ServerWriter<maze::ModelChunk>* writer) override;

    grpc::Status AckModel(grpc::ServerContext* context,
                          const maze::AckModelReq* request,
                          maze::AckModelRsp* response) override;

    grpc::Status GetModelDistributorStatus(
        grpc::ServerContext* context,
        const maze::ModelDistributorStatusReq* request,
        maze::ModelDistributorStatusRsp* response) override;

    const std::string& instance_id() const;

private:
    ModelRegistry registry_;
};
