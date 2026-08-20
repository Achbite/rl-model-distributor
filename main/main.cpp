#include "config/config_loader.h"
#include "grpc/model_distributor_service.h"

#include <grpcpp/grpcpp.h>

#include <iostream>
#include <memory>
#include <string>

int main(int argc, char* argv[]) {
    const std::string config_path =
        argc > 1 ? argv[1] : "configs/model_distributor_config.yaml";
    ModelDistributorConfig config;
    if (!LoadModelDistributorConfig(config_path, config)) {
        return 2;
    }

    ModelDistributorServiceImpl service(config);
    const std::string address =
        "0.0.0.0:" + std::to_string(config.listen_port);
    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    if (!server) {
        std::cerr << "[ModelDistributor] failed to listen on " << address
                  << std::endl;
        return 1;
    }
    std::cout << "[ModelDistributor] listening on " << address
              << ", instance_id=" << service.instance_id() << std::endl;
    server->Wait();
    return 0;
}
