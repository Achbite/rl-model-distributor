#include "grpc/model_distributor_service.h"
#include "maze.grpc.pb.h"
#include "store/model_registry.h"

#include <grpcpp/grpcpp.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unistd.h>

namespace {

int Fail(const std::string& message) {
    std::cerr << message << std::endl;
    return 1;
}

maze::ModelArtifactManifest BuildManifest(
    const std::filesystem::path& path,
    const std::string& checksum) {
    maze::ModelArtifactManifest manifest;
    manifest.set_schema_version(1);
    manifest.set_contract_version("0.3.0");
    manifest.set_run_id("rpc-test-run");
    manifest.set_model_version(0);
    manifest.set_artifact_uri("file://" + path.string());
    manifest.set_model_file(path.filename().string());
    manifest.set_size_bytes(
        static_cast<int64_t>(std::filesystem::file_size(path)));
    manifest.set_sha256(checksum);
    manifest.add_input_shape(1);
    manifest.add_input_shape(13);
    manifest.add_action_shape(1);
    manifest.add_action_shape(9);
    manifest.add_value_shape(1);
    manifest.add_value_shape(1);
    manifest.set_seed(0);
    manifest.set_ready(true);
    manifest.set_published_ts_ms(1);
    return manifest;
}

}  // namespace

int main() {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() /
                          ("model-distributor-rpc-" +
                           std::to_string(::getpid()));
    fs::create_directories(root / "rpc-test-run");
    const fs::path model_path =
        root / "rpc-test-run/model_v000000.onnx";
    const std::string expected_payload =
        "0123456789abcdefghijklmnopqrstuvwxyz";
    {
        std::ofstream output(model_path, std::ios::binary);
        output << expected_payload;
    }

    std::string checksum;
    std::string error;
    if (!ComputeSha256(model_path, &checksum, &error)) {
        return Fail(error);
    }
    ModelDistributorConfig config;
    config.run_id = "rpc-test-run";
    config.artifact_root = root.string();
    config.contract_version = "0.3.0";
    config.chunk_bytes = 5;
    ModelDistributorServiceImpl service(config);

    int selected_port = 0;
    grpc::ServerBuilder builder;
    builder.AddListeningPort(
        "127.0.0.1:0",
        grpc::InsecureServerCredentials(),
        &selected_port);
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    if (!server || selected_port <= 0) {
        fs::remove_all(root);
        return Fail("failed to start test ModelDistributor");
    }
    auto channel = grpc::CreateChannel(
        "127.0.0.1:" + std::to_string(selected_port),
        grpc::InsecureChannelCredentials());
    auto stub = maze::ModelDistributorService::NewStub(channel);

    maze::RegisterModelReq register_request;
    *register_request.mutable_manifest() =
        BuildManifest(model_path, checksum);
    maze::RegisterModelRsp register_response;
    grpc::ClientContext register_context;
    grpc::Status register_status = stub->RegisterModel(
        &register_context, register_request, &register_response);
    if (!register_status.ok() ||
        register_response.result() !=
            maze::MODEL_REGISTER_RESULT_REGISTERED) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("RegisterModel RPC failed");
    }

    maze::RegisterModelReq invalid_request = register_request;
    invalid_request.mutable_manifest()->set_sha256(
        std::string(64, '0'));
    maze::RegisterModelRsp invalid_response;
    grpc::ClientContext invalid_context;
    grpc::Status invalid_status = stub->RegisterModel(
        &invalid_context, invalid_request, &invalid_response);
    if (!invalid_status.ok() ||
        invalid_response.result() !=
            maze::MODEL_REGISTER_RESULT_REJECTED_INVALID) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("checksum mismatch was not rejected");
    }

    maze::DownloadModelReq download_request;
    download_request.set_run_id("rpc-test-run");
    download_request.set_model_version(0);
    download_request.set_aiserver_id("aiserver-0");
    grpc::ClientContext download_context;
    auto reader = stub->DownloadModel(
        &download_context, download_request);
    maze::ModelChunk chunk;
    int64_t expected_offset = 0;
    std::string reconstructed;
    while (reader->Read(&chunk)) {
        if (chunk.offset() != expected_offset ||
            chunk.run_id() != "rpc-test-run" ||
            chunk.model_version() != 0) {
            server->Shutdown();
            fs::remove_all(root);
            return Fail("DownloadModel chunk identity or offset is invalid");
        }
        reconstructed.append(chunk.data());
        expected_offset += static_cast<int64_t>(chunk.data().size());
    }
    if (!reader->Finish().ok() ||
        reconstructed != expected_payload) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("DownloadModel payload reconstruction failed");
    }

    maze::GetModelManifestReq unknown_request;
    unknown_request.set_run_id("rpc-test-run");
    unknown_request.set_model_version(99);
    maze::GetModelManifestRsp unknown_response;
    grpc::ClientContext unknown_context;
    grpc::Status unknown_status = stub->GetModelManifest(
        &unknown_context, unknown_request, &unknown_response);
    if (!unknown_status.ok() || unknown_response.ret_code() == 0) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("unknown model version was returned");
    }

    maze::GetModelManifestReq latest_request;
    latest_request.set_run_id("rpc-test-run");
    latest_request.set_aiserver_id("aiserver-0");
    latest_request.set_latest(true);
    maze::GetModelManifestRsp latest_response;
    grpc::ClientContext latest_context;
    if (!stub->GetModelManifest(
             &latest_context, latest_request, &latest_response).ok() ||
        latest_response.ret_code() != 0 ||
        latest_response.manifest().model_version() != 0) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("latest model lookup failed");
    }

    maze::AckModelReq ack_request;
    ack_request.set_run_id("rpc-test-run");
    ack_request.set_aiserver_id("aiserver-0");
    ack_request.set_model_version(0);
    ack_request.set_sha256(checksum);
    ack_request.set_load_status(maze::MODEL_LOAD_STATUS_LOADED);
    maze::AckModelRsp first_ack;
    maze::AckModelRsp second_ack;
    grpc::ClientContext first_ack_context;
    grpc::ClientContext second_ack_context;
    if (!stub->AckModel(
             &first_ack_context, ack_request, &first_ack).ok() ||
        !stub->AckModel(
             &second_ack_context, ack_request, &second_ack).ok() ||
        first_ack.result() != maze::MODEL_ACK_RESULT_APPLIED ||
        second_ack.result() !=
            maze::MODEL_ACK_RESULT_ALREADY_APPLIED) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("AckModel RPC was not idempotent");
    }

    server->Shutdown();
    fs::remove_all(root);
    return 0;
}
