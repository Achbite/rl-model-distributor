#include "grpc/model_distributor_service.h"
#include "test_helpers.h"

#include <grpcpp/grpcpp.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>

namespace {

int Fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

std::filesystem::path ModelPath(const std::filesystem::path& root,
                                uint64_t model_step) {
    std::ostringstream directory;
    directory << std::setfill('0') << std::setw(7) << model_step;
    const auto path = root / directory.str() / "SaveModel.onnx";
    std::filesystem::create_directories(path.parent_path());
    return path;
}

}  // namespace

int main() {
    namespace fs = std::filesystem;
    using namespace test_helpers;
    const fs::path root = fs::temp_directory_path() /
                          ("model-distributor-rpc-" +
                           std::to_string(::getpid()));
    const fs::path model_path = ModelPath(root, 0);
    const std::string payload = "0123456789abcdefghijklmnopqrstuvwxyz";
    {
        std::ofstream output(model_path, std::ios::binary);
        output << payload;
    }
    std::string checksum;
    std::string error;
    if (!ComputeSha256(model_path, &checksum, &error)) return Fail(error);
    ModelDistributorConfig config = Config(root);
    ModelDistributorServiceImpl service(config);

    int port = 0;
    grpc::ServerBuilder builder;
    builder.AddListeningPort("127.0.0.1:0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    if (!server || port <= 0) {
        fs::remove_all(root);
        return Fail("failed to start test ModelDistributor");
    }
    auto channel = grpc::CreateChannel(
        "127.0.0.1:" + std::to_string(port),
        grpc::InsecureChannelCredentials());
    auto stub = rl::training::v1::ModelDistributorService::NewStub(channel);
    auto manifest = Manifest(model_path, checksum);
    auto latest_registered = manifest;

    rl::training::v1::RegisterModelReq register_request;
    *register_request.mutable_manifest() = manifest;
    rl::training::v1::RegisterModelRsp register_response;
    grpc::ClientContext register_context;
    if (!stub->RegisterModel(&register_context, register_request,
                             &register_response)
             .ok() ||
        register_response.result() !=
            rl::training::v1::MODEL_REGISTER_RESULT_REGISTERED ||
        register_response.distributor().instance_id().empty()) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("RegisterModel RPC failed");
    }

    rl::training::v1::GetModelManifestReq latest_request;
    latest_request.mutable_requested_model()->set_model_lineage_id(
        manifest.identity().model_lineage_id());
    Service(latest_request.mutable_requester(), "aiserver", "aiserver-0");
    latest_request.set_latest_in_lineage(true);
    rl::training::v1::GetModelManifestRsp latest_response;
    grpc::ClientContext latest_context;
    if (!stub->GetModelManifest(&latest_context, latest_request,
                                &latest_response)
             .ok() ||
        latest_response.ret_code() != 0 ||
        !latest_response.has_available_floor_model_step() ||
        latest_response.available_floor_model_step() != 0 ||
        !latest_response.has_latest_available_model_step() ||
        latest_response.latest_available_model_step() != 0 ||
        latest_response.manifest().identity().SerializeAsString() !=
            manifest.identity().SerializeAsString()) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("latest lineage lookup failed");
    }

    rl::training::v1::GetModelManifestReq step_request;
    step_request.mutable_requested_model()->set_model_lineage_id(
        manifest.identity().model_lineage_id());
    step_request.mutable_requested_model()->set_model_step(
        manifest.identity().model_step());
    Service(step_request.mutable_requester(), "aiserver", "aiserver-0");
    rl::training::v1::GetModelManifestRsp step_response;
    grpc::ClientContext step_context;
    if (!stub->GetModelManifest(&step_context, step_request,
                                &step_response)
             .ok() ||
        step_response.ret_code() != 0 ||
        step_response.manifest().identity().SerializeAsString() !=
            manifest.identity().SerializeAsString()) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("lineage/step selector lookup failed");
    }

    rl::training::v1::GetModelManifestReq missing_step_request;
    missing_step_request.mutable_requested_model()->set_model_lineage_id(
        manifest.identity().model_lineage_id());
    Service(missing_step_request.mutable_requester(), "aiserver",
            "aiserver-0");
    rl::training::v1::GetModelManifestRsp missing_step_response;
    grpc::ClientContext missing_step_context;
    if (!stub->GetModelManifest(&missing_step_context, missing_step_request,
                                &missing_step_response)
             .ok() ||
        missing_step_response.ret_code() == 0) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("exact selector without model step was accepted");
    }

    rl::training::v1::DownloadModelReq download_request;
    *download_request.mutable_requested_model() = manifest.identity();
    Service(download_request.mutable_requester(), "aiserver", "aiserver-0");
    grpc::ClientContext download_context;
    auto reader = stub->DownloadModel(&download_context, download_request);
    rl::training::v1::ModelChunk chunk;
    int64_t expected_offset = 0;
    std::string reconstructed;
    while (reader->Read(&chunk)) {
        if (chunk.offset() != expected_offset ||
            chunk.model().SerializeAsString() !=
                manifest.identity().SerializeAsString()) {
            server->Shutdown();
            fs::remove_all(root);
            return Fail("DownloadModel chunk identity or offset invalid");
        }
        reconstructed.append(chunk.data());
        expected_offset += static_cast<int64_t>(chunk.data().size());
    }
    if (!reader->Finish().ok() || reconstructed != payload) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("DownloadModel payload reconstruction failed");
    }

    rl::training::v1::GetModelManifestReq wrong_request;
    *wrong_request.mutable_requested_model() = manifest.identity();
    wrong_request.mutable_requested_model()
        ->mutable_artifact_digest()
        ->set_hex(std::string(64, 'f'));
    Service(wrong_request.mutable_requester(), "aiserver", "aiserver-0");
    rl::training::v1::GetModelManifestRsp wrong_response;
    grpc::ClientContext wrong_context;
    if (!stub->GetModelManifest(&wrong_context, wrong_request, &wrong_response)
             .ok() ||
        wrong_response.ret_code() == 0) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("digest-mismatched identity was returned");
    }

    rl::training::v1::GetModelManifestReq incomplete_request;
    incomplete_request.mutable_requested_model()->set_model_lineage_id(
        manifest.identity().model_lineage_id());
    incomplete_request.mutable_requested_model()->set_model_step(
        manifest.identity().model_step());
    *incomplete_request.mutable_requested_model()->mutable_artifact_digest() =
        manifest.identity().artifact_digest();
    Service(incomplete_request.mutable_requester(), "aiserver", "aiserver-0");
    rl::training::v1::GetModelManifestRsp incomplete_response;
    grpc::ClientContext incomplete_context;
    if (!stub->GetModelManifest(&incomplete_context, incomplete_request,
                                &incomplete_response)
             .ok() ||
        incomplete_response.ret_code() == 0) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("incomplete digest selector was accepted");
    }

    for (uint64_t step = 1; step <= 2; ++step) {
        const fs::path next_path = ModelPath(root, step);
        {
            std::ofstream output(next_path, std::ios::binary);
            output << "model-step-" << step;
        }
        std::string next_checksum;
        if (!ComputeSha256(next_path, &next_checksum, &error)) {
            server->Shutdown();
            fs::remove_all(root);
            return Fail(error);
        }
        latest_registered = Manifest(next_path, next_checksum, step);
        rl::training::v1::RegisterModelReq next_request;
        *next_request.mutable_manifest() = latest_registered;
        rl::training::v1::RegisterModelRsp next_response;
        grpc::ClientContext next_context;
        if (!stub->RegisterModel(&next_context, next_request, &next_response)
                 .ok() ||
            next_response.result() !=
                rl::training::v1::MODEL_REGISTER_RESULT_REGISTERED) {
            server->Shutdown();
            fs::remove_all(root);
            return Fail("newer model registration failed");
        }
    }

    rl::training::v1::AckModelReq ack_request;
    Service(ack_request.mutable_aiserver(), "aiserver", "aiserver-0");
    *ack_request.mutable_model() = manifest.identity();
    ack_request.set_load_instance_id("load-0");
    ack_request.set_load_status(rl::training::v1::MODEL_LOAD_STATUS_LOADED);
    rl::training::v1::AckModelRsp first_ack;
    rl::training::v1::AckModelRsp second_ack;
    grpc::ClientContext first_ack_context;
    grpc::ClientContext second_ack_context;
    if (!stub->AckModel(&first_ack_context, ack_request, &first_ack).ok() ||
        !stub->AckModel(&second_ack_context, ack_request, &second_ack).ok() ||
        first_ack.result() != rl::training::v1::MODEL_ACK_RESULT_APPLIED ||
        second_ack.result() !=
            rl::training::v1::MODEL_ACK_RESULT_ALREADY_APPLIED) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("AckModel RPC was not idempotent");
    }

    rl::training::v1::ModelDistributorStatusReq status_request;
    rl::training::v1::ModelDistributorStatusRsp status_response;
    grpc::ClientContext status_context;
    if (!stub->GetModelDistributorStatus(&status_context, status_request,
                                         &status_response)
             .ok() ||
        !status_response.ready() ||
        status_response.latest_model().SerializeAsString() !=
            latest_registered.identity().SerializeAsString() ||
        !status_response.has_available_floor_model_step() ||
        status_response.available_floor_model_step() != 0 ||
        !status_response.has_latest_available_model_step() ||
        status_response.latest_available_model_step() != 2 ||
        status_response.loaded_ack_count() != 1) {
        server->Shutdown();
        fs::remove_all(root);
        return Fail("status RPC did not preserve model/ACK identity");
    }

    server->Shutdown();
    fs::remove_all(root);
    std::cout << "model_distributor_rpc_contract: PASS" << std::endl;
    return 0;
}
