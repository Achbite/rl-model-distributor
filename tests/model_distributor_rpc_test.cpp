#include "grpc/model_distributor_service.h"
#include "test_helpers.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <grpcpp/grpcpp.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>

namespace {

void Require(bool condition, const std::string& message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
}

std::filesystem::path ModelPath(const std::filesystem::path& root) {
    const auto path = root / "published" / "policy.bundle";
    std::filesystem::create_directories(path.parent_path());
    return path;
}

class TemporaryRoot {
public:
    TemporaryRoot()
        : path_(std::filesystem::temp_directory_path() /
                ("model-distributor-development-" +
                 std::to_string(::getpid()))) {
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }

    ~TemporaryRoot() { std::filesystem::remove_all(path_); }

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

void TestRegisterManifestDownloadAck() {
    using namespace test_helpers;

    TemporaryRoot temporary_root;
    const auto model_path = ModelPath(temporary_root.path());
    const std::string model_bytes = "fixed-model-artifact-bytes";
    {
        std::ofstream output(model_path, std::ios::binary);
        Require(output.is_open(), "open fixed model artifact");
        output.write(model_bytes.data(),
                     static_cast<std::streamsize>(model_bytes.size()));
        Require(output.good(), "write fixed model artifact");
    }

    const auto manifest = Manifest(model_path);

    ModelDistributorServiceImpl service(Config(temporary_root.path()));
    int port = 0;
    grpc::ServerBuilder builder;
    builder.AddListeningPort(
        "127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    Require(server != nullptr && port > 0,
            "start local ModelDistributor test service");

    auto channel = grpc::CreateChannel(
        "127.0.0.1:" + std::to_string(port),
        grpc::InsecureChannelCredentials());
    auto stub = rl::training::v1::ModelDistributorService::NewStub(channel);

    rl::training::v1::RegisterModelReq register_request;
    *register_request.mutable_manifest() = manifest;
    register_request.set_local_artifact_path(model_path.string());
    rl::training::v1::RegisterModelRsp register_response;
    grpc::ClientContext register_context;
    const grpc::Status register_status = stub->RegisterModel(
        &register_context, register_request, &register_response);
    Require(register_status.ok() &&
                register_response.result() ==
                    rl::training::v1::MODEL_REGISTER_RESULT_REGISTERED &&
                register_response.manifest().identity().SerializeAsString() ==
                    manifest.identity().SerializeAsString(),
            "RegisterModel preserves the published model identity");

    rl::training::v1::GetModelManifestReq manifest_request;
    *manifest_request.mutable_requested_model() = manifest.identity();
    Service(manifest_request.mutable_requester(),
            "aiserver", "aiserver-fixed");
    rl::training::v1::GetModelManifestRsp manifest_response;
    grpc::ClientContext manifest_context;
    const grpc::Status manifest_status = stub->GetModelManifest(
        &manifest_context, manifest_request, &manifest_response);
    Require(manifest_status.ok() &&
                manifest_response.result() ==
                    rl::training::v1::MODEL_LOOKUP_RESULT_FOUND &&
                manifest_response.manifest().identity().SerializeAsString() ==
                    manifest.identity().SerializeAsString(),
            "GetModelManifest returns the registered object identity");

    rl::training::v1::DownloadModelReq download_request;
    *download_request.mutable_requested_model() = manifest.identity();
    Service(download_request.mutable_requester(),
            "aiserver", "aiserver-fixed");
    grpc::ClientContext download_context;
    auto reader = stub->DownloadModel(&download_context, download_request);
    std::string downloaded_bytes;
    int64_t expected_offset = 0;
    rl::training::v1::ModelChunk chunk;
    while (reader->Read(&chunk)) {
        Require(chunk.offset() == expected_offset &&
                    chunk.model().SerializeAsString() ==
                        manifest.identity().SerializeAsString(),
                "DownloadModel chunk belongs to the registered model");
        downloaded_bytes.append(chunk.data());
        expected_offset += static_cast<int64_t>(chunk.data().size());
    }
    Require(reader->Finish().ok() && downloaded_bytes == model_bytes,
            "DownloadModel returns the registered model bytes");

    rl::training::v1::AckModelReq ack_request;
    Service(ack_request.mutable_aiserver(),
            "aiserver", "aiserver-fixed");
    *ack_request.mutable_model() = manifest.identity();
    ack_request.set_load_instance_id("load-fixed");
    ack_request.set_load_status(rl::training::v1::MODEL_LOAD_STATUS_LOADED);
    rl::training::v1::AckModelRsp ack_response;
    grpc::ClientContext ack_context;
    const grpc::Status ack_status =
        stub->AckModel(&ack_context, ack_request, &ack_response);
    Require(ack_status.ok() &&
                ack_response.result() ==
                    rl::training::v1::MODEL_ACK_RESULT_APPLIED,
            "AckModel settles the exact downloaded model identity");

    rl::training::v1::AckModelReq second_load_request = ack_request;
    second_load_request.set_load_instance_id("load-reused-bytes");
    rl::training::v1::AckModelRsp second_load_response;
    grpc::ClientContext second_load_context;
    const grpc::Status second_load_status = stub->AckModel(
        &second_load_context, second_load_request, &second_load_response);
    Require(second_load_status.ok() &&
                second_load_response.result() ==
                    rl::training::v1::MODEL_ACK_RESULT_APPLIED,
            "AckModel records each exact load instance independently");

    rl::training::v1::AckModelRsp repeated_load_response;
    grpc::ClientContext repeated_load_context;
    const grpc::Status repeated_load_status = stub->AckModel(
        &repeated_load_context, second_load_request, &repeated_load_response);
    Require(repeated_load_status.ok() &&
                repeated_load_response.result() ==
                    rl::training::v1::MODEL_ACK_RESULT_ALREADY_APPLIED,
            "AckModel is idempotent for one load instance");

    // Another actual load attempt can fail; Status must preserve its origin.
    auto failed = ack_request;
    failed.set_load_instance_id("load-prepare-failed");
    failed.set_load_status(rl::training::v1::MODEL_LOAD_STATUS_FAILED);
    failed.set_message("prepare: observation dimension mismatch in bootstrap model");
    rl::training::v1::AckModelRsp failed_response;
    grpc::ClientContext failed_context;
    Require(stub->AckModel(&failed_context, failed, &failed_response).ok() &&
                failed_response.result() == rl::training::v1::MODEL_ACK_RESULT_APPLIED,
            "actual registry accepts a definite FAILED load result");
    rl::training::v1::ModelDistributorStatusRsp observed;
    grpc::ClientContext status_context;
    Require(stub->GetModelDistributorStatus(&status_context, {}, &observed).ok() &&
                observed.latest_ack_status() == rl::training::v1::MODEL_LOAD_STATUS_FAILED &&
                observed.latest_ack_model().SerializeAsString() == failed.model().SerializeAsString() &&
                observed.latest_ack_aiserver().SerializeAsString() == failed.aiserver().SerializeAsString() &&
                observed.last_error() == failed.message(),
            "Status delivers FAILED identity, producer and original error to its consumer");

    server->Shutdown();
    server->Wait();
}

}  // namespace

int main() {
    TestRegisterManifestDownloadAck();
    std::cout << "model_distributor_data_path: PASS" << std::endl;
    return 0;
}
