#include "test_helpers.h"

#include <filesystem>
#include <array>
#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>

namespace {

int Fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

bool WriteModel(const std::filesystem::path& path,
                const std::string& payload,
                std::string* digest) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output << payload;
    output.close();
    std::string error;
    return ComputeSha256(path, digest, &error);
}

std::filesystem::path ModelPath(const std::filesystem::path& root,
                                uint64_t version) {
    std::ostringstream directory;
    directory << std::setfill('0') << std::setw(6) << version;
    return root / "archive" / directory.str() / "SaveModel.onnx";
}

}  // namespace

int main() {
    namespace fs = std::filesystem;
    using namespace test_helpers;
    const fs::path root = fs::temp_directory_path() /
                          ("model-registry-test-" + std::to_string(::getpid()));
    const fs::path model_path = ModelPath(root, 0);
    std::string checksum;
    if (!WriteModel(model_path, "deterministic-model", &checksum)) {
        return Fail("failed to create model artifact");
    }
    ModelRegistry registry(Config(root));
    RegisteredModel record;
    std::string message;
    auto manifest = Manifest(model_path, checksum);
    if (registry.Register(manifest, &record, &message) !=
        rl::training::v1::MODEL_REGISTER_RESULT_REGISTERED) {
        return Fail("initial register failed: " + message);
    }
    if (registry.Register(manifest, &record, &message) !=
        rl::training::v1::MODEL_REGISTER_RESULT_ALREADY_REGISTERED) {
        return Fail("exact manifest retry was not idempotent");
    }

    auto manifest_conflict = manifest;
    manifest_conflict.set_seed(1);
    RefreshManifestDigest(&manifest_conflict);
    if (registry.Register(manifest_conflict, &record, &message) !=
        rl::training::v1::MODEL_REGISTER_RESULT_REJECTED_CONFLICT) {
        return Fail("lineage/version manifest conflict was accepted");
    }
    auto stale_contract = manifest;
    stale_contract.mutable_contract()->set_package_version("0.7.0");
    RefreshManifestDigest(&stale_contract);
    if (registry.Register(stale_contract, &record, &message) !=
        rl::training::v1::MODEL_REGISTER_RESULT_REJECTED_INVALID) {
        return Fail("old contract identity was accepted");
    }
    auto bad_manifest_digest = manifest;
    bad_manifest_digest.mutable_identity()->mutable_manifest_digest()->set_hex(
        std::string(64, '0'));
    if (registry.Register(bad_manifest_digest, &record, &message) !=
        rl::training::v1::MODEL_REGISTER_RESULT_REJECTED_INVALID) {
        return Fail("incorrect canonical manifest digest was accepted");
    }
    auto outside = manifest;
    outside.set_artifact_uri("file:///tmp/outside.onnx");
    RefreshManifestDigest(&outside);
    if (registry.Register(outside, &record, &message) !=
        rl::training::v1::MODEL_REGISTER_RESULT_REJECTED_INVALID) {
        return Fail("outside artifact path was accepted");
    }
    const fs::path noncanonical_path =
        root / "runtime" / "000000" / "SaveModel.onnx";
    std::string noncanonical_digest;
    if (!WriteModel(noncanonical_path, "noncanonical",
                    &noncanonical_digest)) {
        return Fail("failed to create noncanonical model artifact");
    }
    auto noncanonical = Manifest(noncanonical_path, noncanonical_digest);
    if (registry.Register(noncanonical, &record, &message) !=
        rl::training::v1::MODEL_REGISTER_RESULT_REJECTED_INVALID) {
        return Fail("non-archive model publication path was accepted");
    }

    if (!registry.Get(manifest.identity(), &record, &message) ||
        record.manifest.identity().artifact_digest().hex() != checksum) {
        return Fail("exact registered identity lookup failed");
    }
    const RegisteredModel held_version_zero = record;
    auto wrong_identity = manifest.identity();
    wrong_identity.mutable_artifact_digest()->set_hex(std::string(64, 'f'));
    if (registry.Get(wrong_identity, &record, &message)) {
        return Fail("same version with wrong digest was returned");
    }
    if (!registry.GetLatest("maze-fixed-map-seed-0", &record, &message) ||
        record.manifest.identity().model_version() != 0) {
        return Fail("lineage latest lookup failed");
    }

    rl::training::v1::AckModelReq ack;
    Service(ack.mutable_aiserver(), "aiserver", "aiserver-0");
    *ack.mutable_model() = manifest.identity();
    ack.set_load_instance_id("load-0");
    ack.set_load_status(rl::training::v1::MODEL_LOAD_STATUS_LOADED);
    ack.set_message("loaded");
    rl::common::v1::ServiceInstanceIdentity requester;
    Service(&requester, "aiserver", "aiserver-0");
    registry.RecordDownloadSuccess(requester, manifest.identity(), 19);
    if (registry.Ack(ack, &message) !=
        rl::training::v1::MODEL_ACK_RESULT_APPLIED) {
        return Fail("initial model ACK was rejected");
    }
    auto retry = ack;
    retry.set_message("diagnostic text may differ on retry");
    if (registry.Ack(retry, &message) !=
        rl::training::v1::MODEL_ACK_RESULT_ALREADY_APPLIED) {
        return Fail("same load identity was not idempotent");
    }
    auto ack_conflict = ack;
    ack_conflict.set_load_status(rl::training::v1::MODEL_LOAD_STATUS_FAILED);
    if (registry.Ack(ack_conflict, &message) !=
        rl::training::v1::MODEL_ACK_RESULT_REJECTED) {
        return Fail("load idempotency conflict was accepted");
    }

    std::mutex download_mutex;
    std::condition_variable download_cv;
    bool download_holds_handle = false;
    bool resume_download = false;
    ssize_t held_count = -1;
    std::string held_payload;
    std::thread in_flight_download([&] {
        {
            std::unique_lock<std::mutex> lock(download_mutex);
            download_holds_handle = true;
            download_cv.notify_one();
            download_cv.wait(lock, [&] { return resume_download; });
        }
        std::array<char, 64> held_bytes{};
        held_count = ::pread(
            held_version_zero.artifact->descriptor, held_bytes.data(),
            held_bytes.size(), 0);
        if (held_count > 0) {
            held_payload.assign(held_bytes.data(),
                                static_cast<size_t>(held_count));
            registry.RecordDownloadSuccess(requester, manifest.identity(),
                                           held_count);
        }
    });
    {
        std::unique_lock<std::mutex> lock(download_mutex);
        download_cv.wait(lock, [&] { return download_holds_handle; });
    }

    rl::training::v1::ModelIdentity version_one_identity;
    for (uint64_t version = 1; version <= 101; ++version) {
        const fs::path path = ModelPath(root, version);
        std::string digest;
        if (!WriteModel(path, "model-version-" + std::to_string(version),
                        &digest)) {
            return Fail("failed to create retained model");
        }
        auto next = Manifest(path, digest, version);
        if (registry.Register(next, &record, &message) !=
            rl::training::v1::MODEL_REGISTER_RESULT_REGISTERED) {
            return Fail("retained version register failed: " + message);
        }
        if (version == 1) {
            version_one_identity = next.identity();
            registry.RecordDownloadSuccess(requester, next.identity(),
                                           next.size_bytes());
        }
    }
    if (registry.Get(manifest.identity(), &record, &message)) {
        return Fail("old exact model was not evicted");
    }
    if (!registry.GetLatest("maze-fixed-map-seed-0", &record, &message) ||
        record.manifest.identity().model_version() != 101) {
        return Fail("latest lineage identity is incorrect after retention");
    }
    if (!registry.GetByVersion("maze-fixed-map-seed-0", 1,
                               &record, &message) ||
        record.manifest.identity().model_version() != 1) {
        return Fail("retained model version selector failed");
    }
    if (registry.GetByVersion("maze-fixed-map-seed-0", 0,
                              &record, &message)) {
        return Fail("evicted model version selector unexpectedly succeeded");
    }
    if (registry.Ack(ack, &message) !=
        rl::training::v1::MODEL_ACK_RESULT_NOT_FOUND) {
        return Fail("expired ACK fact outlived serving retention");
    }
    fs::remove_all(model_path.parent_path());
    {
        std::lock_guard<std::mutex> lock(download_mutex);
        resume_download = true;
    }
    download_cv.notify_one();
    in_flight_download.join();
    if (held_count != static_cast<ssize_t>(
                          std::string("deterministic-model").size()) ||
        held_payload != "deterministic-model") {
        return Fail("in-flight open model handle did not survive eviction");
    }

    rl::training::v1::AckModelReq delayed_ack;
    *delayed_ack.mutable_aiserver() = requester;
    *delayed_ack.mutable_model() = manifest.identity();
    delayed_ack.set_load_instance_id("load-after-serving-eviction");
    delayed_ack.set_load_status(
        rl::training::v1::MODEL_LOAD_STATUS_LOADED);
    if (registry.Ack(delayed_ack, &message) !=
        rl::training::v1::MODEL_ACK_RESULT_APPLIED) {
        return Fail("completed download ACK was lost after serving eviction");
    }
    if (registry.Ack(delayed_ack, &message) !=
        rl::training::v1::MODEL_ACK_RESULT_ALREADY_APPLIED) {
        return Fail("evicted model ACK retry was not idempotent");
    }
    if (registry.Ack(ack, &message) !=
        rl::training::v1::MODEL_ACK_RESULT_ALREADY_APPLIED) {
        return Fail("same model settlement diverged by load instance");
    }
    rl::training::v1::AckModelReq version_one_ack;
    *version_one_ack.mutable_aiserver() = requester;
    *version_one_ack.mutable_model() = version_one_identity;
    version_one_ack.set_load_instance_id("load-version-one");
    version_one_ack.set_load_status(
        rl::training::v1::MODEL_LOAD_STATUS_LOADED);
    if (registry.Ack(version_one_ack, &message) !=
        rl::training::v1::MODEL_ACK_RESULT_APPLIED) {
        return Fail("same requester lost a distinct model download fact");
    }
    rl::training::v1::ModelDistributorStatusRsp status;
    registry.GetStatus(&status);
    uint64_t floor = 0;
    uint64_t latest = 0;
    if (!registry.GetAvailableRange("maze-fixed-map-seed-0", &floor,
                                    &latest, &message) ||
        floor != 1 || latest != 101) {
        return Fail("actual contiguous model range is incorrect");
    }
    if (!status.ready() || status.registered_model_count() != 101 ||
        status.latest_model().model_version() != 101 ||
        !status.has_available_floor_model_version() ||
        status.available_floor_model_version() != 1 ||
        !status.has_latest_available_model_version() ||
        status.latest_available_model_version() != 101 ||
        status.loaded_ack_count() != 3 ||
        status.duplicate_register_count() != 1 ||
        status.rejected_register_count() != 5 ||
        status.contract().package_version() != "0.11.0" ||
        status.distributor().component() != "model-distributor") {
        return Fail("status identity or counters do not match operations");
    }

    ModelRegistry boundary_registry(Config(root));
    const uint64_t max_version = std::numeric_limits<uint64_t>::max();
    for (uint64_t version = max_version - 100;; ++version) {
        const fs::path path = ModelPath(root, version);
        std::string digest;
        if (!WriteModel(path, "boundary-" + std::to_string(version),
                        &digest)) {
            return Fail("failed to create uint64 boundary model");
        }
        const auto boundary = Manifest(
            path, digest, version, "uint64-boundary-lineage");
        if (boundary_registry.Register(boundary, &record, &message) !=
            rl::training::v1::MODEL_REGISTER_RESULT_REGISTERED) {
            return Fail("valid uint64 retention boundary was rejected");
        }
        if (version == max_version) break;
    }
    const uint64_t rejected_version = max_version - 101;
    const fs::path rejected_path = ModelPath(root, rejected_version);
    std::string rejected_digest;
    if (!WriteModel(rejected_path, "outside-boundary", &rejected_digest)) {
        return Fail("failed to create rejected boundary model");
    }
    const auto rejected_boundary = Manifest(
        rejected_path, rejected_digest, rejected_version,
        "uint64-boundary-lineage");
    if (boundary_registry.Register(rejected_boundary, &record, &message) !=
        rl::training::v1::MODEL_REGISTER_RESULT_REJECTED_INVALID) {
        return Fail("uint64 model outside retention was accepted");
    }
    uint64_t boundary_floor = 0;
    uint64_t boundary_latest = 0;
    if (!boundary_registry.GetAvailableRange(
            "uint64-boundary-lineage", &boundary_floor, &boundary_latest,
            &message) ||
        boundary_floor != max_version - 100 ||
        boundary_latest != max_version) {
        return Fail("uint64 contiguous range is incorrect");
    }

    const fs::path gap_path = ModelPath(root, 1000000);
    std::string gap_digest;
    if (!WriteModel(gap_path, "gap-version", &gap_digest)) {
        return Fail("failed to create gap model");
    }
    const auto gap = Manifest(
        gap_path, gap_digest, 1000000, "maze-fixed-map-seed-0");
    if (registry.Register(gap, &record, &message) !=
        rl::training::v1::MODEL_REGISTER_RESULT_REJECTED_INVALID) {
        return Fail("non-contiguous model publication was accepted");
    }
    fs::remove_all(root);
    std::cout << "model_registry_contract: PASS" << std::endl;
    return 0;
}
