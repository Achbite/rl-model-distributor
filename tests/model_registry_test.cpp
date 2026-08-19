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
                                uint64_t model_step) {
    std::ostringstream directory;
    directory << std::setfill('0') << std::setw(7) << model_step;
    return root / directory.str() / "SaveModel.onnx";
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
        return Fail("lineage/step manifest conflict was accepted");
    }
    auto stale_contract = manifest;
    stale_contract.mutable_contract()->set_package_version("0.12.0");
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
    auto missing_step = manifest;
    missing_step.mutable_identity()->clear_model_step();
    RefreshManifestDigest(&missing_step);
    if (registry.Register(missing_step, &record, &message) !=
        rl::training::v1::MODEL_REGISTER_RESULT_REJECTED_INVALID) {
        return Fail("missing model step was accepted");
    }
    auto mismatched_update = manifest;
    mismatched_update.set_train_updates(1);
    RefreshManifestDigest(&mismatched_update);
    if (registry.Register(mismatched_update, &record, &message) !=
        rl::training::v1::MODEL_REGISTER_RESULT_REJECTED_INVALID) {
        return Fail("model step and train_updates mismatch was accepted");
    }
    auto outside = manifest;
    outside.set_artifact_uri("file:///tmp/outside.onnx");
    RefreshManifestDigest(&outside);
    if (registry.Register(outside, &record, &message) !=
        rl::training::v1::MODEL_REGISTER_RESULT_REJECTED_INVALID) {
        return Fail("outside artifact path was accepted");
    }
    const fs::path noncanonical_path =
        root / "000000" / "SaveModel.onnx";
    std::string noncanonical_digest;
    if (!WriteModel(noncanonical_path, "noncanonical",
                    &noncanonical_digest)) {
        return Fail("failed to create noncanonical model artifact");
    }
    auto noncanonical = Manifest(noncanonical_path, noncanonical_digest);
    if (registry.Register(noncanonical, &record, &message) !=
        rl::training::v1::MODEL_REGISTER_RESULT_REJECTED_INVALID) {
        return Fail("noncanonical model publication path was accepted");
    }

    if (!registry.Get(manifest.identity(), &record, &message) ||
        record.manifest.identity().artifact_digest().hex() != checksum) {
        return Fail("exact registered identity lookup failed");
    }
    const RegisteredModel held_version_zero = record;
    auto wrong_identity = manifest.identity();
    wrong_identity.mutable_artifact_digest()->set_hex(std::string(64, 'f'));
    if (registry.Get(wrong_identity, &record, &message)) {
        return Fail("same step with wrong digest was returned");
    }
    if (!registry.GetLatest("maze-fixed-map-seed-0", &record, &message) ||
        record.manifest.identity().model_step() != 0) {
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

    const fs::path nonzero_initial_path = ModelPath(root, 1);
    std::string nonzero_initial_digest;
    if (!WriteModel(nonzero_initial_path, "nonzero-initial",
                    &nonzero_initial_digest)) {
        return Fail("failed to create nonzero initial model");
    }
    ModelRegistry nonzero_initial_registry(Config(root));
    const auto nonzero_initial = Manifest(
        nonzero_initial_path, nonzero_initial_digest, 1,
        "nonzero-initial-lineage");
    if (nonzero_initial_registry.Register(nonzero_initial, &record, &message) !=
        rl::training::v1::MODEL_REGISTER_RESULT_REJECTED_INVALID) {
        return Fail("a lineage beginning above model step 0 was accepted");
    }

    rl::training::v1::ModelIdentity step_one_identity;
    for (uint64_t step = 1; step <= 101; ++step) {
        const fs::path path = ModelPath(root, step);
        std::string digest;
        if (!WriteModel(path, "model-step-" + std::to_string(step),
                        &digest)) {
            return Fail("failed to create retained model");
        }
        auto next = Manifest(path, digest, step);
        if (registry.Register(next, &record, &message) !=
            rl::training::v1::MODEL_REGISTER_RESULT_REGISTERED) {
            return Fail("retained step register failed: " + message);
        }
        if (step == 1) {
            step_one_identity = next.identity();
            registry.RecordDownloadSuccess(requester, next.identity(),
                                           next.size_bytes());
        }
    }
    if (registry.Get(manifest.identity(), &record, &message)) {
        return Fail("old exact model was not evicted");
    }
    if (!registry.GetLatest("maze-fixed-map-seed-0", &record, &message) ||
        record.manifest.identity().model_step() != 101) {
        return Fail("latest lineage identity is incorrect after retention");
    }
    if (!registry.GetByStep("maze-fixed-map-seed-0", 1,
                            &record, &message) ||
        record.manifest.identity().model_step() != 1) {
        return Fail("retained model step selector failed");
    }
    if (registry.GetByStep("maze-fixed-map-seed-0", 0,
                           &record, &message)) {
        return Fail("evicted model step selector unexpectedly succeeded");
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
    rl::training::v1::AckModelReq step_one_ack;
    *step_one_ack.mutable_aiserver() = requester;
    *step_one_ack.mutable_model() = step_one_identity;
    step_one_ack.set_load_instance_id("load-step-one");
    step_one_ack.set_load_status(
        rl::training::v1::MODEL_LOAD_STATUS_LOADED);
    if (registry.Ack(step_one_ack, &message) !=
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
        status.latest_model().model_step() != 101 ||
        !status.has_available_floor_model_step() ||
        status.available_floor_model_step() != 1 ||
        !status.has_latest_available_model_step() ||
        status.latest_available_model_step() != 101 ||
        status.loaded_ack_count() != 3 ||
        status.duplicate_register_count() != 1 ||
        status.rejected_register_count() != 7 ||
        status.contract().package_version() != "0.13.0" ||
        status.distributor().component() != "model-distributor") {
        return Fail("status identity or counters do not match operations");
    }

    const fs::path gap_path = ModelPath(root, 1000000);
    std::string gap_digest;
    if (!WriteModel(gap_path, "gap-step", &gap_digest)) {
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
