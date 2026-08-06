#include "test_helpers.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

namespace {

int Fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

bool WriteModel(const std::filesystem::path& path,
                const std::string& payload,
                std::string* digest) {
    std::ofstream output(path, std::ios::binary);
    output << payload;
    output.close();
    std::string error;
    return ComputeSha256(path, digest, &error);
}

}  // namespace

int main() {
    namespace fs = std::filesystem;
    using namespace test_helpers;
    const fs::path root = fs::temp_directory_path() /
                          ("model-registry-test-" + std::to_string(::getpid()));
    fs::create_directories(root / "serving");
    const fs::path model_path = root / "serving/model_v000000.onnx";
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

    if (!registry.Get(manifest.identity(), &record, &message) ||
        record.manifest.identity().artifact_digest().hex() != checksum) {
        return Fail("exact registered identity lookup failed");
    }
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

    for (uint64_t version = 1; version <= 2; ++version) {
        const fs::path path = root / "serving" /
                              ("model_v00000" + std::to_string(version) +
                               ".onnx");
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
    }
    if (registry.Get(manifest.identity(), &record, &message)) {
        return Fail("old exact model was not evicted");
    }
    if (!registry.GetLatest("maze-fixed-map-seed-0", &record, &message) ||
        record.manifest.identity().model_version() != 2) {
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

    rl::training::v1::ModelDistributorStatusRsp status;
    registry.GetStatus(&status);
    if (!status.ready() || status.registered_model_count() != 2 ||
        status.latest_model().model_version() != 2 ||
        status.loaded_ack_count() != 1 ||
        status.duplicate_register_count() != 1 ||
        status.rejected_register_count() != 4 ||
        status.contract().package_version() != "0.8.0" ||
        status.distributor().component() != "model-distributor") {
        return Fail("status identity or counters do not match operations");
    }
    fs::remove_all(root);
    std::cout << "model_registry_contract: PASS" << std::endl;
    return 0;
}
