#include "store/model_registry.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

namespace {

int Fail(const std::string& message) {
    std::cerr << message << std::endl;
    return 1;
}

maze::ModelArtifactManifest Manifest(const std::filesystem::path& path,
                                     const std::string& checksum,
                                     int version = 0) {
    maze::ModelArtifactManifest manifest;
    manifest.set_schema_version(1);
    manifest.set_contract_version("0.6.0");
    manifest.set_model_version(version);
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
    const auto root = std::filesystem::temp_directory_path() /
                      ("model-registry-test-" + std::to_string(::getpid()));
    std::filesystem::create_directories(root / "serving");
    const auto model_path = root / "serving/model_v000000.onnx";
    {
        std::ofstream output(model_path, std::ios::binary);
        output << "deterministic-model";
    }
    std::string checksum;
    std::string error;
    if (!ComputeSha256(model_path, &checksum, &error)) return Fail(error);

    ModelDistributorConfig config;
    config.artifact_root = root.string();
    config.contract_version = "0.6.0";
    config.retention_versions = 2;
    ModelRegistry registry(config);

    RegisteredModel record;
    std::string message;
    auto manifest = Manifest(model_path, checksum);
    if (registry.Register(manifest, &record, &message) !=
        maze::MODEL_REGISTER_RESULT_REGISTERED) {
        return Fail("initial register failed: " + message);
    }
    if (registry.Register(manifest, &record, &message) !=
        maze::MODEL_REGISTER_RESULT_ALREADY_REGISTERED) {
        return Fail("duplicate register was not idempotent");
    }
    auto equivalent = manifest;
    equivalent.set_published_ts_ms(2);
    equivalent.set_artifact_uri(
        "file://" + (root / "serving/../serving/model_v000000.onnx").string());
    if (registry.Register(equivalent, &record, &message) !=
        maze::MODEL_REGISTER_RESULT_ALREADY_REGISTERED) {
        return Fail("equivalent model identity was not idempotent");
    }
    const auto replacement_path =
        root / "serving/model_v000000_replacement.onnx";
    {
        std::ofstream output(replacement_path, std::ios::binary);
        output << "different-model";
    }
    std::string replacement_checksum;
    if (!ComputeSha256(
            replacement_path, &replacement_checksum, &error)) {
        return Fail(error);
    }
    auto checksum_conflict = manifest;
    checksum_conflict.set_artifact_uri(
        "file://" + replacement_path.string());
    checksum_conflict.set_model_file(
        replacement_path.filename().string());
    checksum_conflict.set_size_bytes(
        static_cast<int64_t>(
            std::filesystem::file_size(replacement_path)));
    checksum_conflict.set_sha256(replacement_checksum);
    if (registry.Register(checksum_conflict, &record, &message) !=
        maze::MODEL_REGISTER_RESULT_REJECTED_CONFLICT) {
        return Fail("same-version checksum conflict was accepted");
    }
    auto conflicting = manifest;
    conflicting.set_seed(1);
    if (registry.Register(conflicting, &record, &message) !=
        maze::MODEL_REGISTER_RESULT_REJECTED_CONFLICT) {
        return Fail("conflicting register was accepted");
    }

    auto outside = manifest;
    outside.set_artifact_uri("file:///tmp/outside.onnx");
    if (registry.Register(outside, &record, &message) !=
        maze::MODEL_REGISTER_RESULT_REJECTED_INVALID) {
        return Fail("outside artifact path was accepted");
    }

    if (!registry.Get(0, &record, &message) ||
        record.manifest.sha256() != checksum) {
        return Fail("registered model lookup failed");
    }
    if (!registry.GetLatest(&record, &message) ||
        record.manifest.model_version() != 0) {
        return Fail("latest registered model lookup failed");
    }
    if (registry.Get(99, &record, &message)) {
        return Fail("unknown model lookup succeeded");
    }

    maze::AckModelReq ack;
    ack.set_aiserver_id("aiserver-0");
    ack.set_model_version(0);
    ack.set_sha256(checksum);
    ack.set_load_status(maze::MODEL_LOAD_STATUS_LOADED);
    ack.set_message("loaded");
    if (registry.Ack(ack, &message) != maze::MODEL_ACK_RESULT_APPLIED) {
        return Fail("initial model ACK was rejected");
    }
    auto equivalent_ack = ack;
    equivalent_ack.set_message("loaded after retry");
    if (registry.Ack(equivalent_ack, &message) !=
        maze::MODEL_ACK_RESULT_ALREADY_APPLIED) {
        return Fail("model ACK was not idempotent");
    }

    maze::ModelDistributorStatusRsp status;
    registry.GetStatus(&status);
    if (!status.ready() || status.registered_model_count() != 1 ||
        status.loaded_ack_count() != 1 ||
        status.duplicate_register_count() != 2 ||
        status.rejected_register_count() != 3) {
        return Fail("status counters do not match registry operations");
    }

    for (int version = 1; version <= 2; ++version) {
        const auto version_path =
            root / "serving" /
            ("model_v" + std::string(5, '0') +
             std::to_string(version) + ".onnx");
        {
            std::ofstream output(version_path, std::ios::binary);
            output << "model-version-" << version;
        }
        std::string version_checksum;
        if (!ComputeSha256(version_path, &version_checksum, &error)) {
            return Fail(error);
        }
        auto version_manifest =
            Manifest(version_path, version_checksum, version);
        if (registry.Register(version_manifest, &record, &message) !=
            maze::MODEL_REGISTER_RESULT_REGISTERED) {
            return Fail("retained version register failed: " + message);
        }
    }
    if (registry.Get(0, &record, &message)) {
        return Fail("old model version was not evicted");
    }
    if (!registry.Get(1, &record, &message) ||
        !registry.Get(2, &record, &message)) {
        return Fail("serving retention removed a current model");
    }
    registry.GetStatus(&status);
    if (status.registered_model_count() != 2) {
        return Fail("serving retention count is not bounded");
    }
    std::filesystem::remove_all(root);
    return 0;
}
