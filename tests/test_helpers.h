#pragma once

#include "store/model_registry.h"

#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>

namespace test_helpers {

constexpr char kSourceDigest[] =
    "dec56e32d6cd66c4edb2257388f06e0d7e158ea51459d2ef0295d4cdb966b4f3";
constexpr char kContractArtifactDigest[] =
    "528624c85754187cfa5736c253042dc096f47568393340fd8df9ac984a21ec6f";
constexpr char kGeneratorIdentity[] =
    "0eb73fc2cb675bdb34bf3db9c99dae62a82f93a5e3a72db84dcf3936464729c8";

inline void Digest(rl::common::v1::ContentDigest* digest,
                   const std::string& value) {
    digest->set_algorithm(rl::common::v1::DIGEST_ALGORITHM_SHA256);
    digest->set_hex(value);
}

inline void Contract(rl::common::v1::ContractIdentity* contract) {
    contract->set_package_name("rl-contracts");
    contract->set_package_version("0.13.0");
    Digest(contract->mutable_source_digest(), kSourceDigest);
    Digest(contract->mutable_artifact_digest(), kContractArtifactDigest);
    contract->set_platform("linux/arm64");
    contract->set_generator_identity(kGeneratorIdentity);
}

inline void Schema(rl::common::v1::SchemaIdentity* schema,
                   const std::string& id,
                   char digest) {
    schema->set_schema_id(id);
    schema->set_schema_version(1);
    Digest(schema->mutable_canonical_digest(), std::string(64, digest));
}

inline void Service(rl::common::v1::ServiceInstanceIdentity* service,
                    const std::string& component,
                    const std::string& instance) {
    service->set_component(component);
    service->set_instance_id(instance);
    service->set_lifecycle_epoch(1);
}

inline ModelDistributorConfig Config(const std::filesystem::path& root) {
    ModelDistributorConfig config;
    config.artifact_root = root.string();
    config.chunk_bytes = 5;
    config.retention_steps = 101;
    config.contract.package_name = "rl-contracts";
    config.contract.package_version = "0.13.0";
    config.contract.source_digest = kSourceDigest;
    config.contract.artifact_digest = kContractArtifactDigest;
    config.contract.platform = "linux/arm64";
    config.contract.generator_identity = kGeneratorIdentity;
    return config;
}

inline rl::training::v1::ModelArtifactManifest Manifest(
    const std::filesystem::path& path,
    const std::string& artifact_digest,
    uint64_t model_step = 0,
    const std::string& lineage = "maze-fixed-map-seed-0") {
    rl::training::v1::ModelArtifactManifest manifest;
    manifest.set_manifest_schema_version(2);
    Contract(manifest.mutable_contract());
    manifest.mutable_identity()->set_model_lineage_id(lineage);
    manifest.mutable_identity()->set_model_step(model_step);
    Digest(manifest.mutable_identity()->mutable_artifact_digest(),
           artifact_digest);
    Schema(manifest.mutable_observation_schema(), "maze.observation.v3", '1');
    Schema(manifest.mutable_action_schema(), "maze.action.v1", '2');
    manifest.set_model_architecture_id("maze.mlp-17x64x64.v1");
    manifest.set_tensor_dtype("float32");
    manifest.add_input_shape(1);
    manifest.add_input_shape(17);
    manifest.add_action_shape(1);
    manifest.add_action_shape(9);
    manifest.add_value_shape(1);
    manifest.add_value_shape(1);
    manifest.set_artifact_uri("file://" + path.string());
    manifest.set_model_file(path.filename().string());
    manifest.set_size_bytes(
        static_cast<int64_t>(std::filesystem::file_size(path)));
    manifest.set_seed(0);
    manifest.set_train_updates(model_step);
    manifest.set_trained_samples(model_step);
    Digest(manifest.mutable_training_config_digest(), std::string(64, '3'));
    auto* semantics = manifest.mutable_training_semantics();
    semantics->set_training_contract_id("maze.training.v3");
    *semantics->mutable_observation_schema() = manifest.observation_schema();
    *semantics->mutable_action_schema() = manifest.action_schema();
    Schema(semantics->mutable_reward_schema(), "maze.reward.v4", '4');
    semantics->set_policy_distribution_schema_id("categorical.logits.v1");
    semantics->set_model_architecture_id(manifest.model_architecture_id());
    Digest(semantics->mutable_semantics_digest(), std::string(64, '5'));
    manifest.set_published_at_unix_ms(
        model_step >=
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
            ? std::numeric_limits<int64_t>::max()
            : static_cast<int64_t>(model_step + 1));
    manifest.set_ready(true);
    std::string digest;
    std::string error;
    if (!ComputeManifestDigest(manifest, &digest, &error)) {
        throw std::runtime_error(error);
    }
    Digest(manifest.mutable_identity()->mutable_manifest_digest(), digest);
    return manifest;
}

inline void RefreshManifestDigest(
    rl::training::v1::ModelArtifactManifest* manifest) {
    std::string digest;
    std::string error;
    if (!ComputeManifestDigest(*manifest, &digest, &error)) {
        throw std::runtime_error(error);
    }
    Digest(manifest->mutable_identity()->mutable_manifest_digest(), digest);
}

}  // namespace test_helpers
