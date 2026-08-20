#pragma once

#include "store/model_registry.h"

#include <filesystem>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <openssl/evp.h>
#include <stdexcept>
#include <string>

namespace test_helpers {

constexpr char kSourceDigest[] =
    "1111111111111111111111111111111111111111111111111111111111111111";
constexpr char kContractArtifactDigest[] =
    "2222222222222222222222222222222222222222222222222222222222222222";
constexpr char kGeneratorIdentity[] =
    "3333333333333333333333333333333333333333333333333333333333333333";

inline void Digest(rl::common::v1::ContentDigest* digest,
                   const std::string& value) {
    digest->set_algorithm(rl::common::v1::DIGEST_ALGORITHM_SHA256);
    digest->set_hex(value);
}

inline std::string Sha256Bytes(const std::string& data) {
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (context == nullptr) throw std::runtime_error("create SHA-256 context");
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_size = 0;
    const bool ok =
        EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
        EVP_DigestUpdate(context, data.data(), data.size()) == 1 &&
        EVP_DigestFinal_ex(context, digest, &digest_size) == 1;
    EVP_MD_CTX_free(context);
    if (!ok || digest_size != 32) {
        throw std::runtime_error("compute SHA-256");
    }
    static constexpr char kHex[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (unsigned int index = 0; index < digest_size; ++index) {
        result.push_back(kHex[digest[index] >> 4]);
        result.push_back(kHex[digest[index] & 0x0f]);
    }
    return result;
}

template <typename Message>
inline std::string DeterministicBytes(const Message& message) {
    std::string serialized;
    google::protobuf::io::StringOutputStream output(&serialized);
    google::protobuf::io::CodedOutputStream coded(&output);
    coded.SetSerializationDeterministic(true);
    if (!message.SerializeToCodedStream(&coded) || coded.HadError()) {
        throw std::runtime_error("serialize protobuf deterministically");
    }
    coded.Trim();
    return serialized;
}

inline void Contract(rl::common::v1::ContractIdentity* contract) {
    contract->set_package_name("rl-contracts");
    contract->set_package_version("0.14.0");
    Digest(contract->mutable_source_digest(), kSourceDigest);
    Digest(contract->mutable_artifact_digest(), kContractArtifactDigest);
    contract->set_platform("linux/arm64");
    contract->set_generator_identity(kGeneratorIdentity);
}

inline void Schema(rl::common::v1::SchemaIdentity* schema,
                   const std::string& id,
                   char digest_character) {
    schema->set_schema_id(id);
    schema->set_schema_version(1);
    Digest(schema->mutable_canonical_digest(),
           std::string(64, digest_character));
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
    config.contract.package_version = "0.14.0";
    config.contract.source_digest = kSourceDigest;
    config.contract.artifact_digest = kContractArtifactDigest;
    config.contract.platform = "linux/arm64";
    config.contract.generator_identity = kGeneratorIdentity;
    return config;
}

inline rl::training::v1::ModelArtifactManifest Manifest(
    const std::filesystem::path& path,
    const std::string& artifact_digest) {
    rl::training::v1::ModelArtifactManifest manifest;
    manifest.set_manifest_schema_version(3);
    Contract(manifest.mutable_contract());
    manifest.mutable_identity()->set_model_lineage_id("lineage-fixed");
    manifest.mutable_identity()->set_model_step(0);
    Digest(manifest.mutable_identity()->mutable_artifact_digest(),
           artifact_digest);
    Schema(manifest.mutable_observation_schema(), "maze.observation.v3", '4');
    Schema(manifest.mutable_action_schema(), "maze.action.v1", '5');
    manifest.set_model_architecture_id("maze.mlp-17x64x64.v1");
    manifest.set_tensor_dtype("float32");
    manifest.add_input_shape(1);
    manifest.add_input_shape(17);
    manifest.add_action_shape(1);
    manifest.add_action_shape(9);
    manifest.add_value_shape(1);
    manifest.add_value_shape(1);
    manifest.set_artifact_uri("file://" + path.string());
    manifest.set_model_file("SaveModel.onnx");
    manifest.set_size_bytes(
        static_cast<int64_t>(std::filesystem::file_size(path)));
    manifest.set_seed(0);
    manifest.set_train_updates(0);
    manifest.set_trained_samples(0);
    Digest(manifest.mutable_training_config_digest(), std::string(64, '6'));

    auto* semantics = manifest.mutable_training_semantics();
    semantics->set_training_contract_id("maze.training.v3");
    *semantics->mutable_observation_schema() = manifest.observation_schema();
    *semantics->mutable_action_schema() = manifest.action_schema();
    Schema(semantics->mutable_reward_schema(), "maze.reward.v4", '7');
    semantics->set_policy_distribution_schema_id("categorical.logits.v1");
    semantics->set_model_architecture_id(manifest.model_architecture_id());
    Digest(semantics->mutable_semantics_digest(), std::string(64, '8'));

    auto* profile = manifest.mutable_rollout_estimator_profile();
    profile->set_profile_schema_version(1);
    profile->set_gamma(0.99);
    profile->set_gae_lambda(0.95);
    profile->set_tmax(128);
    profile->set_gae_formula_id("gae.backward.v1");
    profile->set_terminal_bootstrap_semantics_id(
        "maze.timeout-keep-and-cut-bootstrap.v1");
    profile->set_value_target_formula_id(
        "advantage-plus-behavior-value.v1");
    profile->set_value_head_abi_id("scalar-value.float32.v1");
    *profile->mutable_reward_semantics_digest() =
        semantics->reward_schema().canonical_digest();
    profile->set_numeric_dtype("float32");
    profile->set_finite_rule_id("reject-nonfinite.v1");
    profile->set_model_pin_semantics_id("per-agent-segment-pin.v1");
    profile->clear_profile_digest();
    const std::string profile_digest =
        Sha256Bytes(DeterministicBytes(*profile));
    Digest(profile->mutable_profile_digest(), profile_digest);

    manifest.set_published_at_unix_ms(1700000000000);
    manifest.set_ready(true);
    std::string manifest_digest;
    std::string error;
    if (!ComputeManifestDigest(manifest, &manifest_digest, &error)) {
        throw std::runtime_error(error);
    }
    Digest(manifest.mutable_identity()->mutable_manifest_digest(),
           manifest_digest);
    return manifest;
}

}  // namespace test_helpers
