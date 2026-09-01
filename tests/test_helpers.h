#pragma once

#include "store/model_registry.h"

#include <filesystem>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <openssl/evp.h>
#include <stdexcept>
#include <string>

namespace test_helpers {

constexpr char kTrainingContractDigest[] =
    "8888888888888888888888888888888888888888888888888888888888888888";

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
    contract->set_package_version("0.15.0");
    contract->set_platform("linux/arm64");
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
    config.contract.package_version = "0.15.0";
    config.contract.platform = "linux/arm64";
    return config;
}

inline rl::training::v1::ModelArtifactManifest Manifest(
    const std::filesystem::path& path,
    const std::string& artifact_digest) {
    rl::training::v1::ModelArtifactManifest manifest;
    manifest.mutable_identity()->set_model_lineage_id("lineage-fixed");
    manifest.mutable_identity()->set_model_step(0);
    Digest(manifest.mutable_identity()->mutable_artifact_digest(),
           artifact_digest);
    manifest.set_size_bytes(
        static_cast<int64_t>(std::filesystem::file_size(path)));
    manifest.set_trained_samples(0);
    Digest(manifest.mutable_training_config_digest(), std::string(64, '6'));
    Digest(manifest.mutable_training_contract_digest(),
           kTrainingContractDigest);

    auto* profile = manifest.mutable_rollout_estimator_profile();
    profile->set_gamma(0.99);
    profile->set_gae_lambda(0.95);
    profile->set_tmax(128);
    profile->clear_profile_digest();
    const std::string profile_digest =
        Sha256Bytes(DeterministicBytes(*profile));
    Digest(profile->mutable_profile_digest(), profile_digest);

    manifest.set_published_at_unix_ms(1700000000000);
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
