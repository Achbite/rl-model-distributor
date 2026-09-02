#pragma once

#include "store/model_registry.h"

#include <filesystem>
#include <string>

namespace test_helpers {

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
    return config;
}

inline rl::training::v1::ModelArtifactManifest Manifest(
    const std::filesystem::path& path) {
    rl::training::v1::ModelArtifactManifest manifest;
    manifest.mutable_identity()->set_model_lineage_id("lineage-fixed");
    manifest.mutable_identity()->set_model_step(0);
    manifest.set_size_bytes(
        static_cast<int64_t>(std::filesystem::file_size(path)));
    manifest.set_trained_samples(0);
    manifest.set_published_at_unix_ms(1700000000000);
    return manifest;
}

}  // namespace test_helpers
