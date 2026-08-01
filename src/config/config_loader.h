#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ModelDistributorConfig {
    int listen_port = 9200;
    uint32_t protocol_version = 3;
    std::string artifact_root = "models/local-train";
    std::size_t chunk_bytes = 1024 * 1024;
    std::size_t retention_versions = 2;
    std::string contract_version = "0.6.0";
};

bool LoadModelDistributorConfig(const std::string& yaml_path,
                                ModelDistributorConfig& out_config);
