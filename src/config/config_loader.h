#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ModelDistributorConfig {
    int listen_port = 9200;
    std::string run_id = "local-run";
    uint32_t protocol_version = 2;
    std::string artifact_root = "models/published";
    std::size_t chunk_bytes = 1024 * 1024;
    std::string contract_version = "0.3.0";
};

bool LoadModelDistributorConfig(const std::string& yaml_path,
                                ModelDistributorConfig& out_config);
