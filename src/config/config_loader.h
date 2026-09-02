#pragma once

#include <cstddef>
#include <string>

struct ModelDistributorConfig {
    int listen_port = 9200;
    std::string artifact_root;
    std::size_t chunk_bytes = 1024 * 1024;
    std::size_t retention_steps = 101;
};

bool LoadModelDistributorConfig(const std::string& yaml_path,
                                ModelDistributorConfig& out_config);
