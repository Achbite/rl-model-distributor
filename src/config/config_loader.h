#pragma once

#include <cstddef>
#include <string>

struct ContractConfig {
    std::string package_name;
    std::string package_version;
    std::string source_digest;
    std::string artifact_digest;
    std::string platform;
    std::string generator_identity;
};

struct ModelDistributorConfig {
    int listen_port = 9200;
    std::string artifact_root;
    std::size_t chunk_bytes = 1024 * 1024;
    std::size_t retention_versions = 2;
    ContractConfig contract;
};

bool LoadModelDistributorConfig(const std::string& yaml_path,
                                ModelDistributorConfig& out_config);
