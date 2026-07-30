#include "config/config_loader.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct YamlEntry {
    std::string section;
    std::string key;
    std::string value;
};

std::string Trim(const std::string& value) {
    const auto start = value.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}

std::string StripQuotes(const std::string& value) {
    if (value.size() >= 2 &&
        ((value.front() == '"' && value.back() == '"') ||
         (value.front() == '\'' && value.back() == '\''))) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

std::vector<YamlEntry> ParseYaml(const std::string& text) {
    std::vector<YamlEntry> entries;
    std::istringstream stream(text);
    std::string line;
    std::string section;
    while (std::getline(stream, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) line = line.substr(0, comment);
        const std::string trimmed = Trim(line);
        if (trimmed.empty()) continue;
        const auto colon = trimmed.find(':');
        if (colon == std::string::npos) continue;
        const std::string key = Trim(trimmed.substr(0, colon));
        const std::string value = Trim(trimmed.substr(colon + 1));
        const bool indented =
            !line.empty() && (line.front() == ' ' || line.front() == '\t');
        if (!indented && value.empty()) {
            section = key;
        } else if (indented && !key.empty()) {
            entries.push_back({section, key, StripQuotes(value)});
        }
    }
    return entries;
}

std::string FindValue(const std::vector<YamlEntry>& entries,
                      const std::string& section,
                      const std::string& key) {
    for (const auto& entry : entries) {
        if (entry.section == section && entry.key == key) return entry.value;
    }
    return "";
}

std::string Env(const char* name) {
    const char* value = std::getenv(name);
    return value ? value : "";
}

int SafeInt(const std::string& value, int fallback) {
    if (value.empty()) return fallback;
    try {
        return std::stoi(value);
    } catch (...) {
        return fallback;
    }
}

std::size_t SafeSize(const std::string& value, std::size_t fallback) {
    if (value.empty()) return fallback;
    try {
        return static_cast<std::size_t>(std::stoull(value));
    } catch (...) {
        return fallback;
    }
}

}  // namespace

bool LoadModelDistributorConfig(const std::string& yaml_path,
                                ModelDistributorConfig& out_config) {
    std::ifstream input(yaml_path);
    if (!input.is_open()) {
        std::cerr << "[ModelDistributor] config not found, using defaults: "
                  << yaml_path << std::endl;
        out_config = ModelDistributorConfig{};
        return false;
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    const auto entries = ParseYaml(buffer.str());
    out_config.listen_port =
        SafeInt(FindValue(entries, "server", "listen_port"), 9200);
    const std::string run_id = FindValue(entries, "server", "run_id");
    if (!run_id.empty()) out_config.run_id = run_id;
    out_config.protocol_version = static_cast<uint32_t>(
        SafeInt(FindValue(entries, "server", "protocol_version"), 1));
    const std::string artifact_root =
        FindValue(entries, "model", "artifact_root");
    if (!artifact_root.empty()) out_config.artifact_root = artifact_root;
    out_config.chunk_bytes = SafeSize(
        FindValue(entries, "model", "chunk_bytes"), 1024 * 1024);
    const std::string contract_version =
        FindValue(entries, "model", "contract_version");
    if (!contract_version.empty()) {
        out_config.contract_version = contract_version;
    }

    const std::string env_port = Env("MAZE_MODEL_DISTRIBUTOR_PORT");
    if (!env_port.empty()) {
        out_config.listen_port = SafeInt(env_port, out_config.listen_port);
    }
    const std::string env_run_id = Env("MAZE_RUN_ID");
    if (!env_run_id.empty()) out_config.run_id = env_run_id;
    const std::string env_root = Env("MAZE_MODEL_ARTIFACT_ROOT");
    if (!env_root.empty()) out_config.artifact_root = env_root;
    const std::string env_chunk = Env("MAZE_MODEL_CHUNK_BYTES");
    if (!env_chunk.empty()) {
        out_config.chunk_bytes = SafeSize(env_chunk, out_config.chunk_bytes);
    }

    if (out_config.listen_port <= 0 || out_config.listen_port > 65535) {
        out_config.listen_port = 9200;
    }
    if (out_config.chunk_bytes == 0 ||
        out_config.chunk_bytes > 4 * 1024 * 1024) {
        out_config.chunk_bytes = 1024 * 1024;
    }
    return true;
}
