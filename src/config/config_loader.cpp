#include "config/config_loader.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Entry {
    std::string section;
    std::string key;
    std::string value;
};

std::string Trim(const std::string& value) {
    const size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string StripQuotes(const std::string& value) {
    if (value.size() >= 2 &&
        ((value.front() == '"' && value.back() == '"') ||
         (value.front() == '\'' && value.back() == '\''))) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

bool Parse(const std::string& text,
           std::vector<Entry>* entries,
           std::string* error) {
    std::istringstream stream(text);
    std::string line;
    std::string section;
    int line_number = 0;
    while (std::getline(stream, line)) {
        ++line_number;
        bool quoted = false;
        for (size_t index = 0; index < line.size(); ++index) {
            if (line[index] == '"' || line[index] == '\'') {
                quoted = !quoted;
            } else if (line[index] == '#' && !quoted) {
                line.erase(index);
                break;
            }
        }
        const std::string trimmed = Trim(line);
        if (trimmed.empty()) continue;
        const size_t colon = trimmed.find(':');
        if (colon == std::string::npos) {
            *error = "line " + std::to_string(line_number) + " lacks ':'";
            return false;
        }
        const std::string key = Trim(trimmed.substr(0, colon));
        const std::string value = Trim(trimmed.substr(colon + 1));
        const bool indented = !line.empty() &&
                              (line.front() == ' ' || line.front() == '\t');
        if (!indented && value.empty()) {
            section = key;
        } else if (indented && !section.empty() && !key.empty() &&
                   !value.empty()) {
            entries->push_back({section, key, StripQuotes(value)});
        } else {
            *error = "line " + std::to_string(line_number) +
                     " has unsupported YAML structure";
            return false;
        }
    }
    return true;
}

const std::string* Required(const std::vector<Entry>& entries,
                            const std::string& section,
                            const std::string& key) {
    const std::string* found = nullptr;
    for (const auto& entry : entries) {
        if (entry.section == section && entry.key == key) {
            if (found != nullptr) return nullptr;
            found = &entry.value;
        }
    }
    return found;
}

template <typename T>
bool Integer(const std::string& text, T* output) {
    try {
        size_t used = 0;
        const unsigned long long value = std::stoull(text, &used);
        if (used != text.size()) return false;
        *output = static_cast<T>(value);
        return static_cast<unsigned long long>(*output) == value;
    } catch (...) {
        return false;
    }
}

}  // namespace

bool LoadModelDistributorConfig(const std::string& yaml_path,
                                ModelDistributorConfig& output) {
    std::ifstream input(yaml_path);
    if (!input.is_open()) {
        std::cerr << "[ModelDistributor] required config not found: "
                  << yaml_path << std::endl;
        return false;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    std::vector<Entry> entries;
    std::string error;
    if (!Parse(buffer.str(), &entries, &error)) {
        std::cerr << "[ModelDistributor] invalid config: " << error
                  << std::endl;
        return false;
    }
    static const std::set<std::string> allowed = {
        "server.listen_port",          "model.artifact_root",
        "model.chunk_bytes",           "model.retention_steps",
    };
    for (const auto& entry : entries) {
        if (allowed.count(entry.section + "." + entry.key) == 0) {
            std::cerr << "[ModelDistributor] unknown config key: "
                      << entry.section << "." << entry.key << std::endl;
            return false;
        }
    }
    const auto get = [&](const char* section,
                         const char* key) -> const std::string* {
        const std::string* value = Required(entries, section, key);
        if (value == nullptr || value->empty()) {
            std::cerr << "[ModelDistributor] missing or duplicate config key: "
                      << section << "." << key << std::endl;
            return nullptr;
        }
        return value;
    };
    const std::string* port = get("server", "listen_port");
    const std::string* root = get("model", "artifact_root");
    const std::string* chunk = get("model", "chunk_bytes");
    const std::string* retention = get("model", "retention_steps");
    if (!port || !root || !chunk || !retention) {
        return false;
    }
    ModelDistributorConfig parsed;
    if (!Integer(*port, &parsed.listen_port) ||
        !Integer(*chunk, &parsed.chunk_bytes) ||
        !Integer(*retention, &parsed.retention_steps)) {
        std::cerr << "[ModelDistributor] malformed numeric config value"
                  << std::endl;
        return false;
    }
    parsed.artifact_root = *root;
    if (parsed.listen_port <= 0 || parsed.listen_port > 65535 ||
        parsed.artifact_root.empty() || parsed.chunk_bytes == 0 ||
        parsed.chunk_bytes > 4 * 1024 * 1024 ||
        parsed.retention_steps == 0) {
        std::cerr << "[ModelDistributor] config value is out of range"
                  << std::endl;
        return false;
    }
    if (const char* port_override = std::getenv("RL_MODEL_DISTRIBUTOR_PORT")) {
        int value = 0;
        if (!Integer(std::string(port_override), &value) || value <= 0 ||
            value > 65535) {
            std::cerr << "[ModelDistributor] RL_MODEL_DISTRIBUTOR_PORT invalid"
                      << std::endl;
            return false;
        }
        parsed.listen_port = value;
    }
    if (const char* root_override = std::getenv("RL_MODEL_ARTIFACT_ROOT")) {
        if (*root_override == '\0') return false;
        parsed.artifact_root = root_override;
    }
    if (parsed.artifact_root ==
        "__DIRECT_ARTIFACT_ROOT_REQUIRED__") {
        std::cerr << "[ModelDistributor] RL_MODEL_ARTIFACT_ROOT must bind "
                     "the current isolated models/train directory"
                  << std::endl;
        return false;
    }
    output = parsed;
    return true;
}
