#pragma once

#include "config/config_loader.h"
#include "maze.pb.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <tuple>

struct RegisteredModel {
    maze::ModelArtifactManifest manifest;
    std::filesystem::path path;
};

class ModelRegistry {
public:
    explicit ModelRegistry(ModelDistributorConfig config);

    maze::ModelRegisterResult Register(
        const maze::ModelArtifactManifest& manifest,
        RegisteredModel* registered,
        std::string* message);

    bool Get(const std::string& run_id,
             int model_version,
             RegisteredModel* model,
             std::string* message);

    bool GetLatest(const std::string& run_id,
                   RegisteredModel* model,
                   std::string* message);

    maze::ModelAckResult Ack(const maze::AckModelReq& request,
                            std::string* message);

    void RecordDownloadAttempt();
    void RecordDownloadSuccess(int64_t bytes);
    void RecordDownloadFailure(const std::string& error);
    void GetStatus(const std::string& run_id,
                   maze::ModelDistributorStatusRsp* response) const;

    const std::string& instance_id() const;
    std::size_t chunk_bytes() const;

private:
    using ModelKey = std::pair<std::string, int>;
    using AckKey = std::tuple<std::string, int, std::string>;

    bool ValidateManifest(const maze::ModelArtifactManifest& manifest,
                          std::filesystem::path* model_path,
                          std::string* error) const;

    ModelDistributorConfig config_;
    std::filesystem::path artifact_root_;
    std::string instance_id_;
    mutable std::mutex mutex_;
    std::map<ModelKey, RegisteredModel> models_;
    std::map<AckKey, maze::AckModelReq> acknowledgements_;
    int64_t register_attempt_count_ = 0;
    int64_t duplicate_register_count_ = 0;
    int64_t rejected_register_count_ = 0;
    int64_t download_request_count_ = 0;
    int64_t download_success_count_ = 0;
    int64_t download_failure_count_ = 0;
    int64_t bytes_served_ = 0;
    int64_t loaded_ack_count_ = 0;
    int64_t failed_ack_count_ = 0;
    std::string latest_ack_aiserver_id_;
    int latest_ack_model_version_ = -1;
    maze::ModelLoadStatus latest_ack_status_ =
        maze::MODEL_LOAD_STATUS_UNSPECIFIED;
    std::string last_error_;
};

bool ComputeSha256(const std::filesystem::path& path,
                   std::string* checksum,
                   std::string* error);
