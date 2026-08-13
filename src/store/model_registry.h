#pragma once

#include "config/config_loader.h"
#include "training.pb.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>

struct ModelArtifactHandle {
    explicit ModelArtifactHandle(int descriptor_value,
                                 int64_t size_value,
                                 std::filesystem::path path_value);
    ~ModelArtifactHandle();

    ModelArtifactHandle(const ModelArtifactHandle&) = delete;
    ModelArtifactHandle& operator=(const ModelArtifactHandle&) = delete;

    int descriptor = -1;
    int64_t size = 0;
    std::filesystem::path path;
};

struct RegisteredModel {
    rl::training::v1::ModelArtifactManifest manifest;
    std::filesystem::path path;
    std::shared_ptr<const ModelArtifactHandle> artifact;
};

class ModelRegistry {
public:
    explicit ModelRegistry(ModelDistributorConfig config);

    rl::training::v1::ModelRegisterResult Register(
        const rl::training::v1::ModelArtifactManifest& manifest,
        RegisteredModel* registered,
        std::string* message);
    bool Get(const rl::training::v1::ModelIdentity& identity,
             RegisteredModel* model,
             std::string* message);
    bool GetByVersion(const std::string& lineage_id,
                      uint64_t model_version,
                      RegisteredModel* model,
                      std::string* message);
    bool GetLatest(const std::string& lineage_id,
                   RegisteredModel* model,
                   std::string* message);
    bool GetManifestSnapshot(
        const rl::training::v1::ModelIdentity& selector,
        bool latest_in_lineage,
        RegisteredModel* model,
        uint64_t* floor,
        uint64_t* latest,
        std::string* message);
    bool GetAvailableRange(const std::string& lineage_id,
                           uint64_t* floor,
                           uint64_t* latest,
                           std::string* message) const;
    rl::training::v1::ModelAckResult Ack(
        const rl::training::v1::AckModelReq& request,
        std::string* message);

    void RecordDownloadAttempt();
    void RecordDownloadSuccess(
        const rl::common::v1::ServiceInstanceIdentity& requester,
        const rl::training::v1::ModelIdentity& model,
        int64_t bytes);
    void RecordDownloadFailure(const std::string& error);
    void GetStatus(rl::training::v1::ModelDistributorStatusRsp* response) const;
    void FillDistributorIdentity(
        rl::common::v1::ServiceInstanceIdentity* identity) const;

    const std::string& instance_id() const;
    std::size_t chunk_bytes() const;

    static bool ServiceIdentityValid(
        const rl::common::v1::ServiceInstanceIdentity& identity);
    static bool ModelIdentityValid(
        const rl::training::v1::ModelIdentity& identity);

private:
    struct ModelKey {
        std::string lineage_id;
        uint64_t version = 0;

        bool operator<(const ModelKey& other) const {
            return lineage_id < other.lineage_id ||
                   (lineage_id == other.lineage_id && version < other.version);
        }
    };

    bool ContractMatches(
        const rl::common::v1::ContractIdentity& contract) const;
    void FillContractIdentity(
        rl::common::v1::ContractIdentity* identity) const;
    bool ValidateManifest(
        const rl::training::v1::ModelArtifactManifest& manifest,
        std::filesystem::path* model_path,
        std::string* error) const;
    static std::string AckKey(const rl::training::v1::AckModelReq& request);
    static std::string ServiceKey(
        const rl::common::v1::ServiceInstanceIdentity& service);
    static std::string DownloadKey(
        const rl::common::v1::ServiceInstanceIdentity& service,
        const rl::training::v1::ModelIdentity& model);
    static bool AckEquivalent(const rl::training::v1::AckModelReq& left,
                              const rl::training::v1::AckModelReq& right);
    bool AvailableRangeLocked(const std::string& lineage_id,
                              uint64_t* floor,
                              uint64_t* latest,
                              std::string* message) const;
    void PruneSettlementFactsLocked(const std::string& lineage_id);

    ModelDistributorConfig config_;
    std::filesystem::path artifact_root_;
    std::string instance_id_;
    mutable std::mutex mutex_;
    std::map<ModelKey, RegisteredModel> models_;
    std::map<std::string, uint64_t> latest_version_seen_;
    std::map<std::string, rl::training::v1::ModelIdentity>
        completed_downloads_;
    std::map<std::string, rl::training::v1::AckModelReq> acknowledgements_;
    std::map<std::string, ModelKey> acknowledgement_models_;
    bool has_latest_model_ = false;
    rl::training::v1::ModelIdentity latest_model_;
    int64_t register_attempt_count_ = 0;
    int64_t duplicate_register_count_ = 0;
    int64_t rejected_register_count_ = 0;
    int64_t download_request_count_ = 0;
    int64_t download_success_count_ = 0;
    int64_t download_failure_count_ = 0;
    int64_t bytes_served_ = 0;
    int64_t loaded_ack_count_ = 0;
    int64_t failed_ack_count_ = 0;
    bool has_latest_ack_ = false;
    rl::common::v1::ServiceInstanceIdentity latest_ack_aiserver_;
    rl::training::v1::ModelIdentity latest_ack_model_;
    rl::training::v1::ModelLoadStatus latest_ack_status_ =
        rl::training::v1::MODEL_LOAD_STATUS_UNSPECIFIED;
    std::string last_error_;
};

bool ComputeSha256(const std::filesystem::path& path,
                   std::string* checksum,
                   std::string* error);

bool ComputeManifestDigest(
    const rl::training::v1::ModelArtifactManifest& manifest,
    std::string* digest,
    std::string* error);
