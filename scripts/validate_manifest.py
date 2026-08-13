#!/usr/bin/env python3

import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


SHA256 = re.compile(r"[a-f0-9]{64}")
MAX_MODEL_VERSION = (1 << 64) - 1
CONTRACT_SOURCE_DIGEST = (
    "7e3eb7227e67a2a880130c9c82f87041691c0095f838a60f80abc1f387c1c5b3"
)
CONTRACT_ARTIFACT_DIGEST = (
    "077ac6d61486fafd5f0430eeb05a492764b36e073282f6d7626d0414bb5b2ddf"
)
CONTRACT_GENERATOR_IDENTITY = (
    "0eb73fc2cb675bdb34bf3db9c99dae62a82f93a5e3a72db84dcf3936464729c8"
)
LEGACY_FIELDS = {
    "schema_version",
    "contract_version",
    "model_version",
    "sha256",
    "shape",
}


def require_digest(value: object, field: str) -> None:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def require_schema(value: object, field: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a schema identity")
    if not isinstance(value.get("schema_id"), str) or not value["schema_id"]:
        raise ValueError(f"{field}.schema_id is required")
    if not isinstance(value.get("schema_version"), int) or value["schema_version"] <= 0:
        raise ValueError(f"{field}.schema_version must be positive")
    require_digest(value.get("canonical_digest"), f"{field}.canonical_digest")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_manifest.py MANIFEST", file=sys.stderr)
        return 2
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    forbidden = LEGACY_FIELDS & data.keys()
    if forbidden:
        raise ValueError(f"legacy fields are forbidden: {sorted(forbidden)}")
    required = {
        "manifest_schema_version",
        "contract",
        "identity",
        "observation_schema",
        "action_schema",
        "model_architecture_id",
        "tensor_dtype",
        "input_shape",
        "action_shape",
        "value_shape",
        "artifact_uri",
        "model_file",
        "size_bytes",
        "seed",
        "train_updates",
        "trained_samples",
        "training_config_digest",
        "training_semantics",
        "published_at_unix_ms",
        "ready",
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if data["manifest_schema_version"] != 1:
        raise ValueError("manifest_schema_version must be 1")
    contract = data["contract"]
    if not isinstance(contract, dict) or contract.get("package_name") != "rl-contracts" or contract.get("package_version") != "0.11.0":
        raise ValueError("contract must identify rl-contracts 0.11.0")
    require_digest(contract.get("source_digest"), "contract.source_digest")
    require_digest(contract.get("artifact_digest"), "contract.artifact_digest")
    require_digest(contract.get("generator_identity"), "contract.generator_identity")
    if (
        contract.get("source_digest") != CONTRACT_SOURCE_DIGEST
        or contract.get("artifact_digest") != CONTRACT_ARTIFACT_DIGEST
        or contract.get("generator_identity")
        != CONTRACT_GENERATOR_IDENTITY
        or contract.get("platform") != "linux/arm64"
    ):
        raise ValueError("contract identity does not match the selected 0.11.0 artifact")
    identity = data["identity"]
    if not isinstance(identity, dict) or not identity.get("model_lineage_id"):
        raise ValueError("identity.model_lineage_id is required")
    version = identity.get("model_version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version < 0
        or version > MAX_MODEL_VERSION
    ):
        raise ValueError("identity.model_version must be a uint64 integer")
    require_digest(identity.get("artifact_digest"), "identity.artifact_digest")
    require_digest(identity.get("manifest_digest"), "identity.manifest_digest")
    require_schema(data["observation_schema"], "observation_schema")
    require_schema(data["action_schema"], "action_schema")
    require_digest(data["training_config_digest"], "training_config_digest")
    semantics = data["training_semantics"]
    if not isinstance(semantics, dict):
        raise ValueError("training_semantics must be an object")
    semantics_required = {
        "training_contract_id",
        "observation_schema",
        "action_schema",
        "reward_schema",
        "policy_distribution_schema_id",
        "model_architecture_id",
        "semantics_digest",
    }
    semantics_missing = semantics_required - semantics.keys()
    if semantics_missing:
        raise ValueError(
            f"training_semantics missing fields: {sorted(semantics_missing)}"
        )
    require_schema(semantics["observation_schema"], "training_semantics.observation_schema")
    require_schema(semantics["action_schema"], "training_semantics.action_schema")
    require_schema(semantics["reward_schema"], "training_semantics.reward_schema")
    require_digest(semantics.get("semantics_digest"), "training_semantics.semantics_digest")
    if (
        semantics["observation_schema"] != data["observation_schema"]
        or semantics["action_schema"] != data["action_schema"]
        or semantics.get("model_architecture_id") != data["model_architecture_id"]
        or not semantics.get("training_contract_id")
        or not semantics.get("policy_distribution_schema_id")
    ):
        raise ValueError("training_semantics does not match the model contract")
    for field in ("input_shape", "action_shape", "value_shape"):
        shape = data[field]
        if not isinstance(shape, list) or not shape or any(
            not isinstance(dimension, int) or dimension <= 0
            for dimension in shape
        ):
            raise ValueError(f"{field} must contain positive dimensions")
    if not isinstance(data["artifact_uri"], str):
        raise ValueError("artifact_uri must use the canonical file URI")
    parsed_uri = urlparse(data["artifact_uri"])
    artifact_path = Path(unquote(parsed_uri.path))
    version_directory = f"{version:06d}"
    if (
        parsed_uri.scheme != "file"
        or parsed_uri.netloc not in ("", "localhost")
        or artifact_path.name != "SaveModel.onnx"
        or artifact_path.parent.name != version_directory
        or artifact_path.parent.parent.name != "archive"
    ):
        raise ValueError("artifact_uri must use archive/<version>/SaveModel.onnx")
    if data["model_file"] != "SaveModel.onnx":
        raise ValueError("model_file must be SaveModel.onnx")
    for field in ("size_bytes", "train_updates", "trained_samples", "published_at_unix_ms"):
        if not isinstance(data[field], int) or data[field] < 0:
            raise ValueError(f"{field} must be non-negative")
    if not isinstance(data["ready"], bool):
        raise ValueError("ready must be boolean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
