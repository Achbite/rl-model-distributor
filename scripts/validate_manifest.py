#!/usr/bin/env python3

import json
import re
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_manifest.py MANIFEST", file=sys.stderr)
        return 2

    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "contract_version",
        "model_version",
        "run_id",
        "artifact_uri",
        "sha256",
        "ready",
    }
    if required - data.keys():
        raise ValueError(f"missing fields: {sorted(required - data.keys())}")
    if data["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if not re.fullmatch(r"0\.[1-9]\d*\.\d+", data["contract_version"]):
        raise ValueError("contract_version must be a supported 0.x version")
    if not isinstance(data["model_version"], int) or data["model_version"] < 0:
        raise ValueError("model_version must be a non-negative integer")
    if not isinstance(data["run_id"], str) or not data["run_id"]:
        raise ValueError("run_id must be non-empty")
    if not isinstance(data["artifact_uri"], str) or not data["artifact_uri"]:
        raise ValueError("artifact_uri must be non-empty")
    if not re.fullmatch(r"[a-f0-9]{64}", data["sha256"]):
        raise ValueError("sha256 must be a lowercase hexadecimal digest")
    if not isinstance(data["ready"], bool):
        raise ValueError("ready must be boolean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
