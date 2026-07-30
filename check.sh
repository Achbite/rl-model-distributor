#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${repo_dir}/scripts/validate_manifest.py" \
    "${repo_dir}/tests/manifest.example.json"
