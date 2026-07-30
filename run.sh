#!/usr/bin/env bash

set -euo pipefail
repo_dir="$(cd "$(dirname "$0")" && pwd)"
binary="${MODEL_DISTRIBUTOR_BIN:-${repo_dir}/build/maze_model_distributor}"
config="${MODEL_DISTRIBUTOR_CONFIG:-${repo_dir}/configs/model_distributor_config.yaml}"
if [ ! -x "${binary}" ]; then
    echo "ModelDistributor executable is missing: ${binary}" >&2
    exit 1
fi
exec "${binary}" "${config}"
