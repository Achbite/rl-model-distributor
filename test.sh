#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
contract_cpp_dir="${CONTRACT_CPP_DIR:-/contracts/cpp}"

if [ "$#" -ne 0 ]; then
    echo "usage: bash ./test.sh" >&2
    exit 2
fi

if [ ! -f "${contract_cpp_dir}/training.pb.cc" ]; then
    echo "CONTRACT_CPP_DIR must point to generated rl-contracts C++ bindings" >&2
    echo "current value: ${contract_cpp_dir}" >&2
    exit 1
fi

test_build_dir="$(mktemp -d "${TMPDIR:-/tmp}/rl-model-distributor-test.XXXXXX")"
trap 'rm -rf "${test_build_dir}"' EXIT

# TCR-A3-TEST-BOUNDARY-001: these two native contracts are the complete
# approved suite. New targets or cases require a separately approved TCR.
cmake -S "${repo_dir}" -B "${test_build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DCONTRACT_CPP_DIR="${contract_cpp_dir}"
cmake --build "${test_build_dir}" --parallel \
    --target model_registry_test model_distributor_rpc_test
ctest --test-dir "${test_build_dir}" \
    --output-on-failure \
    -R '^(model_registry_contract|model_distributor_rpc_contract)$'
