#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd -P)}"
version="$(tr -d '[:space:]' < "${repo_dir}/VERSION")"
platform="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')"
platform_dir="${platform//\//-}"
contract_dir="${workspace_root}/.workspace/artifacts/rl-contracts/training"
output_dir="${workspace_root}/.workspace/artifacts/rl-model-distributor/${platform_dir}"
builder_image="rl-training/model-distributor-builder:${version}"

test -f "${contract_dir}/cpp/proto/common/identity.pb.cc"
test -f "${contract_dir}/cpp/proto/training/training.pb.cc"

mkdir -p "${workspace_root}/.workspace/artifacts" "$(dirname "${output_dir}")"
temp_dir="$(mktemp -d "${workspace_root}/.workspace/artifacts/.tmp-rl-model-distributor.XXXXXX")"
trap 'rm -rf "${temp_dir}"' EXIT
mkdir -p "${temp_dir}/bin" "${temp_dir}/config"

docker build \
    --file "${repo_dir}/Dockerfile.build" \
    --tag "${builder_image}" \
    "${repo_dir}"
docker run --rm \
    --volume "${repo_dir}:/source:ro" \
    --volume "${contract_dir}:/contracts:ro" \
    --volume "${temp_dir}:/output" \
    "${builder_image}" \
    bash -lc '
        set -euo pipefail
        cmake -S /source -B /tmp/model-distributor-build -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_TESTING=OFF \
            -DCONTRACT_CPP_DIR=/contracts/cpp
        cmake --build /tmp/model-distributor-build --parallel --target maze_model_distributor
        cp /tmp/model-distributor-build/maze_model_distributor /output/bin/
    '

cp "${repo_dir}/configs/model_distributor_config.yaml" "${temp_dir}/config/model_distributor_config.yaml"
test -x "${temp_dir}/bin/maze_model_distributor"
test -f "${temp_dir}/config/model_distributor_config.yaml"

rm -rf "${output_dir}"
mv "${temp_dir}" "${output_dir}"
trap - EXIT
printf '%s\n' "${output_dir}"
