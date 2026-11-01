# RL Model Distributor

[简体中文](README.md) | English

## 1. Run tests

Run inside the build container with the Contracts C++ bindings mounted:

```bash
bash ./test.sh
```

`test.sh` is the repository's unified test entrypoint and runs the current
development checks from an explicit allowlist.

## 2. Build the current artifact

Supply an existing Contracts Training bundle containing `cpp/` and `sdk/`:

```bash
CONTRACT_CPP_DIR=/path/to/training/cpp bash ./build.sh
```

This repository builds only its own source. It never builds or synchronizes upstream dependencies.
Tests run explicitly through `bash ./test.sh`. The default output is
`build/artifacts/production/<platform>/`; set `RL_ARTIFACT_ROOT` to select another output root.
`build_dev_artifact.sh` uses the same implementation with the default output
`build/artifacts/development/<platform>/current/`.

## 3. Stage it into Learner

With Framework's sibling repository layout, run from `rl-framework`:

```bash
bash ./sync_artifacts.sh
# Development artifact channel
bash ./sync_artifacts.sh --development
```

Framework calls the Contracts, Pool and Distributor build entrypoints and explicitly stages
selected artifacts into Learner. Component builds do not require this workspace layout.

## 4. Model manifest

Learner's current local publication layout is:

```text
models/train/<step:min-width-7>/SaveModel.onnx
```

`SaveModel.onnx` and the seven-digit directory are Learner-internal layout, not
Distributor registration requirements. `RegisterModel` accepts an absolute,
regular, non-symlink model file path explicitly supplied by the caller within the
configured artifact root, while `manifest.pb` remains the sole authoritative model
manifest. Distributor validates lineage/step, publication time, declared byte size,
the regular non-symlink file path under the artifact root, and the size of the opened
file. It computes and checks no content digest. Model bytes are streamed from the
opened descriptor by model identity, offset, and declared size; the consuming AIServer
owns task semantics and ONNX name/shape/type validation. ACK binds the AIServer service
lifecycle, `load_instance_id`, and exact registered model identity. Distributor does
not decide per-Agent activation timing.

`GetModelLoadReceipt` reads the last applied load receipt within retention for one AIServer lifecycle and model identity. The receipt preserves the load instance, result and original message. Duplicate ACKs keep their original order. An absent receipt and an expired retention window are distinguished by `outside_retention`.

## 5. Run it directly

```bash
# When build/maze_model_distributor already exists
bash ./run.sh
```

During full training the Learner container supervises this process; no separate launch is required.

## License

[MIT License](LICENSE)

## Build input and cache ownership

CMake/Ninja trees persist under `build/<platform>/<builder-image-id>/<channel>/` and ccache
uses a persistent Docker volume. Successful builds select `build/maze_model_distributor` for `run.sh`
and export the same binary. Run it in the same Linux toolchain environment; macOS hosts
only orchestrate Docker compilation. Toolchain identity partitions caches, not runtime admission.

Linux container builds refresh input attributes with `stat --cached=never` before CMake/Ninja, so a host edit is not hidden by virtiofs metadata caching. This neither changes source timestamps nor synchronizes upstream artifacts.
