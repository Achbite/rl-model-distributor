# RL Model Distributor

[简体中文](README.md) | English

## 1. Run tests

Run inside the build container with the Contracts C++ bindings mounted:

```bash
bash ./test.sh
```

`test.sh` is the repository's unified test entrypoint and runs the current
development checks from an explicit allowlist.

## 2. Build the 0.15.0 artifact

Create the task-neutral training-contract artifact first, then build from the
current source:

```bash
(cd ../rl-contracts && bash build_artifact.sh training)
bash build_artifact.sh
```

The artifact script only compiles the production binary. Tests remain an explicit
`bash ./test.sh` operation from section 1. Output:

```text
../.workspace/artifacts/rl-model-distributor/0.15.0/<platform>/
```

## 3. Stage it into Learner

Prefer the Learner sync entrypoint to copy the complete artifact:

```bash
(cd ../rl-learner && bash scripts/sync_runtime_artifacts.sh)
```

For development artifacts, Learner's `make deps` builds and stages both required
binaries. `make shell` never performs that synchronization implicitly.

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

## 5. Run it directly

```bash
# When build/maze_model_distributor already exists
bash ./run.sh
```

During full training the Learner container supervises this process; no separate launch is required.

## License

[MIT License](LICENSE)
