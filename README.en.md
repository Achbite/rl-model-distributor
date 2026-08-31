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

Create the Contracts artifact first, then run the build from a clean Git savepoint:

```bash
(cd ../rl-contracts && bash build_artifact.sh)
bash build_artifact.sh
```

The script compiles the service and runs its tests. Output:

```text
../.workspace/artifacts/rl-model-distributor/0.15.0/<platform>/
```

## 3. Stage it into Learner

Do not copy the binary manually. Use the Learner sync entry point:

```bash
(cd ../rl-learner && bash scripts/sync_runtime_artifacts.sh)
```

## 4. Model manifest

Model paths must use:

```text
models/train/<step:min-width-7>/SaveModel.onnx
```

`manifest.pb` is the sole authoritative model manifest. Distributor validates
its deterministic protobuf, training-contract digest, profile, lineage, step,
and digests at `RegisterModel`; it does not decide when individual AIServer
Agents activate the model.

## 5. Run it directly

```bash
# When build/maze_model_distributor already exists
bash ./run.sh
```

During full training the Learner container supervises this process; no separate launch is required.

## License

[MIT License](LICENSE)
