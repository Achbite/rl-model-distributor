# RL Model Distributor

[简体中文](README.md) | English

## 1. Build the 0.11.0 artifact

Create the Contracts artifact first, then run the build from a clean Git savepoint:

```bash
(cd ../rl-contracts && bash build_artifact.sh)
bash build_artifact.sh
```

The script compiles the service and runs its tests. Output:

```text
../.workspace/artifacts/rl-model-distributor/0.11.0/<platform>/
```

## 2. Stage it into Learner

Do not copy the binary manually. Use the Learner sync entry point:

```bash
(cd ../rl-learner && bash scripts/sync_runtime_artifacts.sh)
```

## 3. Validate a model manifest

```bash
python3 scripts/validate_manifest.py tests/manifest.example.json
```

Model paths must use:

```text
archive/<step:min-width-6>/SaveModel.onnx
```

## 4. Run it directly

```bash
./maze_model_distributor configs/model_distributor_config.yaml
```

During full training the Learner container supervises this process; no separate launch is required.

## License

[MIT License](LICENSE)
