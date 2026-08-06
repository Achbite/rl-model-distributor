# RL Model Distributor

English | [简体中文](README.md)

C++ model registration, download, and AIServer load-ACK service at version `0.8.0`. It consumes only `rl.training.v1` model identities and has no Maze task-protocol dependency.

## Build

Build `rl-contracts` first, then run:

```bash
bash build_artifact.sh
```

Output directory:

```text
../.workspace/artifacts/rl-model-distributor/0.8.0/<platform>/
```

Runtime configuration must contain the complete `rl-contracts 0.8.0` artifact identity. Missing, old, or digest-mismatched identities fail closed.

```bash
./maze_model_distributor configs/model_distributor_config.yaml
```

## License

[MIT License](LICENSE)
