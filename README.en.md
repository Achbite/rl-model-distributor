# RL Model Distributor

English | [简体中文](README.md)

C++ model registration, download, and AIServer load-ACK service at version `0.8.1`. It consumes only `rl.training.v1` model identities and has no Maze task-protocol dependency.

## Build

Build `rl-contracts` first, then run:

```bash
bash build_artifact.sh
```

Output directory:

```text
../.workspace/artifacts/rl-model-distributor/0.8.1/<platform>/
```

Creating a version for the first time requires a reviewed and committed clean
Git savepoint. The build entrypoint refuses to create a new artifact from a
dirty worktree. An existing artifact can only be reused when both source
identity and every file checksum still match.

Runtime configuration must contain the complete `rl-contracts 0.9.1` artifact identity. Missing, old, or digest-mismatched identities fail closed.

```bash
./maze_model_distributor configs/model_distributor_config.yaml
```

## License

[MIT License](LICENSE)
