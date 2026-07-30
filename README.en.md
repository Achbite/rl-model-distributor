# RL Model Distributor

English | [简体中文](README.md)

C++ model registration and download artifact repository, currently at version `0.2.0`. The generated ModelDistributor binary is used by the Learner image.

## Build

Build `rl-contracts` first, then run:

```bash
bash build_artifact.sh
```

Output directory:

```text
../.workspace/artifacts/rl-model-distributor/0.2.0/<platform>/
```

Copy the artifact for the selected platform into `rl-learner/model-distributor/` before building the Learner image.

## License

[MIT License](LICENSE)
