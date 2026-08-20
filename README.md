# RL Model Distributor

简体中文 | [English](README.en.md)

## 1. 运行测试

在已经挂载 Contracts C++ bindings 的构建容器中执行：

```bash
bash ./test.sh
```

`test.sh` 是本仓库的统一测试入口，并按显式清单运行当前开发校验。

## 2. 构建 0.14.0 制品

先生成 Contracts 制品，再从 clean Git 保存点执行：

```bash
(cd ../rl-contracts && bash build_artifact.sh)
bash build_artifact.sh
```

构建脚本会编译并运行测试。输出：

```text
../.workspace/artifacts/rl-model-distributor/0.14.0/<platform>/
```

## 3. 装配到 Learner

不要手工复制二进制。使用 Learner 同步入口：

```bash
(cd ../rl-learner && bash scripts/sync_runtime_artifacts.sh)
```

## 4. 校验模型 manifest

```bash
python3 scripts/validate_manifest.py /path/to/manifest.json
```

模型路径必须使用：

```text
models/train/<step:min-width-7>/SaveModel.onnx
```

0.14 manifest schema v3 还必须绑定完整 `RolloutEstimatorProfile`；Distributor 校验 profile、
训练语义、lineage、step 与 digest，但不决定 AIServer 中各 Agent 的激活时机。

## 5. 单独运行

```bash
# 已有本地 build/maze_model_distributor 时
bash ./run.sh
```

完整训练由 Learner 容器监管该进程，不需要单独启动。

## License

[MIT License](LICENSE)
