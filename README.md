# RL Model Distributor

简体中文 | [English](README.en.md)

C++ 模型注册与下载服务制品仓库，当前版本为 `0.4.0`。生成的 ModelDistributor 二进制由 Learner 镜像使用。

## 构建

先构建 `rl-contracts`，再执行：

```bash
bash build_artifact.sh
```

输出目录：

```text
../.workspace/artifacts/rl-model-distributor/0.4.0/<platform>/
```

将对应平台的制品显式复制到 `rl-learner/model-distributor/` 后，再构建 Learner 镜像。

## License

[MIT License](LICENSE)
