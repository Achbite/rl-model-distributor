# RL Model Distributor

简体中文 | [English](README.en.md)

C++ 模型注册、下载与 AIServer 加载 ACK 服务，当前版本为 `0.10.1`。服务只解释 `rl.training.v1` 的模型身份，不依赖任何 Maze 任务协议。

## 构建

先构建 `rl-contracts`，再执行：

```bash
bash build_artifact.sh
```

输出目录：

```text
../.workspace/artifacts/rl-model-distributor/0.10.1/<platform>/
```

首次生成一个版本的制品时，仓库必须位于已经审核并提交的 clean Git 保存点；构建
入口拒绝从 dirty worktree 创建新版本。已存在的同版本制品只允许在源码身份和文件
checksum 全部一致时复用。

运行时必须使用配置文件中的完整 `rl-contracts 0.10.0` 制品身份；缺失、旧版本或 digest 不匹配会拒绝启动或注册。

```bash
./maze_model_distributor configs/model_distributor_config.yaml
```

## License

[MIT License](LICENSE)
