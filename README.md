# RL Model Distributor

简体中文 | [English](README.en.md)

C++ 模型注册、下载与 AIServer 加载 ACK 服务，当前版本为 `0.8.0`。服务只解释 `rl.training.v1` 的模型身份，不依赖任何 Maze 任务协议。

## 构建

先构建 `rl-contracts`，再执行：

```bash
bash build_artifact.sh
```

输出目录：

```text
../.workspace/artifacts/rl-model-distributor/0.8.0/<platform>/
```

运行时必须使用配置文件中的完整 `rl-contracts 0.8.0` 制品身份；缺失、旧版本或 digest 不匹配会拒绝启动或注册。

```bash
./maze_model_distributor configs/model_distributor_config.yaml
```

## License

[MIT License](LICENSE)
