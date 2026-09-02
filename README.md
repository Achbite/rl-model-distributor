# RL Model Distributor

简体中文 | [English](README.en.md)

## 1. 运行测试

在已经挂载 Contracts C++ bindings 的构建容器中执行：

```bash
bash ./test.sh
```

`test.sh` 是本仓库的统一测试入口，并按显式清单运行当前开发校验。

## 2. 构建 0.15.0 制品

先生成任务无关的训练协议制品，再从当前源码执行：

```bash
(cd ../rl-contracts && bash build_artifact.sh training)
bash build_artifact.sh
```

构建脚本只编译正式二进制；测试仍只通过第 1 节的 `bash ./test.sh` 显式运行。输出：

```text
../.workspace/artifacts/rl-model-distributor/0.15.0/<platform>/
```

## 3. 装配到 Learner

推荐使用 Learner 同步入口复制完整制品：

```bash
(cd ../rl-learner && bash scripts/sync_runtime_artifacts.sh)
```

开发制品可由 Learner 的 `make deps` 统一构建和同步；`make shell` 不会隐式执行该操作。

## 4. 模型 manifest

Learner 当前的本地发布布局示例为：

```text
models/train/<step:min-width-7>/SaveModel.onnx
```

`SaveModel.onnx` 和七位目录只是 Learner 当前内部布局，不是 Distributor 的注册合同。
`RegisterModel` 接受调用方在已配置 artifact root 内显式给出的绝对、常规且非符号链接的模型文件路径；
`manifest.pb` 是唯一权威模型 manifest。Distributor 校验 lineage/step、发布时间、声明字节数、
artifact root 内的常规非符号链接文件路径，以及打开后文件大小；它不计算或校验内容摘要。
模型下载按已打开文件句柄、模型身份、offset 与声明字节数传输，任务语义和 ONNX 名称/shape/type
由消费模型的 AIServer 判断。ACK 绑定 AIServer 服务生命周期、`load_instance_id` 与已注册模型
身份，Distributor 不决定各 Agent 的激活时机。

## 5. 单独运行

```bash
# 已有本地 build/maze_model_distributor 时
bash ./run.sh
```

完整训练由 Learner 容器监管该进程，不需要单独启动。

## License

[MIT License](LICENSE)
