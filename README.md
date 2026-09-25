# RL Model Distributor

简体中文 | [English](README.en.md)

## 1. 运行测试

在已经挂载 Contracts C++ bindings 的构建容器中执行：

```bash
bash ./test.sh
```

`test.sh` 是本仓库的统一测试入口，并按显式清单运行当前开发校验。

## 2. 构建当前制品

显式提供已经生成的 Contracts Training 制品目录（其中包含 `cpp/` 与 `sdk/`）：

```bash
CONTRACT_CPP_DIR=/path/to/training/cpp bash ./build.sh
```

本仓只构建自身，不构建或同步上游。构建脚本只编译正式二进制；测试仍通过 `bash ./test.sh` 显式运行。
默认输出到 `build/artifacts/production/<platform>/`；调用者可以用 `RL_ARTIFACT_ROOT` 指定输出根目录。
`bash build_dev_artifact.sh` 使用相同实现，默认输出到 `build/artifacts/development/<platform>/current/`。

## 3. 装配到 Learner

在遵循 Framework 同级仓库布局的工作区，从 `rl-framework` 执行：

```bash
bash ./sync_artifacts.sh
# 开发制品通道
bash ./sync_artifacts.sh --development
```

Framework 显式调用 Contracts、Pool 和 Distributor 各自的构建入口，再将所选制品装配到 Learner。
组件仓不依赖这种目录布局，单独构建也不会修改 Learner。

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

## 构建与运行路径

CMake/Ninja 构建目录位于 `build/<platform>/<builder-image-id>/<channel>/`，ccache 使用持久 Docker 卷。构建成功后 `build/maze_model_distributor` 指向本次二进制，同时导出制品；在相同 Linux 工具链环境中执行 `bash run.sh` 使用该输出。macOS 宿主负责 Docker 构建，不直接执行 Linux ELF。正常重建不清理缓存；构建镜像身份仅隔离工具链缓存，不参与运行准入。

Linux 容器构建前用 `stat --cached=never` 刷新实际输入的文件属性，避免 virtiofs 暂存旧 mtime 导致刚保存的宿主源码被误判为 no-op；不会修改源码时间戳或同步上游制品。
