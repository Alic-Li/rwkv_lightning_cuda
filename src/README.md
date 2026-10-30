# 源码结构

源码按职责组织。BF16 是训练实现的数值类型，不再作为容纳所有训练代码的顶层目录。

| 目录 | 职责 |
| --- | --- |
| `app/` | 可执行程序入口；`rwkv_tune.cpp` 共享 state、MiSS、NoRA 的参数解析和训练循环，`rwkv_fast_server.cpp` 启动推理服务 |
| `training/` | 训练执行调度、chunk/tape 生命周期、适配器参数管理和优化器调用；`adapter_checkpoint.cpp` 管理训练恢复和导出流程 |
| `model/bf16/` | RWKV block/model 的 BF16 前向、反向组合与工作区布局 |
| `backend/bf16/` | BF16 基础 GPU 算子、反向算子、损失与优化器内核；不负责文件或 CLI |
| `backend/` | 现有推理后端实现 |
| `io/` | PTH 档案、张量、指纹；`bf16_model_loader.cpp` 流式加载冻结基模，`bf16_state.cpp` 读写 state，`bf16_adapter_writer.cpp` 编码适配器格式 |
| `miss/`、`nora/` | 各适配器算法；`kernels.cu` 用于推理，`training_kernels.cu` 用于 BF16 训练 |
| `state_tuning/` | 训练数据集加载与使用文档 |
| `inference/` | 推理引擎、tokenizer 和 sampler 接口实现 |
| `server/` | 推理 HTTP API、路由、缓存与资源管理 |
| `online_rl/` | 复用训练库和推理后端的在线训练服务 |
| `*_tune_server/` | 各训练 HTTP 服务的启动入口 |
| `utils/` | tokenizer、采样等现有基础实现 |

## 构建与依赖

`src/CMakeLists.txt` 组装推理核心和服务；开启 `RWKV7_STATE_TUNING` 后，
`training/CMakeLists.txt` 组装训练库，`app/CMakeLists.txt` 构建三个训练 CLI。
CLI 和训练服务调用训练执行层；执行层组织模型前后向；模型层调用基础算子和
MiSS/NoRA 算子。文件转换集中在 `io/`，checkpoint 的训练配置与恢复流程归训练层。

训练库仍为独立的 `rwkv_bf16_training` 静态库，并提供 `rwkv::training` 别名。
它不链接 FP16/量化推理核心。两个核心继续各自编译所需的公共 I/O 和 tokenizer
源码；本次没有引入新的跨后端链接依赖。WKV 仍复用仓库根目录 `cuda/`、`hip/`
中的 state-passing 实现。HIP 构建将训练 `.cu` 源文件标记为 HIP 语言。

公开 API 继续使用 `include/rwkv/bf16_training/` 和原有命名空间，避免迫使 CLI、
服务、测试及外部调用方同时迁移。公开头文件按接口保持兼容，源文件按实现职责归属。
可执行文件名、CLI 参数、checkpoint/PTH 格式和数值计算顺序保持不变。

训练精度、加载策略和验证说明见 [训练文档](training/README.md)。
