<div align="center">
  <img src="assets/banner.png" alt="RWKV Lightning CUDA — 推理、状态微调、量化和 Web UI" width="820">

  [![CI and Release](https://github.com/Alic-Li/rwkv_lightning_cuda/actions/workflows/ci.yml/badge.svg)](https://github.com/Alic-Li/rwkv_lightning_cuda/actions/workflows/ci.yml)
  ![C++ 20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)
  ![CUDA](https://img.shields.io/badge/CUDA-12.9%20%C2%B7%2013.2-76B900?logo=nvidia&logoColor=white)
  ![ROCm](https://img.shields.io/badge/ROCm-7.2.5-ed1c24?logo=amd&logoColor=white&style=flat&height=20)
</div>

# RWKV Lightning CUDA

[English](README.md)

RWKV-7 高性能 GPU 推理服务器，主要支持 NVIDIA CUDA，同时提供 AMD HIP
后端。项目包含兼容 OpenAI 的接口、原生批量与流式接口、L1/L2/SQLite
会话状态缓存、W8A16/W4A16 量化推理、state tuning、MiSS adapter、负载均衡
路由器以及带 Web UI 的桌面启动器。

两个训练二进制已切换到独立的 [BF16 训练 backbone](src/bf16_training/README.md)：基模逐层流式加载，直接复用原有 BF16 state-passing 算子，最终 state/adapter 导出为 BF16；推理仍使用原有 FP16/量化路线。

## 主要功能

- 分块 prefill、逐 token decode、SSE 流式输出和显存自适应准入。
- 在显存、CPU RAM 和 SQLite 之间复用会话 state，并支持上传 `.pth` 初始 state。
- 使用 `.rwkvq` 格式进行 W8A16/W4A16 推理。
- 独立训练 `time_state`，不更新 Linear 权重。
- 冻结基模训练 MiSS adapter，并按请求动态绑定不同 adapter。
- 多后端路由和带 Web UI 的桌面启动器。

## 快速开始

```bash
cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="75;80;86;87;89;90;100;120"
cmake --build ./build -j --config Release --target bundle_all

./build/bundle/rwkv_lighting_cuda/rwkv_lighting_cuda \
  --model-path /path/to/model.pth \
  --vocab-path ./assets/rwkv_vocab_v20230424.txt \
  --host 127.0.0.1 \
  --port 8000
```

检查服务状态：

```bash
curl -sS "http://127.0.0.1:8000/v1/server/status"
```

## 中文文档

| 文档 | 内容 |
|---|---|
| [构建](docs/build.zh-CN.md) | CUDA、ROCm、Windows、量化工具和打包方式 |
| [运行](docs/run.zh-CN.md) | 服务参数、动态模型加载、state tuning 和运行示例 |
| [HTTP API](docs/http-api.zh-CN.md) | 生成、流式、批量、state 和管理接口 |
| [MiSS 适配器与参数高效微调](src/miss/README.zh-CN.md) | 训练、续训、导出、动态加载、缓存和验收 |
| [状态微调](src/state_tuning/README.zh-CN.md) | `time_state` 训练与检查点说明 |
| [MiSS 验证记录](src/miss/VALIDATION.zh-CN.md) / [优化计划](src/miss/OPTIMIZATION_PLAN.zh-CN.md) | 数值验证、实测性能和后续优化目标 |
| [HIP 后端](hip/README.zh-CN.md) / [移植验证](hip/VALIDATION.zh-CN.md) | AMD 构建、状态微调与验证范围 |
| [W8A16 / W4A16 量化](quant/README.zh-CN.md) / [W7900 量化验证](hip/QUANTIZATION.zh-CN.md) | 量化格式、精度与吞吐实测 |
| [多后端路由器](RWKV_Lightning_CUDA_router/README.zh-CN.md) | 负载均衡、会话亲和和状态文件广播 |
| [Windows 构建与运行](docs/windows-build-run.zh-CN.md) | Windows 环境、编译、打包和启动 |
| [CI 与版本发布](docs/releasing.zh-CN.md) | CI 检查、安装包和版本发布流程 |
| [桌面启动器](RWKV_Lightning_CUDA_Launcher/README.md) | Web 界面、模型管理、聊天和训练 |
| [W4A16 CUDA 算子](quant/gemmv/README_w4a16.md) / [算子审计](docs/kernel-audit-2026-09-12.md) | 量化算子实现、优化与验证 |
| [W7900 风扇控制](docs/w7900-fan-control.zh-CN.md) | Linux 下的 AMD 风扇配置 |

完整接口字段还可参考 [API 参考](rwkv_lightning_api_doc.md)。英文主文档中的
[Documentation](README.md#documentation) 只列英文页面，中文入口统一放在本页，
避免两种语言混在同一个目录表中。

## 项目目录

- `include/rwkv/`：公共头文件。
- `src/backend/`：CUDA/HIP 模型后端。
- `src/inference/`：分词、采样和生成编排。
- `src/io/`：PTH 与张量读写。
- `src/server/`：HTTP API、模型路由、准入和 state 存储。
- `src/miss/`：MiSS 训练、推理包和动态 adapter 文档。
- `quant/`：W8A16/W4A16 量化工具。
- `RWKV_Lightning_CUDA_router/`：多后端负载均衡路由器。
- `RWKV_Lightning_CUDA_Launcher`: 带 Web UI 的桌面启动器。
