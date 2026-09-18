<div align="center">
  <img src="assets/banner.png" alt="RWKV Lightning CUDA — Inference · State Tuning · Quantization · Web UI" width="820">

  [![CI and Release](https://github.com/Alic-Li/rwkv_lightning_cuda/actions/workflows/ci.yml/badge.svg)](https://github.com/Alic-Li/rwkv_lightning_cuda/actions/workflows/ci.yml)
  ![C++ 20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)
  ![CUDA](https://img.shields.io/badge/CUDA-12.9%20%C2%B7%2013.2-76B900?logo=nvidia&logoColor=white)
  ![ROCm](https://img.shields.io/badge/ROCm-7.2.5-ed1c24?logo=amd&logoColor=white&style=flat&height=20)
  [![Release](https://img.shields.io/github/v/tag/Alic-Li/rwkv_lightning_cuda?sort=semver&label=release)](https://github.com/Alic-Li/rwkv_lightning_cuda/releases)
</div>

# RWKV Lightning CUDA

[中文说明](README_zh.md)

High-performance RWKV-7 inference server for NVIDIA CUDA GPUs, with an AMD HIP
fallback path. It ships OpenAI-style and native batch APIs with streaming SSE,
an L1/L2/SQLite session state cache, W8A16/W4A16 quantized serving, standalone
state tuning, a multi-backend Go router, and a Go desktop launcher with a Web UI.

- **Fast prefill and decode** — chunked prefill with VRAM-adaptive admission control, SSE streaming with configurable chunk size.
- **Session state cache** — reuse conversation states across L1 VRAM, L2 RAM, and SQLite persistence; uploaded `.pth` states work with every generation endpoint.
- **Quantization** — W4A16/W8A16 `.rwkvq` checkpoints with packed INT4/INT8 weights on device.
- **State tuning** — train `time_state` blocks standalone from JSONL data without touching linear weights.
- **MiSS adapters&peft** — train one compact `D` matrix per target linear, then load adapters per request without modifying shared model weights.
- **Batteries included** — weighted least-inflight load-balancing router and a desktop launcher with Web UI.

Both training binaries use the independent [BF16 training backbone](src/bf16_training/README.md), with streamed base loading, native state passing and BF16 exports. Inference keeps its existing FP16/quantized paths.

## Quick Start

```bash
cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="75;80;86;87;89;90;100;120"
cmake --build ./build -j --config Release --target bundle_all

./build/bundle/rwkv_lighting_cuda/rwkv_lighting_cuda \
  --model-path /path/to/model.pth \
  --vocab-path ./assets/rwkv_vocab_v20230424.txt \
  --host 127.0.0.1 \
  --port 8000
```

Verify the server is up:

```bash
curl -sS "http://127.0.0.1:8000/v1/server/status"
```

All build variants (Windows, AMD ROCm, the quantization tool, the Go launcher)
are described in the [build guide](docs/build.md).

## Documentation

| Guide | Description |
|---|---|
| [Build](docs/build.md) | CMake build for CUDA and ROCm, W8A16/W4A16 quantization tool, Go launcher; step-by-step [Windows guide](docs/windows-build-run.md) |
| [Run](docs/run.md) | Server flags, Windows runtime, dynamic model loading, standalone state tuning |
| [HTTP API](docs/http-api.md) | curl examples for every endpoint; complete reference in [rwkv_lightning_api_doc.md](rwkv_lightning_api_doc.md) |
| [MiSS adapters&peft](src/miss/README.md) | Frozen-base adapter training, checkpoint/resume, export, dynamic serving, caches, and validation |
| [State tuning](src/state_tuning/README.md) | Standalone `time_state` training and checkpoint format |
| [Router](RWKV_Lightning_CUDA_router/README.md) | Multi-backend load-balancing reverse proxy with session affinity and state fan-out |
| [Releasing](docs/releasing.md) | CI packaging matrix and how to publish a versioned release |

## CI / Release

GitHub Actions builds Linux and Windows CUDA packages on PRs and `main` against
a CUDA 12.9 / 13.2 matrix. Bump the root [`VERSION`](VERSION) file on `main` to
publish a new Release with Linux/Windows binaries and SHA-256 checksums; see
[the release guide](docs/releasing.md) for the workflow.

## Project layout

- `include/rwkv/`: public headers grouped into `common`, `io`, `runtime`, `inference`, and `server` APIs.
- `src/backend/`: GPU model backend integration.
- `src/inference/`: tokenization, sampling, and generation orchestration.
- `src/io/`: PTH archive and tensor readers.
- `src/server/`: HTTP API, model routing, admission control, and state storage.
- `src/app/`: executable entry points.
- `assets/`: runtime data files such as the tokenizer vocabulary, plus the README banner image.
- `cmake/`: dependency, compiler-option, and packaging modules.

The root `CMakeLists.txt` only selects the GPU backend and composes these modules. Target definitions live next to their corresponding source trees in `src/CMakeLists.txt`, `tools/CMakeLists.txt`, and `test/CMakeLists.txt`.
