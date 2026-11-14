# 构建 RWKV Lightning CUDA

[English](build.md) | 简体中文 | [返回 README](../README_zh.md)

## 前置依赖

- CMake 3.24 或更新版本，以及支持 C++20 的 C++ 编译器。
- GPU 工具链二选一：NVIDIA CUDA Toolkit（项目 CI 覆盖 CUDA 12.9 和 13.2）；或 ROCm 7.2.5，以及 AMD 后端所需的 HIP、hipBLAS、hipBLASLt 和 hipRAND。
- SQLite3、OpenSSL、JsonCpp 和 zlib 的开发文件。CMake 会优先通过 JsonCpp 的 config 包查找；回退时使用 `pkg-config`（只有回退路径需要安装 `pkg-config`）。
- 构建 `rwkv_lighting_cuda` HTTP 服务端目标需要 Drogon。未安装 Drogon 时，CMake 仍会构建其他可用目标，但会跳过服务端。

Debian 或 Ubuntu 上，完整 CUDA 构建所需的主机端依赖可通过以下命令安装：

```bash
sudo apt install build-essential cmake pkg-config libsqlite3-dev libssl-dev libjsoncpp-dev zlib1g-dev libdrogon-dev
```

NVIDIA 驱动和 CUDA Toolkit 请根据系统按照 NVIDIA 官方说明单独安装。构建 HIP 后端时，安装对应版本的 ROCm 开发包并使用 HIP 工具链；详见 [AMD ROCm 部分](#amd-rocm-hip)。

## CMake 构建

```bash
cmake -S . -B ./build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;87;89;90;100;120"

cmake --build ./build -j --config Release
cmake --build ./build -j --config Release --target bundle_all
```

`bundle_all` 会在 `build/bundle/` 下组装所有可用的运行时目标。所有可执行文件与 `rwkv_vocab_v20230424.txt` 直接放在该目录，
运行时依赖统一放在 `build/bundle/lib/`，由全部程序共用。
单独的 `bundle_<target>` 目标依然可用。

## CUDA W8A16 与 SB6 W4/W6 量化

原始 W8 导出不需要校准，保留双位置参数用法：

```bash
./build/bundle/rwkv_quantize model.pth model.w8a16.rwkvq
```

SB6 W4/W6 使用 Python 离线 AW/最小二乘校准，再由 C++ 验证并打包。
参考[量化指南](../quant/README.zh-CN.md)的完整命令。Python 校准需要 PyTorch；
C++ 导出与推理不依赖 PyTorch。CUDA 自动识别 `.rwkvq` 中的 W8/SB6 权重。
HIP 仅支持 W8 量化推理，详见 [HIP 支持说明](../hip/QUANTIZATION.zh-CN.md)。
旧对称 W4 与 SB6 W5 文件不再支持，需从来源 PTH 重新导出。

## Windows

```powershell
$env:CudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\"
cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="75;80;86;87;89;90;100;120" -DCMAKE_TOOLCHAIN_FILE="D:/vcpkg/scripts/buildsystems/vcpkg.cmake"  -DCMAKE_CXX_FLAGS="/Zc:preprocessor" -DCMAKE_CUDA_FLAGS="-Xcompiler=/Zc:preprocessor"

cmake --build ./build --config Release -j --target bundle_rwkv_quantize bundle_rwkv_lighting_cuda
```

在 Windows 10 + RTX 3080 环境下逐步验证过的完整构建与运行指南见
[Windows 构建运行指南](windows-build-run.zh-CN.md)。

## AMD ROCm (HIP)

```bash
cmake -S . -B build-hip -DRWKV_GPU_BACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j
```

## Go Web 前端（launcher）

在 `RWKV_Lightning_CUDA_Launcher/` 内执行，先安装 Bun，详见[前端开发指南](../RWKV_Lightning_CUDA_Launcher/docs/frontend-development.md)。先构建前端，再编译 Go：

```bash
bun install --frozen-lockfile
bun run build
```

Linux GPU 指标需要 `CGO_ENABLED=1` 与 gcc/libdl；下方通用构建禁用 NVML。

```bash
## Linux
CGO_ENABLED=0 go build -ldflags="-s -w" -o rwkv_launcher .
```

```powershell
## Windows
$env:CGO_ENABLED="0"
go build -trimpath -ldflags="-s -w" -o .\rwkv_launcher.exe .
```
