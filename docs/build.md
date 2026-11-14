# Building RWKV Lightning CUDA

[English](build.md) | [简体中文](build.zh-CN.md) | [Back to README](../README.md)

## Prerequisites

- CMake 3.24 or newer and a C++20-capable compiler.
- One GPU toolchain: NVIDIA CUDA Toolkit (the project CI covers CUDA 12.9 and 13.2), or ROCm 7.2.5 with HIP, hipBLAS, hipBLASLt, and hipRAND for the AMD backend.
- SQLite3, OpenSSL, JsonCpp, and zlib development files. CMake discovers JsonCpp through its config package or, as a fallback, `pkg-config` (`pkg-config` is only needed for that fallback).
- Drogon is needed to build the `rwkv_lighting_cuda` HTTP server target. Without it, CMake still builds the other available targets but skips the server.

On Debian or Ubuntu, the host-side dependencies for the complete CUDA build can be installed with:

```bash
sudo apt install build-essential cmake pkg-config libsqlite3-dev libssl-dev libjsoncpp-dev zlib1g-dev libdrogon-dev
```

Install the NVIDIA driver and CUDA Toolkit separately, following NVIDIA's instructions for your system. For HIP builds, install the matching ROCm development packages instead of CUDA; see the [AMD ROCm section](#amd-rocm-hip).

```bash
cmake -S . -B ./build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;87;89;90;100;120"

cmake --build ./build -j --config Release
cmake --build ./build -j --config Release --target bundle_all
```

`bundle_all` assembles every available runtime target below `build/bundle/`. Executables and `rwkv_vocab_v20230424.txt` sit directly in that directory; all executables share a single `build/bundle/lib/` directory for runtime libraries. Individual `bundle_<target>` targets remain available.

## CUDA W8A16 and SB6 W4/W6 quantization

Original W8 export requires no calibration and retains the two-argument command:

```bash
./build/bundle/rwkv_quantize model.pth model.w8a16.rwkvq
```

SB6 W4/W6 uses offline Python AW/least-squares calibration followed by the C++
archive writer. See the [quantization guide](../quant/README.md) for commands.
Only offline calibration needs PyTorch; the C++ exporter and runtime do not.
CUDA detects W8/SB6 tensors automatically. HIP quantized inference supports W8
only; see [HIP support](../hip/QUANTIZATION.md). Retired symmetric W4 and SB6 W5
archives must be exported again from the source checkpoint.

## Windows

```powershell
$env:CudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\"
cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="75;80;86;87;89;90;100;120" -DCMAKE_TOOLCHAIN_FILE="D:/vcpkg/scripts/buildsystems/vcpkg.cmake"  -DCMAKE_CXX_FLAGS="/Zc:preprocessor" -DCMAKE_CUDA_FLAGS="-Xcompiler=/Zc:preprocessor"

cmake --build ./build --config Release -j --target bundle_rwkv_quantize bundle_rwkv_lighting_cuda
```

A step-by-step guide verified on Windows 10 with an RTX 3080 lives in
[windows-build-run.md](windows-build-run.md).

## AMD ROCm (HIP)

```bash
cmake -S . -B build-hip -DRWKV_GPU_BACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j
```

## Go Web Frontend (launcher)

Run from `RWKV_Lightning_CUDA_Launcher/`. Install Bun first; see the [frontend guide](../RWKV_Lightning_CUDA_Launcher/docs/frontend-development.md). Build the frontend before compiling Go:

```bash
bun install --frozen-lockfile
bun run build
```

Linux GPU metrics require `CGO_ENABLED=1` and gcc/libdl; the portable build below disables NVML.

```bash
## Linux
CGO_ENABLED=0 go build -ldflags="-s -w" -o rwkv_launcher .
```

```powershell
## Windows
$env:CGO_ENABLED="0"
go build -trimpath -ldflags="-s -w" -o .\rwkv_launcher.exe .
```
