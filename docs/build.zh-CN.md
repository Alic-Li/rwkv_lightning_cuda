# 构建 RWKV Lightning CUDA

[English](build.md) | 简体中文 | [返回 README](../README_zh.md)

## CMake 构建

```bash
cmake -S . -B ./build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;87;89;90;100;120"

cmake --build ./build -j --config Release
cmake --build ./build -j --config Release --target bundle_all
```

`bundle_all` 会在 `build/bundle/` 下组装所有可用的运行时目标。Server 与 state-tuning
bundle 自带 `rwkv_vocab_v20230424.txt`；共享运行时库放在各 bundle 的 `lib/` 目录下。
单独的 `bundle_<target>` 目标依然可用。

## CUDA W8A16 / W4A16 量化

构建会产出 `build/bundle/rwkv_quantize/rwkv_quantize`，这是一个独立的转换器，用于
BF16 RWKV checkpoint。它输出流式 `.rwkvq` 文件：权重可以是按输出通道的 INT8，也可以
是分组 INT4 加 FP16 scale；embedding、layer norm、LoRA 因子及其他非线性张量保持
BF16。

```bash
./build/bundle/rwkv_quantize/rwkv_quantize /path/to/model.pth /path/to/model.w8a16.rwkvq

# W4，每 128 个权重共享一个 FP16 scale（推荐默认值）
./build/bundle/rwkv_quantize/rwkv_quantize \
  --format w4a16 --group-size 128 \
  /path/to/model.pth /path/to/model.w4a16.rwkvq
```

CUDA 推理后端会自动识别 `.rwkvq` 中张量的 dtype，把打包后的 INT4/INT8 权重保留在显存
中，并对 attention 投影、FFN 投影和输出 head 分别调度 W4A16 或 W8A16。W4 还支持
`--group-size 32`，在愿意付出更多 scale 代价时换取更高保真度。旧的「两个位置参数」
命令仍然导出 W8A16。HIP 同样支持 BF16/PTH 和 W4A16/W8A16 `.rwkvq` 推理，详见
[W7900 量化验证](../hip/QUANTIZATION.zh-CN.md)。

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
