# Building RWKV Lightning CUDA

[English](build.md) | [简体中文](build.zh-CN.md) | [Back to README](../README.md)

```bash
cmake -S . -B ./build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;87;89;90;100;120"

cmake --build ./build -j --config Release
cmake --build ./build -j --config Release --target bundle_all
```

`bundle_all` assembles every available runtime target below `build/bundle/`. Server and state-tuning bundles include `rwkv_vocab_v20230424.txt`; shared runtime libraries are placed in each bundle's `lib/` directory. Individual `bundle_<target>` targets remain available.

## CUDA W8A16 / W4A16 quantization

The build produces `build/bundle/rwkv_quantize/rwkv_quantize`, a standalone converter for BF16 RWKV
checkpoints. It writes a streaming `.rwkvq` file containing either per-output-channel
INT8 weights or grouped INT4 weights with FP16 scales; embeddings, layer norms,
LoRA factors, and other non-linear tensors remain BF16.

```bash
./build/bundle/rwkv_quantize/rwkv_quantize /path/to/model.pth /path/to/model.w8a16.rwkvq

# W4 with one FP16 scale for each 128-weight group (recommended default)
./build/bundle/rwkv_quantize/rwkv_quantize \
  --format w4a16 --group-size 128 \
  /path/to/model.pth /path/to/model.w4a16.rwkvq
```

The CUDA inference backend detects the tensor dtype in `.rwkvq` automatically,
keeps packed INT4 or INT8 weights on device, and dispatches W4A16 or W8A16 for
attention projections, FFN projections, and the output head. W4 also supports
`--group-size 32` when higher fidelity is worth the extra scales. Existing
two-positional-argument commands still export W8A16. HIP builds support both BF16/PTH and W4A16/W8A16 `.rwkvq` inference; see
the [W7900 quantization report](../hip/QUANTIZATION.md).

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

```bash
## Linux
CGO_ENABLED=0 go build -ldflags="-s -w" -o rwkv_launcher main.go
```

```powershell
## Windows
$env:CGO_ENABLED="0"
go build -trimpath -ldflags="-s -w" -o .\rwkv_launcher.exe .\main.go
```
