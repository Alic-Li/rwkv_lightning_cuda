# Windows build and run guide

This document records the Windows build that was verified on Windows 10 with an RTX 3080.

## Verified environment

- OS: Windows 10 Pro 22H2, build 19045
- GPU: NVIDIA GeForce RTX 3080, compute capability 8.6
- NVIDIA driver: 576.57
- CUDA Toolkit: 12.9
- Visual Studio: Visual Studio 2022 with MSVC x64 toolchain
- CMake: 4.0.3 or newer
- Go: 1.24.4 or newer, only required for `rwkv_launcher.exe`
- vcpkg: current enough to install `drogon:x64-windows`

The exact GPU architecture used for this build is `86`. For other NVIDIA GPUs, change `CMAKE_CUDA_ARCHITECTURES` to the matching architecture, or use a multi-architecture list for a wider release package.

## Required build dependencies

Install these tools before building:

- Visual Studio 2022, including `Desktop development with C++`
- Windows 10 or Windows 11 SDK from the Visual Studio installer
- NVIDIA CUDA Toolkit 12.9
- CMake
- Git
- vcpkg
- Go, if the web launcher should be built

Install the C++ dependencies with vcpkg:

```powershell
C:\vcpkg\vcpkg.exe install sqlite3:x64-windows drogon:x64-windows
```

`drogon:x64-windows` brings the runtime dependencies used by the server, including Trantor, OpenSSL, zlib, c-ares, Brotli, and JsonCpp.

## Build the CUDA backend

Run from the repository root:

```powershell
cd D:\repo\rwkv_lightning_cuda

$env:CudaToolkitDir = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\"

cmake -S . -B .\build_win10_sm86 `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CUDA_ARCHITECTURES="86" `
  -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake" `
  -DCMAKE_CXX_FLAGS="/Zc:preprocessor" `
  -DCMAKE_CUDA_FLAGS="-Xcompiler=/Zc:preprocessor" `
  -DRWKV7_FAST_BUILD_TESTS=OFF

cmake --build .\build_win10_sm86 --config Release -j --target bundle_rwkv_lighting_cuda
```

For a wider binary package, replace `86` with a list such as:

```powershell
-DCMAKE_CUDA_ARCHITECTURES="75;80;86;87;89;90"
```

The backend executable is generated at:

```text
build_win10_sm86\bundle\rwkv_lighting_cuda\rwkv_lighting_cuda.exe
```

## Build the web launcher

Run from the repository root:

```powershell
cd D:\repo\rwkv_lightning_cuda\RWKV_Lightning_CUDA_Launcher

$env:CGO_ENABLED = "0"

go build -trimpath -ldflags="-s -w" `
  -o ..\build_win10_sm86\bundle\rwkv_lighting_cuda\rwkv_launcher.exe `
  .\main.go
```

The launcher starts an HTTP control page on `http://127.0.0.1:8088`. On Windows, it prepends the bundled `lib` directory to the child backend process `PATH`.

## Complete the runtime bundle

The CMake bundle target copies the backend executable and many vcpkg runtime DLLs into:

```text
build_win10_sm86\bundle\rwkv_lighting_cuda
```

Copy the CUDA runtime DLLs and vocab file into the bundle:

```powershell
$bundle = "D:\repo\rwkv_lightning_cuda\build_win10_sm86\bundle\rwkv_lighting_cuda"
$cudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"

Copy-Item "$cudaBin\cudart64_12.dll" -Destination "$bundle\lib" -Force
Copy-Item "$cudaBin\cublas64_12.dll" -Destination "$bundle\lib" -Force
Copy-Item "$cudaBin\cublasLt64_12.dll" -Destination "$bundle\lib" -Force
Copy-Item "D:\repo\rwkv_lightning_cuda\src\rwkv_vocab_v20230424.txt" -Destination "$bundle\rwkv_vocab_v20230424.txt" -Force
```

The expected runtime bundle layout is:

```text
rwkv_lighting_cuda
|-- lib
|   |-- brotlicommon.dll
|   |-- brotlidec.dll
|   |-- brotlienc.dll
|   |-- cares.dll
|   |-- cublas64_12.dll
|   |-- cublasLt64_12.dll
|   |-- cudart64_12.dll
|   |-- drogon.dll
|   |-- jsoncpp.dll
|   |-- libcrypto-3-x64.dll
|   |-- libssl-3-x64.dll
|   |-- msvcp140.dll
|   |-- sqlite3.dll
|   |-- trantor.dll
|   |-- vcruntime140.dll
|   |-- vcruntime140_1.dll
|   `-- zlib1.dll
|-- rwkv_launcher.exe
|-- rwkv_lighting_cuda.exe
`-- rwkv_vocab_v20230424.txt
```

If CMake copied Windows system DLLs into `lib`, they can be removed from the bundle. Keep the vcpkg DLLs, VC runtime DLLs, and CUDA DLLs listed above.

## Run with the launcher

The launcher is the recommended way to run the backend on Windows because it prepares the child process environment:

```powershell
cd D:\repo\rwkv_lightning_cuda\build_win10_sm86\bundle\rwkv_lighting_cuda
.\rwkv_launcher.exe
```

Open:

```text
http://127.0.0.1:8088
```

Use the UI to select the model, vocab, port, password, and WKV mode.

## Run the backend directly

When starting `rwkv_lighting_cuda.exe` directly, prepend the bundle `lib` directory to `PATH` first:

```powershell
cd D:\repo\rwkv_lightning_cuda\build_win10_sm86\bundle\rwkv_lighting_cuda

$env:PATH = "$PWD\lib;$env:PATH"

.\rwkv_lighting_cuda.exe `
  --model-path E:\rwkv7-g1g-2.9b-20260526-ctx8192.pth `
  --vocab-path .\rwkv_vocab_v20230424.txt `
  --host 127.0.0.1 `
  --port 8000
```

Optional arguments:

```powershell
--host 127.0.0.1
--password your-password
--wkv32
```

The backend binds to `127.0.0.1` by default. Use `--host 0.0.0.0` only when
you intentionally want to listen on all IPv4 interfaces.

## Verify the server

After the backend prints the endpoint list, verify the OpenAI-compatible models endpoint:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/v1/models
```

A successful response looks like:

```json
{"data":[{"id":"rwkv7-g1g-2.9b-20260526-ctx8192","object":"model","owned_by":"rwkv_lighting_cuda"}],"object":"list"}
```

Runtime files such as `rwkv_sessions.db` and `uploads` are generated in the working directory and are intentionally ignored by git.
