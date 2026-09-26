# Windows 构建与运行指南

[English](windows-build-run.md) | 简体中文

本文记录在 Windows 10、RTX 3080 上验证过的 Windows 构建流程。

## 已验证环境

- 操作系统：Windows 10 Pro 22H2，内部版本 19045
- GPU：NVIDIA GeForce RTX 3080，计算能力 8.6
- NVIDIA 驱动：576.57
- CUDA Toolkit：12.9
- Visual Studio：Visual Studio 2022，包含 MSVC x64 工具链
- CMake：4.0.3 或更新版本
- Go：1.24.4 或更新版本，仅构建 `rwkv_launcher.exe` 时需要
- vcpkg：版本需足够新，能够安装 `drogon:x64-windows`

本次构建使用 GPU 架构 `86`。其他 NVIDIA GPU 应将 `CMAKE_CUDA_ARCHITECTURES`
改为对应架构，或使用多架构列表生成适用范围更广的发布包。

## 构建依赖

构建前安装以下工具：

- Visual Studio 2022，包含“使用 C++ 的桌面开发”（`Desktop development with C++`）工作负载
- 通过 Visual Studio 安装器安装 Windows 10 或 Windows 11 SDK
- NVIDIA CUDA Toolkit 12.9
- CMake
- Git
- vcpkg
- Go、Bun（需要构建 Web 启动器时）

通过 vcpkg 安装 C++ 依赖：

```powershell
C:\vcpkg\vcpkg.exe install sqlite3:x64-windows drogon:x64-windows
```

`drogon:x64-windows` 会安装服务端使用的运行时依赖，包括 Trantor、OpenSSL、zlib、
c-ares、Brotli 和 JsonCpp。

## 构建 CUDA 后端

在仓库根目录执行：

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

要生成适用范围更广的二进制包，可将 `86` 替换为如下列表：

```powershell
-DCMAKE_CUDA_ARCHITECTURES="75;80;86;87;89;90"
```

后端可执行文件生成在：

```text
build_win10_sm86\bundle\rwkv_lighting_cuda\rwkv_lighting_cuda.exe
```

## 构建 Web 启动器

从仓库根目录开始执行：

```powershell
cd D:\repo\rwkv_lightning_cuda\RWKV_Lightning_CUDA_Launcher

bun install --frozen-lockfile
bun run build

$env:CGO_ENABLED = "0"

go build -trimpath -ldflags="-s -w" `
  -o ..\build_win10_sm86\bundle\rwkv_lighting_cuda\rwkv_launcher.exe `
  .
```

启动器在 `http://127.0.0.1:10721` 提供 HTTP 控制页面。
在 Windows 上，它将包内的 `lib` 目录加到后端子进程 `PATH` 的开头。

## 补齐运行时安装包

CMake 的打包目标将后端可执行文件、词表和检测到的运行时 DLL 复制到：

```text
build_win10_sm86\bundle\rwkv_lighting_cuda
```

如果某台机器未能自动找到 CUDA 运行时 DLL，请手动复制到安装包中：

```powershell
$bundle = "D:\repo\rwkv_lightning_cuda\build_win10_sm86\bundle\rwkv_lighting_cuda"
$cudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"

Copy-Item "$cudaBin\cudart64_12.dll" -Destination "$bundle\lib" -Force
Copy-Item "$cudaBin\cublas64_12.dll" -Destination "$bundle\lib" -Force
Copy-Item "$cudaBin\cublasLt64_12.dll" -Destination "$bundle\lib" -Force
```

预期的运行时目录结构：

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

如果 CMake 将 Windows 系统 DLL 复制进 `lib`，可以将它们从包中移除。
保留上面列出的 vcpkg DLL、VC 运行时 DLL 和 CUDA DLL。

## 通过启动器运行

Windows 下建议使用启动器运行后端，因为它会准备子进程环境：

```powershell
cd D:\repo\rwkv_lightning_cuda\build_win10_sm86\bundle\rwkv_lighting_cuda
.\rwkv_launcher.exe
```

打开：

```text
http://127.0.0.1:10721
```

在界面中选择模型、词表、端口、密码和 WKV 模式。

## 直接运行后端

直接启动 `rwkv_lighting_cuda.exe` 时，先将包内 `lib` 目录加入 `PATH` 开头：

```powershell
cd D:\repo\rwkv_lightning_cuda\build_win10_sm86\bundle\rwkv_lighting_cuda

$env:PATH = "$PWD\lib;$env:PATH"

.\rwkv_lighting_cuda.exe `
  --model-path E:\rwkv7-g1g-2.9b-20260526-ctx8192.pth `
  --vocab-path .\rwkv_vocab_v20230424.txt `
  --host 127.0.0.1 `
  --port 8000 `
  --chunk-size 128
```

可选参数：

```powershell
--host 127.0.0.1
--chunk-size 128
--password your-password
--wkv32
```

后端默认绑定 `127.0.0.1`。只有确实需要监听全部 IPv4 网络接口时才使用 `--host 0.0.0.0`。

## 验证服务

后端打印端点列表后，检查 OpenAI 兼容的模型列表接口：

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/v1/models
```

成功响应示例：

```json
{"data":[{"id":"rwkv7-g1g-2.9b-20260526-ctx8192","object":"model","owned_by":"rwkv_lighting_cuda"}],"object":"list"}
```

`rwkv_sessions.db` 和 `uploads` 等运行时文件生成在当前工作目录，已明确配置为由 git 忽略。
