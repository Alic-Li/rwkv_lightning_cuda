# CI 与版本发布

[English](releasing.md) | 简体中文

`CI and Release` GitHub Actions 工作流通过 CUDA 12.9 / 13.2 矩阵，为 Linux x86-64
（Ubuntu 22.04）和 Windows x64（Windows Server 2022 / MSVC）构建 CUDA 包。
使用固定版本的 vcpkg 依赖，以及路由器 `go.mod` 中指定的 Go 版本。

## 发布版本

1. 修改仓库根目录 `VERSION` 文件中的语义化版本号：

   ```text
   1.5.0
   ```

2. 将版本修改与发布变更一起提交，推送到 `main`。
3. 等待全部平台构建完成。工作流创建 `v1.5.0` 标签和草稿 Release，上传 Linux `.tar.gz`、Windows `.zip` 及对应 `.sha256` 文件，然后附带自动生成的说明发布 Release。

无需个人访问令牌：只有发布任务通过 `GITHUB_TOKEN` 获得 `contents: write` 权限。
如果在创建标签或草稿后失败，重新运行会继续处理该尚未发布的版本。
如果推送中的 `VERSION` 已有正式 Release，则只执行普通 CI，不替换已有 Release。
发布下一版需要再次递增 `VERSION`。

## 仅构建而不发布

PR 和推送到 `main` 会自动运行 CI。在 `main` 上通过
`Actions → CI and Release → Run workflow` 手动运行，也会构建可下载产物，
且仅当该版本还没有 Release 时才发布。PR 和其他分支的手动运行永不发布。

PR 只编译 SM 86，以缩小评审构建。主分支和手动构建编译 SM 75、80、86、87、89、90、100、120。
修改工具链和架构时，应同时更新 `.github/workflows/ci.yml` 和 `tools/ci/build_release.py`。
此工作流不包含 HIP/ROCm 或 ARM 构建。

## 安装包内容与使用

每个压缩包包含一个 `rwkv_lighting_cuda/` 目录，暂存于 `build/bundle/rwkv_lighting_cuda/`：

```text
rwkv_lighting_cuda/
├── dist/
│   ├── assets/
│   └── index.html
├── lib/                  # Linux 运行时依赖
├── rwkv_launcher
├── rwkv_lighting_cuda
├── rwkv_quantize
├── rwkv_state_tune
└── rwkv_vocab_v20230424.txt
```

CI 根据 `bun.lock` 安装前端依赖，执行 lint/测试，并在编译启动器之前重新构建 `dist/`。
同一份前端既嵌入 Go 可执行文件，也复制到压缩包中。资源哈希和库名称随构建及 CUDA 版本
变化，不硬编码。Windows 可执行文件使用 `.exe`，运行时 DLL 放在其旁边。
包中不包含模型和可选路由器。

解压整个目录，进入后运行 `rwkv_launcher`（Windows 为 `rwkv_launcher.exe`）。
启动器默认使用附带词表。Linux 库通过相对于可执行文件位置的 RPATH 解析。
使用状态微调 CLI 时传入 `--vocab ./rwkv_vocab_v20230424.txt`，因为编译时的默认值
指向构建机器的源码目录。

目标机器需要与安装包 CUDA 版本（12.9 或 13.2）及 GPU 兼容的 NVIDIA 驱动。
Linux 包要求 glibc 2.35 或更新版本。目标机器无需 CUDA 开发工具，
但必须保留可执行文件附带的 CUDA 和第三方运行库。

在 Linux 上校验压缩包：

```bash
sha256sum --check rwkv-lightning-v0.1.0-linux-x64-cuda12.9.tar.gz.sha256
```

Windows 上将 `Get-FileHash <archive.zip> -Algorithm SHA256` 的结果与 `.sha256`
文件比较。校验和用于检测损坏，不是签名。

## CI 验证范围

两个平台均运行路由器测试和 Go vet，编译全部 C++/CUDA 目标（包括 GPU 测试），并运行
五组纯 CPU CTest：分词器、PTH 文件、量化文件、预填充准入和状态微调 API。
若运行时依赖无法解析，打包失败；还会执行移动后各 C++ CLI 的 `--help` 进行冒烟检查。

托管运行器没有 CUDA GPU。算子正确性、推理、GPU 状态微调和性能仍需在发布前使用 GPU
验证；CI 不声称这些测试已通过。即使失败，CTest 日志也会上传。

实现分工：`tools/ci/build_release.py` 负责两个平台的构建/测试/打包；
`tools/ci/vcpkg.json` 列出依赖；工作流负责环境准备、缓存、产物和发布。
本地 Linux/Windows 运行 Python 脚本时需准备 `CUDA_PATH` 指向的 CUDA、
`third_party/vcpkg` 中固定版本的 vcpkg、Go、Bun、CMake、Ninja 和平台编译器；
Windows 使用 MSVC 开发者 shell。先在启动器目录执行
`bun install --frozen-lockfile` 和 `bun run build`。

## 从 GitHub API 故障中恢复

发布器 `tools/ci/publish_release.py` 对临时 API 故障最多退避重试五次。
创建标签和草稿前，每次重试都会重新检查服务端状态，因此创建成功但响应丢失时仍可恢复。
发布说明单独生成；若该服务失败，则使用简短的回退说明。
产物通过 Release API 返回的 REST `upload_url` 上传，直接使用数值 release ID，
不通过 `gh release upload` 再次解析标签。同名产物（包括未完成上传）在重试前删除。
标签查询也会回退到分页 Release 列表以查找已有草稿。
上传后检查完成状态和文件大小，任一上传失败则保留草稿。已发布版本永不覆盖。

修改工作流或发布器后，将修复推送到 `main`，或在更新后的提交上运行工作流。
重新运行旧的失败任务使用的是原始工作流/代码，不会包含修复。
尚未发布的版本（如 `1.4.1`）可以保持当前 `VERSION`，无需删除标签/草稿。
持续的 GitHub 故障仍可能耗尽重试；待恢复后重新运行更新后的任务。
