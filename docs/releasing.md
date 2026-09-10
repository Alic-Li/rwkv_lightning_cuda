# CI and releases

The `CI and Release` GitHub Actions workflow builds CUDA packages for Linux x86-64
(Ubuntu 22.04) and Windows x64 (Windows Server 2022 / MSVC). It uses CUDA 12.9,
pinned vcpkg dependencies, and the Go version in the router's `go.mod`.

## Publish a version

1. Merge the release changes into `main` and wait for CI to pass.
2. Create and push a version tag:

   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0"
   git push origin v0.1.0
   ```

3. Wait for both platform builds. The workflow creates a **draft** GitHub Release
   with generated release notes, a Linux `.tar.gz`, a Windows `.zip`, and a
   `.sha256` file for each archive.
4. Test the packages on a CUDA machine, review the notes, and click **Publish release**.
   For an RC/beta, also select **Set as a pre-release** in the release editor.

No personal access token is needed: only the release job receives
`contents: write` through `GITHUB_TOKEN`. Re-running a tag workflow updates assets
on its existing draft; it fails rather than replacing a published release.
Use a new version tag for changes to an already published version.

## Build without publishing

PRs and pushes to `main` run CI automatically. `Actions → CI and Release → Run
workflow` also builds downloadable artifacts. A branch run does not create a
release; selecting a `v*` tag also runs the draft-release job.

PRs compile for SM 86 to keep review builds smaller. Main, tag and manual builds
compile for SM 75, 80, 86, 87, 89, 90, 100 and 120. Toolkit and architecture changes
should be made together in `.github/workflows/ci.yml` and `tools/ci/build_release.py`.
HIP/ROCm and ARM builds are not included in this workflow.

## Package contents and use

Each archive contains the server (`rwkv_lighting_cuda`, preserving its existing
spelling), launcher, quantizer, state-tuning CLI, router, example router config,
vocabulary, runtime libraries and `BUILD-INFO.json`. Models are not included.

Extract the whole directory, change into it, and run `rwkv_launcher` (Windows:
`rwkv_launcher.exe`). The launcher uses the included vocabulary by default.
Windows DLLs are beside the executables; Linux libraries are in `lib/` and resolved
using the executables' origin-relative RPATH. The router is a separate optional
process; start it with `rwkv_router --config router.config.example.toml` after
editing that example. Pass `--vocab ./rwkv_vocab_v20230424.txt` when using the
state-tuning CLI (its compiled default points to the build machine's source tree).

The target machine needs an NVIDIA driver compatible with CUDA 12.9 and its GPU.
Linux packages require glibc 2.35 or newer. CUDA development tools are not required
on the target machine. Keep the bundled CUDA and third-party runtime libraries
with the executables.

Verify an archive on Linux:

```bash
sha256sum --check rwkv-lightning-v0.1.0-linux-x64-cuda12.9.tar.gz.sha256
```

On Windows, compare `Get-FileHash <archive.zip> -Algorithm SHA256` with the `.sha256`
file. Checksums detect corruption; they are not signatures.

## What CI verifies

Both platforms run router tests and Go vet, compile all C++/CUDA targets (including
GPU tests), and run five CPU-only CTest suites: tokenizer, PTH archive, quantized
archive, prefill admission, and state-tuning API. Packaging fails on unresolved
runtime dependencies and runs each relocated C++ CLI's `--help` as a smoke check.

Hosted runners have no CUDA GPU. Kernel correctness, inference, GPU state tuning
and performance still require validation on a GPU before publishing; CI does not
claim these tests passed. `BUILD-INFO.json` records this limitation alongside the
source commit and compiled architectures. CTest logs are uploaded even on failure.

Implementation: `tools/ci/build_release.py` builds/tests/packages both platforms;
`tools/ci/vcpkg.json` lists dependencies; the workflow handles provisioning, caching,
artifacts and draft releases. You can run the Python script locally on Linux/Windows
with CUDA in `CUDA_PATH`, the pinned vcpkg checkout in `third_party/vcpkg`, Go,
CMake, Ninja and the platform compiler available (MSVC developer shell on Windows).
