# CI and releases

The `CI and Release` GitHub Actions workflow builds CUDA packages for Linux x86-64
(Ubuntu 22.04) and Windows x64 (Windows Server 2022 / MSVC) against a CUDA
12.9 / 13.2 matrix. It uses pinned vcpkg dependencies and the Go version in the
router's `go.mod`.

## Publish a version

1. Change the semantic version in the repository root `VERSION` file:

   ```text
   1.5.0
   ```

2. Commit the version change with the release changes and push it to `main`.
3. Wait for all platform builds. The workflow creates tag `v1.5.0`, creates a
   draft Release, uploads the Linux `.tar.gz`, Windows `.zip`, and matching
   `.sha256` files, then publishes the Release with generated notes.

No personal access token is needed: only the release job receives
`contents: write` through `GITHUB_TOKEN`. If a run fails after creating the tag
or draft, re-running it resumes that unpublished version. A push whose `VERSION`
already has a published Release runs normal CI and does not replace that Release.
Increment `VERSION` again to publish another release.

## Build without publishing

PRs and pushes to `main` run CI automatically. `Actions → CI and Release → Run
workflow` on `main` also builds downloadable artifacts and publishes the version
only if it does not already have a Release. Pull requests and manual runs from
other branches never publish.

PRs compile for SM 86 to keep review builds smaller. Main and manual builds
compile for SM 75, 80, 86, 87, 89, 90, 100 and 120. Toolkit and architecture changes
should be made together in `.github/workflows/ci.yml` and `tools/ci/build_release.py`.
HIP/ROCm and ARM builds are not included in this workflow.

## Package contents and use

Each archive contains a single `rwkv_lighting_cuda/` directory, staged at
`build/bundle/rwkv_lighting_cuda/`:

```text
rwkv_lighting_cuda/
├── dist/
│   ├── assets/
│   └── index.html
├── lib/                  # Linux runtime dependencies
├── rwkv_launcher
├── rwkv_lighting_cuda
├── rwkv_quantize
├── rwkv_state_tune
└── rwkv_vocab_v20230424.txt
```

CI installs frontend dependencies from `bun.lock`, runs lint/tests, and rebuilds
`dist/` before compiling the launcher. The same frontend is embedded in the Go
executable and copied into the archive. Asset hashes and library names vary with
the build and CUDA version; they are not hardcoded. Windows executables use `.exe`
and runtime DLLs sit beside them. Models and the optional router are not included.

Extract the whole directory, change into it, and run `rwkv_launcher` (Windows:
`rwkv_launcher.exe`). The launcher uses the included vocabulary by default.
Linux libraries are resolved using the executables' origin-relative RPATH.
Pass `--vocab ./rwkv_vocab_v20230424.txt` when using the state-tuning CLI
(its compiled default points to the build machine's source tree).

The target machine needs an NVIDIA driver compatible with the CUDA version of the
package (12.9 or 13.2) and its GPU. Linux packages require glibc 2.35 or newer.
CUDA development tools are not required
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
claim these tests passed. CTest logs are uploaded even on failure.

Implementation: `tools/ci/build_release.py` builds/tests/packages both platforms;
`tools/ci/vcpkg.json` lists dependencies; the workflow handles provisioning, caching,
artifacts and releases. You can run the Python script locally on Linux/Windows
with CUDA in `CUDA_PATH`, the pinned vcpkg checkout in `third_party/vcpkg`, Go,
Bun (run `bun install --frozen-lockfile` and `bun run build` in the launcher first),
CMake, Ninja and the platform compiler available (MSVC developer shell on Windows).

## Recover from GitHub API failures

The publisher (`tools/ci/publish_release.py`) retries transient API failures up to
five times with backoff. Tag and draft creation re-check server state before each
retry, so a successful create with a lost response can resume. Release notes are
generated separately; if that service fails, a short fallback description is used.
Assets are uploaded individually with retries, and the release remains a draft if
any upload fails. Already published releases are never overwritten.

After changing the workflow or publisher, push the fix to `main` (or run the
workflow on that updated commit). Re-running an old failed run uses its original
workflow/code, not this fix. An unpublished version such as `1.4.1` can keep its
current `VERSION`; no tag/draft deletion is required. A persistent GitHub outage
can still exhaust retries; retry the updated run once GitHub recovers.
