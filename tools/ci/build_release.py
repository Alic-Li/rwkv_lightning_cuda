#!/usr/bin/env python3
"""Build and package CUDA releases on GitHub's Linux/Windows x64 runners."""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
WINDOWS = platform.system() == "Windows"
SUFFIX = ".exe" if WINDOWS else ""
BUILD = ROOT / "build"
VCPKG = ROOT / "third_party/vcpkg"
TRIPLET = "x64-windows" if WINDOWS else "x64-linux"
ARCHS = os.environ.get("CUDA_ARCHITECTURES", "75;80;86;87;89;90;100;120")
CPU_TESTS = (
    "tokenizer", "pth_archive", "quantized_archive", "prefill_admission",
    "state_tuning_api",
)
BINARIES = ("rwkv_lighting_cuda", "rwkv_quantize", "rwkv_state_tune")


def run(*args, cwd=ROOT, env=None):
    command = [str(arg) for arg in args]
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main():
    if platform.system() not in ("Windows", "Linux"):
        raise SystemExit("CUDA release builds require Linux or Windows")
    if platform.machine().lower() not in ("x86_64", "amd64"):
        raise SystemExit("These release packages target x86-64")
    cuda = Path(os.environ["CUDA_PATH"])
    version = os.environ.get("GITHUB_REF_NAME", "local")
    version = re.sub(r"[^A-Za-z0-9._-]", "-", version)
    cuda_tag = "cuda" + ".".join(os.environ.get("CUDA_VERSION", "12.9.0").split(".")[:2])
    name = f"rwkv-lightning-{version}-{'windows' if WINDOWS else 'linux'}-x64-{cuda_tag}"
    bundle = BUILD / "release" / name
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    (ROOT / ".vcpkg-cache").mkdir(exist_ok=True)

    if WINDOWS:
        run("cmd", "/c", VCPKG / "bootstrap-vcpkg.bat", "-disableMetrics")
    else:
        run("bash", VCPKG / "bootstrap-vcpkg.sh", "-disableMetrics")
    configure = [
        "cmake", "-S", ROOT, "-B", BUILD, "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CUDA_ARCHITECTURES={ARCHS}",
        f"-DCMAKE_TOOLCHAIN_FILE={VCPKG / 'scripts/buildsystems/vcpkg.cmake'}",
        f"-DVCPKG_MANIFEST_DIR={ROOT / 'tools/ci'}",
        f"-DVCPKG_TARGET_TRIPLET={TRIPLET}",
        "-DRWKV7_FAST_BUILD_TESTS=ON", "-DRWKV7_STATE_TUNING=ON",
    ]
    if WINDOWS:
        configure += ["-DCMAKE_CXX_FLAGS=/Zc:preprocessor /EHsc /utf-8",
                      "-DCMAKE_CUDA_FLAGS=-Xcompiler=/Zc:preprocessor"]
    run(*configure)
    # Build every target, including GPU tests, to catch compile/link regressions.
    # Limit parallelism: multiple nvcc processes can exhaust hosted-runner memory.
    run("cmake", "--build", BUILD, "--config", "Release", "--parallel", "2")
    dependency_dirs = [BUILD / "vcpkg_installed" / TRIPLET / "bin", cuda / "bin"]
    # CUDA 13 installs Windows DLLs into bin/x64 instead of bin.
    if WINDOWS and (cuda / "bin" / "x64").is_dir():
        dependency_dirs.append(cuda / "bin" / "x64")
    test_env = os.environ.copy()
    if WINDOWS:
        test_env["PATH"] = os.pathsep.join(map(str, dependency_dirs)) + os.pathsep + test_env["PATH"]
    expression = "^rwkv_(" + "|".join(CPU_TESTS) + ")_test$"
    run("ctest", "--test-dir", BUILD, "-C", "Release", "--output-on-failure",
        "--no-tests=error", "-R", expression, env=test_env)

    # Put Windows DLLs beside all executables so direct invocation also works.
    lib_dir = bundle if WINDOWS else bundle / "lib"
    for binary in BINARIES:
        run("cmake", f"-DINPUT_FILE={BUILD / (binary + SUFFIX)}",
            f"-DOUTPUT_DIR={bundle}", f"-DLIB_OUTPUT_DIR={lib_dir}",
            "-DCOPY_BINARY=ON", "-DSTRICT_RUNTIME_DEPS=ON",
            "-DDEPENDENCY_DIRS=" + ";".join(map(str, dependency_dirs)),
            "-P", ROOT / "cmake/packaging/CopyRuntimeDependencies.cmake", env=test_env)
    if WINDOWS:
        # GET_RUNTIME_DEPENDENCIES excludes System32; ship the redistributable CRT.
        redist = Path(os.environ["VCToolsRedistDir"]) / "x64"
        crt_files = list(redist.glob("Microsoft.VC*.CRT/*.dll"))
        if not crt_files:
            raise RuntimeError(f"MSVC runtime not found under {redist}")
        for dll in crt_files:
            shutil.copy2(dll, bundle / dll.name)
        # copy_runtime_deps resolves the DLLs the executables actually import;
        # these globs are a safety net covering the CUDA 12 (bin) and CUDA 13
        # (bin/x64) toolkit layouts.
        for sub in dependency_dirs[1:]:
            for pattern in ("cudart*.dll", "cublas*.dll"):
                for dll in sorted(sub.glob(pattern)):
                    shutil.copy2(dll, bundle / dll.name)
    run("go", "build", "-trimpath", "-ldflags=-s -w", "-o",
        bundle / ("rwkv_launcher" + SUFFIX), "main.go",
        cwd=ROOT / "RWKV_Lightning_CUDA_Launcher")
    run("go", "build", "-trimpath", "-ldflags=-s -w", "-o",
        bundle / ("rwkv_router" + SUFFIX), ".",
        cwd=ROOT / "RWKV_Lightning_CUDA_router")
    shutil.copy2(ROOT / "assets/rwkv_vocab_v20230424.txt", bundle)
    shutil.copy2(ROOT / "RWKV_Lightning_CUDA_router/config.example.toml",
                 bundle / "router.config.example.toml")
    shutil.copy2(ROOT / "docs/releasing.md", bundle / "RELEASING.md")
    shutil.copy2(ROOT / "README.md", bundle)
    (bundle / "BUILD-INFO.json").write_text(json.dumps({
        "commit": os.environ.get("GITHUB_SHA", "local"),
        "ref": os.environ.get("GITHUB_REF", "local"),
        "cuda": os.environ.get("CUDA_VERSION", "12.9.0"),
        "architectures": ARCHS.split(";"),
        "runner": platform.platform(),
        "cpu_tests": list(CPU_TESTS),
        "gpu_tests_run": False,
    }, indent=2) + "\n", encoding="utf-8")
    # Check relocated CLIs with a clean library path, without loading a model.
    smoke_env = os.environ.copy()
    smoke_env.pop("LD_LIBRARY_PATH", None)
    if WINDOWS:
        smoke_env["PATH"] = str(Path(os.environ["SystemRoot"]) / "System32")
    for binary in BINARIES:
        run(bundle / (binary + SUFFIX), "--help", cwd=bundle, env=smoke_env)

    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    archive = Path(shutil.make_archive(str(dist / name), "zip" if WINDOWS else "gztar",
                                       root_dir=bundle.parent, base_dir=name))
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    archive.with_name(archive.name + ".sha256").write_text(
        f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"Package: {archive}")


if __name__ == "__main__":
    main()
