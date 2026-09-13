#!/usr/bin/env bash
set -euo pipefail
mode=${1:?expected scalar or wmma}
action=${2:?expected build, test or bench}
case "$mode" in scalar|wmma) ;; *) exit 2;; esac
case "$action" in build|test|bench) ;; *) exit 2;; esac
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
output="build-hip/quant-validation/$mode"
if [[ "$action" == build ]]; then
  mkdir -p "$output"
  flags=(-O3 -std=c++17 --offload-arch=gfx1100 -DRWKV_USE_HIP
         "-DRWKV_TEST_REPO_ROOT=\"$repo\"" -Iinclude -Iquant/gemmv -Itest)
  if [[ "$mode" == scalar ]]; then flags+=(-DRWKV_QUANT_SCALAR_ONLY); fi
  for source in test/w4a16_kernels_test.cpp test/w8a16_hip_test.cpp tools/bench_quantized_hip.cpp; do
    name=$(basename "$source" .cpp)
    hipcc "${flags[@]}" "$source" hip/rwkv_quantized.hip -o "$output/$name"
  done
elif [[ "$action" == test ]]; then
  "$output/w4a16_kernels_test"
  "$output/w8a16_hip_test"
else
  "$output/bench_quantized_hip" | tee "$output/timings.csv"
fi
