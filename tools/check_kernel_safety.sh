#!/usr/bin/env bash
# Run after a CUDA build, preferably configured with -DCMAKE_CUDA_FLAGS=-lineinfo.
set -euo pipefail
build_dir=${1:-build}
log_dir=${2:-"$build_dir/kernel-sanitizer"}
mkdir -p "$log_dir"
for tool in memcheck racecheck synccheck initcheck; do
  for target in rwkv_kernel_safety_test rwkv_cuda_non4096_kernels_test \
                rwkv_state_tuning_cuda_test rwkv_w4a16_kernels_test rwkv_w8a16_kernels_test; do
    if [[ "$target" == rwkv_state_tuning_cuda_test && ! -x "$build_dir/test/$target" ]]; then
      echo "SKIP: state tuning was not built"
      continue
    fi
    args=()
    if [[ "$target" == rwkv_w8a16_kernels_test ]]; then args+=(--quick); fi
    echo "$tool: $target"
    # Explicit capacity avoids synccheck's automatic barrier-count overflow
    # with CUDA 13.3 cuBLAS kernels on Blackwell.
    compute-sanitizer --tool "$tool" --num-cuda-barriers 256 --error-exitcode 99 \
      "$build_dir/test/$target" "${args[@]}" > "$log_dir/$target-$tool.log" 2>&1
    tail -n 2 "$log_dir/$target-$tool.log"
  done
done
