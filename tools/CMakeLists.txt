set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

function(rwkv_add_tool target_name source_file)
  add_executable(${target_name} ${source_file})
  target_include_directories(${target_name} PRIVATE
    ${PROJECT_SOURCE_DIR}/cuda
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/quant/modeling
    ${PROJECT_SOURCE_DIR}/quant/gemmv
  )
  target_link_libraries(${target_name} PRIVATE rwkv::backend)
  rwkv_optimize_target(${target_name})
endfunction()

add_executable(rwkv_quantize
  ${PROJECT_SOURCE_DIR}/quant/modeling/rwkv_quantize.cpp
  ${PROJECT_SOURCE_DIR}/src/io/pth_archive.cpp
  ${PROJECT_SOURCE_DIR}/src/io/pth_tensor.cpp
  ${PROJECT_SOURCE_DIR}/quant/modeling/rwkv_quantized.cpp
)
target_include_directories(rwkv_quantize PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/quant/modeling
)
rwkv_optimize_target(rwkv_quantize)
rwkv_configure_runtime_bundle(rwkv_quantize)

if(RWKV_GPU_BACKEND STREQUAL "CUDA")
  rwkv_add_tool(rwkv_w4a16_bench bench_w4a16.cpp)
  rwkv_add_tool(rwkv_w8a16_bench bench_w8a16.cpp)
  rwkv_add_tool(rwkv_quantized_smoke quantized_smoke.cpp)
  rwkv_add_tool(rwkv_cmix_stats calibration/cmix_stats.cpp)
  rwkv_add_tool(rwkv_profile_model calibration/profile_model.cpp)
  rwkv_add_tool(rwkv_prefill_bench calibration/prefill_bench.cpp)
  rwkv_add_tool(rwkv_model_bench bench_model.cpp)
  rwkv_add_tool(rwkv_logit_mse_curve calibration/logit_mse_curve.cpp)

  add_executable(rwkv_calibration_bandwidth calibration/bandwidth.cu)
  target_link_libraries(rwkv_calibration_bandwidth PRIVATE CUDA::cudart)
  set_property(TARGET rwkv_calibration_bandwidth PROPERTY CUDA_RUNTIME_LIBRARY Shared)
  rwkv_optimize_target(rwkv_calibration_bandwidth)
endif()
