set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

set(RWKV_CORE_HOST_SOURCES
  io/pth_archive.cpp
  io/pth_tensor.cpp
  ${PROJECT_SOURCE_DIR}/quant/modeling/rwkv_quantized.cpp
  inference/rwkv_tokenizer.cpp
  utils/tokenizer.cpp
  inference/rwkv_sampler.cpp
  inference/rwkv_inference_engine.cpp
  server/rwkv_prefill_admission.cpp
  server/rwkv_state_cache.cpp
  server/rwkv_uploaded_state_store.cpp
)

if(RWKV_GPU_BACKEND STREQUAL "HIP")
  set(RWKV_CORE_GPU_SOURCES
    ${PROJECT_SOURCE_DIR}/hip/rwkv7_fast_v4.hip
    ${PROJECT_SOURCE_DIR}/hip/sampling.hip
    ${PROJECT_SOURCE_DIR}/hip/rwkv7_v3a_ops.hip
    ${PROJECT_SOURCE_DIR}/hip/rwkv7_fast_ops_fp16.hip
    ${PROJECT_SOURCE_DIR}/hip/rwkv7_wkv_fp16_v2.hip
    ${PROJECT_SOURCE_DIR}/hip/rwkv7_wkv_fp32_v2.hip
  )
else()
  set(RWKV_CORE_GPU_SOURCES
    backend/rwkv7_fast_v4.cu
    utils/sampling.cu
    ${PROJECT_SOURCE_DIR}/cuda/rwkv7_v3a_ops.cu
    ${PROJECT_SOURCE_DIR}/cuda/rwkv7_fast_ops_fp16.cu
    ${PROJECT_SOURCE_DIR}/cuda/rwkv7_wkv_fp16_v2.cu
    ${PROJECT_SOURCE_DIR}/cuda/rwkv7_wkv_fp32_v2.cu
    ${PROJECT_SOURCE_DIR}/quant/gemmv/rwkv_w8a16.cu
    ${PROJECT_SOURCE_DIR}/quant/gemmv/rwkv_w4a16.cu
  )
  if(RWKV7_STATE_TUNING)
    list(APPEND RWKV_CORE_GPU_SOURCES
      ${PROJECT_SOURCE_DIR}/cuda/rwkv7_statepassing_clampw.cpp
      ${PROJECT_SOURCE_DIR}/cuda/rwkv7_statepassing_clampw.cu
      state_tuning/backward_kernels.cu
      state_tuning/block_backward.cu
      state_tuning/block_forward.cu
      state_tuning/training_kernels.cu
      state_tuning/model_backward.cpp
      state_tuning/model_forward.cu
      state_tuning/state_checkpoint.cpp
    )
  endif()
endif()

add_library(rwkv7_fast_backend_core STATIC
  ${RWKV_CORE_HOST_SOURCES}
  ${RWKV_CORE_GPU_SOURCES}
)
add_library(rwkv::backend ALIAS rwkv7_fast_backend_core)

target_include_directories(rwkv7_fast_backend_core
  PUBLIC ${PROJECT_SOURCE_DIR}/include
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/utils
    ${PROJECT_SOURCE_DIR}/quant/modeling
    ${PROJECT_SOURCE_DIR}/quant/gemmv
)
target_compile_definitions(rwkv7_fast_backend_core PRIVATE
  _IO_FP16_
  RWKV_SAMPLING_NO_TORCH
)

if(RWKV_GPU_BACKEND STREQUAL "HIP")
  target_include_directories(rwkv7_fast_backend_core PRIVATE ${PROJECT_SOURCE_DIR}/hip)
  target_compile_definitions(rwkv7_fast_backend_core PUBLIC RWKV_USE_HIP)
  target_link_libraries(rwkv7_fast_backend_core PUBLIC
    hip::host roc::hipblas roc::hipblaslt hip::hiprand ${RWKV_SQLITE_TARGET}
  )
else()
  target_include_directories(rwkv7_fast_backend_core PRIVATE ${PROJECT_SOURCE_DIR}/cuda)
  target_link_libraries(rwkv7_fast_backend_core PUBLIC
    CUDA::cublas CUDA::cublasLt ${RWKV_SQLITE_TARGET}
  )
  set_property(TARGET rwkv7_fast_backend_core PROPERTY CUDA_RUNTIME_LIBRARY Shared)
endif()
rwkv_optimize_backend_target(rwkv7_fast_backend_core)

if(RWKV_GPU_BACKEND STREQUAL "CUDA" AND RWKV7_STATE_TUNING)
  add_executable(rwkv_state_tune
    state_tuning/rwkv_state_tune_main.cpp
    state_tuning/dataset.cpp
  )
  target_include_directories(rwkv_state_tune PRIVATE state_tuning)
  target_link_libraries(rwkv_state_tune PRIVATE rwkv::backend)
  target_compile_definitions(rwkv_state_tune PRIVATE
    RWKV_STATE_TUNE_DEFAULT_VOCAB="${PROJECT_SOURCE_DIR}/assets/rwkv_vocab_v20230424.txt"
  )
  rwkv_optimize_target(rwkv_state_tune)
  rwkv_configure_runtime_bundle(rwkv_state_tune
    ASSETS ${PROJECT_SOURCE_DIR}/assets/rwkv_vocab_v20230424.txt
  )
endif()

if(Drogon_FOUND)
  add_executable(rwkv_lighting_cuda
    app/rwkv_fast_server.cpp
    server/rwkv_api_service.cpp
    server/rwkv_model_router.cpp
  )
  target_link_libraries(rwkv_lighting_cuda PRIVATE rwkv::backend Drogon::Drogon)
  rwkv_optimize_target(rwkv_lighting_cuda)
  rwkv_strip_release_binary(rwkv_lighting_cuda)
  rwkv_configure_runtime_bundle(rwkv_lighting_cuda
    ASSETS ${PROJECT_SOURCE_DIR}/assets/rwkv_vocab_v20230424.txt
  )
else()
  message(WARNING "Drogon not found. 'rwkv_lighting_cuda' target will not be built.")
endif()
