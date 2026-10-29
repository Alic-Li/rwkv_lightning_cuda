set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

set(RWKV_CORE_HOST_SOURCES
  io/pth_archive.cpp
  io/pth_tensor.cpp
  io/pth_writer.cpp
  io/fingerprint.cpp
  miss/adapter.cpp
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
    ${PROJECT_SOURCE_DIR}/hip/rwkv_quantized.hip
    ${PROJECT_SOURCE_DIR}/hip/sampling.hip
    ${PROJECT_SOURCE_DIR}/hip/miss_kernels.hip
    ${PROJECT_SOURCE_DIR}/hip/nora_kernels.hip
    ${PROJECT_SOURCE_DIR}/hip/rwkv7_v3a_ops.hip
    ${PROJECT_SOURCE_DIR}/hip/rwkv7_fast_ops_fp16.hip
    ${PROJECT_SOURCE_DIR}/hip/rwkv7_wkv_fp16_v2.hip
    ${PROJECT_SOURCE_DIR}/hip/rwkv7_wkv_fp32_v2.hip
  )
else()
  set(RWKV_CORE_GPU_SOURCES
    backend/rwkv7_fast_v4.cu
    utils/sampling.cu
    miss/kernels.cu
    nora/kernels.cu
    ${PROJECT_SOURCE_DIR}/cuda/rwkv7_v3a_ops.cu
    ${PROJECT_SOURCE_DIR}/cuda/rwkv7_fast_ops_fp16.cu
    ${PROJECT_SOURCE_DIR}/cuda/rwkv7_wkv_fp16_v2.cu
    ${PROJECT_SOURCE_DIR}/cuda/rwkv7_wkv_fp32_v2.cu
    ${PROJECT_SOURCE_DIR}/quant/gemmv/rwkv_w8a16.cu
    ${PROJECT_SOURCE_DIR}/quant/gemmv/rwkv_w4a16.cu
  )
endif()

add_library(rwkv7_fast_backend_core STATIC
  ${RWKV_CORE_HOST_SOURCES}
  ${RWKV_CORE_GPU_SOURCES}
)
add_library(rwkv::backend ALIAS rwkv7_fast_backend_core)
target_link_libraries(rwkv7_fast_backend_core PUBLIC ${RWKV_JSON_TARGET} OpenSSL::Crypto)

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

if(RWKV7_STATE_TUNING)
  set(BF16_GPU_SOURCES bf16_training/ops.cu
    bf16_training/backward_kernels.cu bf16_training/block_forward.cu
    bf16_training/block_backward.cu bf16_training/model_forward.cu
    bf16_training/training_kernels.cu bf16_training/miss_kernels.cu
    bf16_training/nora_kernels.cu)
  if(RWKV_GPU_BACKEND STREQUAL "HIP")
    set_source_files_properties(${BF16_GPU_SOURCES} PROPERTIES LANGUAGE HIP)
    list(APPEND BF16_GPU_SOURCES ${PROJECT_SOURCE_DIR}/hip/rwkv7_statepassing_clampw.hip)
  else()
    list(APPEND BF16_GPU_SOURCES ${PROJECT_SOURCE_DIR}/cuda/rwkv7_statepassing_clampw.cu)
  endif()
  add_library(rwkv_bf16_training STATIC ${BF16_GPU_SOURCES}
    bf16_training/model.cpp bf16_training/model_backward.cpp bf16_training/executor.cpp
    bf16_training/trainer.cpp
    ${PROJECT_SOURCE_DIR}/cuda/rwkv7_statepassing_clampw.cpp bf16_training/format.cpp
    io/pth_archive.cpp io/pth_tensor.cpp io/pth_writer.cpp io/fingerprint.cpp
    inference/rwkv_tokenizer.cpp utils/tokenizer.cpp)
  target_include_directories(rwkv_bf16_training
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/utils
  )
  target_link_libraries(rwkv_bf16_training PUBLIC ${RWKV_JSON_TARGET} OpenSSL::Crypto)
  if(RWKV_GPU_BACKEND STREQUAL "HIP")
    target_compile_definitions(rwkv_bf16_training PUBLIC RWKV_USE_HIP)
    target_link_libraries(rwkv_bf16_training PUBLIC hip::host roc::hipblas)
  else()
    target_link_libraries(rwkv_bf16_training PUBLIC CUDA::cublas CUDA::cublasLt
      CUDA::cudart)
    set_property(TARGET rwkv_bf16_training PROPERTY CUDA_RUNTIME_LIBRARY Shared)
  endif()
  rwkv_optimize_backend_target(rwkv_bf16_training)
endif()

if(RWKV7_STATE_TUNING)
  add_executable(rwkv_miss_tune bf16_training/main.cpp state_tuning/dataset.cpp)
  target_include_directories(rwkv_miss_tune PRIVATE state_tuning)
  target_link_libraries(rwkv_miss_tune PRIVATE rwkv_bf16_training rwkv::version)
  target_compile_definitions(rwkv_miss_tune PRIVATE RWKV_MISS_TUNE
    RWKV_STATE_TUNE_DEFAULT_VOCAB="${PROJECT_SOURCE_DIR}/assets/rwkv_vocab_v20230424.txt")
  rwkv_optimize_target(rwkv_miss_tune)
  rwkv_configure_runtime_bundle(rwkv_miss_tune
    ASSETS ${PROJECT_SOURCE_DIR}/assets/rwkv_vocab_v20230424.txt
  )
  add_executable(rwkv_nora_tune bf16_training/main.cpp state_tuning/dataset.cpp)
  target_include_directories(rwkv_nora_tune PRIVATE state_tuning)
  target_link_libraries(rwkv_nora_tune PRIVATE rwkv_bf16_training rwkv::version)
  target_compile_definitions(rwkv_nora_tune PRIVATE RWKV_NORA_TUNE
    RWKV_STATE_TUNE_DEFAULT_VOCAB="${PROJECT_SOURCE_DIR}/assets/rwkv_vocab_v20230424.txt")
  rwkv_optimize_target(rwkv_nora_tune)
  rwkv_configure_runtime_bundle(rwkv_nora_tune
    ASSETS ${PROJECT_SOURCE_DIR}/assets/rwkv_vocab_v20230424.txt
  )
  add_executable(rwkv_state_tune
    bf16_training/main.cpp
    state_tuning/dataset.cpp
  )
  target_include_directories(rwkv_state_tune PRIVATE state_tuning)
  target_link_libraries(rwkv_state_tune PRIVATE rwkv_bf16_training rwkv::version)
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
    server/router/adapter_routes.cpp
    server/router/batch_routes.cpp
    server/router/chat_routes.cpp
    server/router/dynamic_routes.cpp
    server/router/generation_helpers.cpp
    server/router/http_helpers.cpp
    server/router/legacy_routes.cpp
    server/router/model_routes.cpp
    server/router/server_routes.cpp
    server/router/session_routes.cpp
    server/router/token_routes.cpp
    server/router/translation_routes.cpp
    server/router/uploaded_state_routes.cpp
    server/rwkv_model_router.cpp
  )
  target_link_libraries(rwkv_lighting_cuda PRIVATE rwkv::backend Drogon::Drogon rwkv::version)
  rwkv_optimize_target(rwkv_lighting_cuda)
  rwkv_strip_release_binary(rwkv_lighting_cuda)
  rwkv_configure_runtime_bundle(rwkv_lighting_cuda
    ASSETS ${PROJECT_SOURCE_DIR}/assets/rwkv_vocab_v20230424.txt
  )
else()
  message(WARNING "Drogon not found. 'rwkv_lighting_cuda' target will not be built.")
endif()

if(RWKV7_STATE_TUNING AND Drogon_FOUND)
  add_library(rwkv_online_rl STATIC online_rl/server.cpp)
  target_link_libraries(rwkv_online_rl PUBLIC rwkv_bf16_training rwkv::backend Drogon::Drogon)
  target_link_libraries(rwkv_online_rl PRIVATE rwkv::version)
  target_compile_definitions(rwkv_online_rl PRIVATE
    RWKV_STATE_TUNE_DEFAULT_VOCAB="${PROJECT_SOURCE_DIR}/assets/rwkv_vocab_v20230424.txt")
  rwkv_optimize_target(rwkv_online_rl)
  add_subdirectory(miss_tune_server)
  add_subdirectory(nora_tune_server)
  add_subdirectory(state_tune_server)
endif()
