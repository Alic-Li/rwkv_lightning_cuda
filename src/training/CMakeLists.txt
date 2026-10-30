# BF16 training is independent of the FP16/quantized inference backend.
# Absolute paths keep CUDA/HIP source properties in this target's directory scope.
set(RWKV_TRAINING_SOURCE_ROOT "${PROJECT_SOURCE_DIR}/src")
set(RWKV_TRAINING_GPU_SOURCES
  ${RWKV_TRAINING_SOURCE_ROOT}/backend/bf16/ops.cu
  ${RWKV_TRAINING_SOURCE_ROOT}/backend/bf16/backward_kernels.cu
  ${RWKV_TRAINING_SOURCE_ROOT}/backend/bf16/training_kernels.cu
  ${RWKV_TRAINING_SOURCE_ROOT}/model/bf16/block_forward.cu
  ${RWKV_TRAINING_SOURCE_ROOT}/model/bf16/block_backward.cu
  ${RWKV_TRAINING_SOURCE_ROOT}/model/bf16/model_forward.cu
  ${RWKV_TRAINING_SOURCE_ROOT}/miss/training_kernels.cu
  ${RWKV_TRAINING_SOURCE_ROOT}/nora/training_kernels.cu
)
if(RWKV_GPU_BACKEND STREQUAL "HIP")
  set_source_files_properties(${RWKV_TRAINING_GPU_SOURCES} PROPERTIES LANGUAGE HIP)
  list(APPEND RWKV_TRAINING_GPU_SOURCES ${PROJECT_SOURCE_DIR}/hip/rwkv7_statepassing_clampw.hip)
else()
  list(APPEND RWKV_TRAINING_GPU_SOURCES ${PROJECT_SOURCE_DIR}/cuda/rwkv7_statepassing_clampw.cu)
endif()

add_library(rwkv_bf16_training STATIC
  ${RWKV_TRAINING_GPU_SOURCES}
  executor.cpp
  adapter_trainer.cpp
  adapter_checkpoint.cpp
  ${RWKV_TRAINING_SOURCE_ROOT}/model/bf16/model_backward.cpp
  ${RWKV_TRAINING_SOURCE_ROOT}/io/bf16_model_loader.cpp
  ${RWKV_TRAINING_SOURCE_ROOT}/io/bf16_state.cpp
  ${RWKV_TRAINING_SOURCE_ROOT}/io/bf16_adapter_writer.cpp
  ${RWKV_TRAINING_SOURCE_ROOT}/io/pth_archive.cpp
  ${RWKV_TRAINING_SOURCE_ROOT}/io/pth_tensor.cpp
  ${RWKV_TRAINING_SOURCE_ROOT}/io/pth_writer.cpp
  ${RWKV_TRAINING_SOURCE_ROOT}/io/fingerprint.cpp
  ${RWKV_TRAINING_SOURCE_ROOT}/inference/rwkv_tokenizer.cpp
  ${RWKV_TRAINING_SOURCE_ROOT}/utils/tokenizer.cpp
  ${PROJECT_SOURCE_DIR}/cuda/rwkv7_statepassing_clampw.cpp
)
add_library(rwkv::training ALIAS rwkv_bf16_training)
target_include_directories(rwkv_bf16_training
  PUBLIC ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${RWKV_TRAINING_SOURCE_ROOT} ${RWKV_TRAINING_SOURCE_ROOT}/utils
)
target_link_libraries(rwkv_bf16_training PUBLIC ${RWKV_JSON_TARGET} OpenSSL::Crypto)
if(RWKV_GPU_BACKEND STREQUAL "HIP")
  target_compile_definitions(rwkv_bf16_training PUBLIC RWKV_USE_HIP)
  target_link_libraries(rwkv_bf16_training PUBLIC hip::host roc::hipblas)
else()
  target_link_libraries(rwkv_bf16_training PUBLIC CUDA::cublas CUDA::cublasLt CUDA::cudart)
  set_property(TARGET rwkv_bf16_training PROPERTY CUDA_RUNTIME_LIBRARY Shared)
endif()
rwkv_optimize_backend_target(rwkv_bf16_training)
