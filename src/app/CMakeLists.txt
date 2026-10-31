# The three training CLIs share argument parsing and the training loop.
foreach(mode IN ITEMS state miss nora)
  set(target_name rwkv_${mode}_tune)
  add_executable(${target_name}
    rwkv_tune.cpp
    ${PROJECT_SOURCE_DIR}/src/state_tuning/dataset.cpp
  )
  target_include_directories(${target_name} PRIVATE ${PROJECT_SOURCE_DIR}/src/state_tuning)
  target_link_libraries(${target_name} PRIVATE rwkv::training rwkv::version)
  target_compile_definitions(${target_name} PRIVATE
    RWKV_STATE_TUNE_DEFAULT_VOCAB="${PROJECT_SOURCE_DIR}/assets/rwkv_vocab_v20230424.txt"
  )
  if(mode STREQUAL "miss")
    target_compile_definitions(${target_name} PRIVATE RWKV_MISS_TUNE)
  elseif(mode STREQUAL "nora")
    target_compile_definitions(${target_name} PRIVATE RWKV_NORA_TUNE)
  endif()
  rwkv_optimize_target(${target_name})
  rwkv_configure_runtime_bundle(${target_name}
    ASSETS ${PROJECT_SOURCE_DIR}/assets/rwkv_vocab_v20230424.txt
  )
endforeach()

# NCCL is optional for the existing single-GPU/HIP build. An explicit ON fails
# configuration if the dependency is missing instead of silently omitting DP.
set(RWKV_NCCL_DP "AUTO" CACHE STRING "Build NCCL training CLIs: AUTO, ON, OFF")
set_property(CACHE RWKV_NCCL_DP PROPERTY STRINGS AUTO ON OFF)
if(NOT RWKV_NCCL_DP MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "RWKV_NCCL_DP must be AUTO, ON or OFF")
endif()
if(RWKV_GPU_BACKEND STREQUAL "CUDA" AND CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT RWKV_NCCL_DP STREQUAL "OFF")
  find_path(NCCL_INCLUDE_DIR nccl.h HINTS ${NCCL_ROOT} ENV NCCL_ROOT PATH_SUFFIXES include)
  find_library(NCCL_LIBRARY NAMES nccl libnccl.so.2 HINTS ${NCCL_ROOT} ENV NCCL_ROOT PATH_SUFFIXES lib lib64)
  if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
    add_library(rwkv_nccl_dp STATIC ../training/nccl_dp.cpp)
    target_include_directories(rwkv_nccl_dp PUBLIC ../training ${NCCL_INCLUDE_DIR})
    target_link_libraries(rwkv_nccl_dp PUBLIC rwkv::training ${NCCL_LIBRARY})
    foreach(mode IN ITEMS state miss nora)
      set(target_name rwkv_${mode}_tune_dp)
      add_executable(${target_name} rwkv_tune.cpp ${PROJECT_SOURCE_DIR}/src/state_tuning/dataset.cpp)
      target_include_directories(${target_name} PRIVATE ${PROJECT_SOURCE_DIR}/src/state_tuning)
      target_link_libraries(${target_name} PRIVATE rwkv_nccl_dp rwkv::version)
      target_compile_definitions(${target_name} PRIVATE RWKV_NCCL_DP
        RWKV_STATE_TUNE_DEFAULT_VOCAB="${PROJECT_SOURCE_DIR}/assets/rwkv_vocab_v20230424.txt")
      if(mode STREQUAL "miss")
        target_compile_definitions(${target_name} PRIVATE RWKV_MISS_TUNE)
      elseif(mode STREQUAL "nora")
        target_compile_definitions(${target_name} PRIVATE RWKV_NORA_TUNE)
      endif()
      rwkv_optimize_target(${target_name})
      rwkv_configure_runtime_bundle(${target_name} ASSETS ${PROJECT_SOURCE_DIR}/assets/rwkv_vocab_v20230424.txt)
    endforeach()
  elseif(RWKV_NCCL_DP STREQUAL "ON")
    message(FATAL_ERROR "NCCL headers/library required. Set NCCL_ROOT or NCCL_INCLUDE_DIR/NCCL_LIBRARY.")
  else()
    message(STATUS "NCCL not found; skipping *_tune_dp (set NCCL_ROOT to enable)")
  endif()
elseif(RWKV_NCCL_DP STREQUAL "ON")
  message(FATAL_ERROR "NCCL DP requires CUDA on Linux")
endif()
