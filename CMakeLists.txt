cmake_minimum_required(VERSION 3.24)

set(RWKV_GPU_BACKEND "CUDA" CACHE STRING "GPU backend (CUDA or HIP)")
set_property(CACHE RWKV_GPU_BACKEND PROPERTY STRINGS CUDA HIP)
string(TOUPPER "${RWKV_GPU_BACKEND}" RWKV_GPU_BACKEND)
if(RWKV_GPU_BACKEND STREQUAL "HIP")
  project(rwkv7_fast_v4 LANGUAGES CXX HIP)
  find_package(hip REQUIRED CONFIG)
  find_package(hipblas REQUIRED CONFIG)
  find_package(hipblaslt REQUIRED CONFIG)
  find_package(hiprand REQUIRED CONFIG)
elseif(RWKV_GPU_BACKEND STREQUAL "CUDA")
  project(rwkv7_fast_v4 LANGUAGES CXX CUDA)
  find_package(CUDAToolkit REQUIRED)
else()
  message(FATAL_ERROR "RWKV_GPU_BACKEND must be CUDA or HIP")
endif()

find_package(SQLite3 REQUIRED)
find_package(Drogon CONFIG QUIET)
if(NOT Drogon_FOUND)
  find_package(Drogon QUIET)
endif()

# Normalize SQLite target name across CMake/vcpkg variants on Windows.
if(TARGET SQLite3::SQLite3)
  set(RWKV_SQLITE_TARGET SQLite3::SQLite3)
elseif(TARGET SQLite::SQLite3)
  set(RWKV_SQLITE_TARGET SQLite::SQLite3)
else()
  add_library(RWKV_SQLITE_FALLBACK INTERFACE)
  if(SQLite3_INCLUDE_DIRS)
    target_include_directories(RWKV_SQLITE_FALLBACK INTERFACE ${SQLite3_INCLUDE_DIRS})
  endif()
  if(SQLite3_LIBRARIES)
    target_link_libraries(RWKV_SQLITE_FALLBACK INTERFACE ${SQLite3_LIBRARIES})
  elseif(SQLite3_LIBRARY)
    target_link_libraries(RWKV_SQLITE_FALLBACK INTERFACE ${SQLite3_LIBRARY})
  else()
    message(FATAL_ERROR "SQLite3 was found, but no linkable target or library variable was provided.")
  endif()
  set(RWKV_SQLITE_TARGET RWKV_SQLITE_FALLBACK)
endif()

option(RWKV7_FAST_V4_GC_SECTIONS "Enable section garbage collection to reduce binary size" ON)
option(RWKV7_FAST_V4_AUTO_STRIP "Strip the release binary after linking when possible" ON)
option(RWKV7_FAST_BUILD_TESTS "Build unit tests" ON)

function(rwkv_configure_msvc_optimized_target target_name)
  target_compile_definitions(${target_name} PRIVATE
    $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
  )
  target_compile_options(${target_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8 /EHsc>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:Debug>>>:/O2>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-Xcompiler=/O2>
  )

  if(RWKV7_FAST_V4_MSVC_LTCG)
    target_compile_options(${target_name} PRIVATE
      $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:Debug>>>:SHELL:/GL>
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:SHELL:-Xcompiler=/GL>
    )
  endif()
endfunction()

function(rwkv_configure_runtime_bundle target_name)
  if(APPLE)
    set(_rwkv_origin "@loader_path")
  elseif(UNIX)
    set(_rwkv_origin "$ORIGIN")
  else()
    set(_rwkv_origin "")
  endif()

  if(_rwkv_origin)
    set_target_properties(${target_name} PROPERTIES
      BUILD_RPATH "${_rwkv_origin};${_rwkv_origin}/lib"
      INSTALL_RPATH "${_rwkv_origin};${_rwkv_origin}/lib"
      BUILD_RPATH_USE_ORIGIN ON
    )
    if(UNIX AND NOT APPLE)
      target_link_options(${target_name} PRIVATE -Wl,--disable-new-dtags)
    endif()
  endif()

  add_custom_target(bundle_${target_name}
    COMMAND ${CMAKE_COMMAND}
      -DINPUT_FILE=$<TARGET_FILE:${target_name}>
      -DOUTPUT_DIR=${PROJECT_BINARY_DIR}/bundle/${target_name}
      -DLIB_OUTPUT_DIR=${PROJECT_BINARY_DIR}/bundle/${target_name}/lib
      -DCOPY_BINARY=ON
      -P ${PROJECT_SOURCE_DIR}/cmake/copy_runtime_deps.cmake
    DEPENDS ${target_name}
    VERBATIM
  )
endfunction()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(RWKV_GPU_BACKEND STREQUAL "HIP")
  set(CMAKE_HIP_STANDARD 17)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
else()
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
endif()

set(RWKV7_FAST_HOST_SOURCES
  src/pth_archive.cpp
  src/pth_tensor.cpp
  quant/modeling/rwkv_quantized.cpp
  src/rwkv_tokenizer.cpp
  src/utils/tokenizer.cpp
  src/rwkv_sampler.cpp
  src/rwkv_inference_engine.cpp
  src/rwkv_prefill_admission.cpp
  src/rwkv_state_cache.cpp
)
if(RWKV_GPU_BACKEND STREQUAL "HIP")
  set(RWKV7_FAST_GPU_SOURCES
    hip/rwkv7_fast_v4.hip
    hip/sampling.hip
    hip/rwkv7_v3a_ops.hip
    hip/rwkv7_fast_ops_fp16.hip
    hip/rwkv7_wkv_fp16_v2.hip
    hip/rwkv7_wkv_fp32_v2.hip
  )
else()
  set(RWKV7_FAST_GPU_SOURCES
    src/rwkv7_fast_v4.cu
    src/utils/sampling.cu
    cuda/rwkv7_v3a_ops.cu
    cuda/rwkv7_fast_ops_fp16.cu
    cuda/rwkv7_wkv_fp16_v2.cu
    cuda/rwkv7_wkv_fp32_v2.cu
    quant/gemmv/rwkv_w8a16.cu
  )
endif()

add_library(rwkv7_fast_backend_core STATIC
  ${RWKV7_FAST_HOST_SOURCES}
  ${RWKV7_FAST_GPU_SOURCES}
)

add_executable(rwkv_quantize
  quant/modeling/rwkv_quantize.cpp
  src/pth_archive.cpp
  src/pth_tensor.cpp
  quant/modeling/rwkv_quantized.cpp
)
target_include_directories(rwkv_quantize PRIVATE include src quant/modeling)
target_compile_options(rwkv_quantize PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)

if(RWKV_GPU_BACKEND STREQUAL "CUDA")
  add_executable(rwkv_w8a16_bench tools/bench_w8a16.cpp)
  target_include_directories(rwkv_w8a16_bench PRIVATE include quant/gemmv)
  target_link_libraries(rwkv_w8a16_bench PRIVATE rwkv7_fast_backend_core CUDA::cublas CUDA::cublasLt)
  target_compile_options(rwkv_w8a16_bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
  add_executable(rwkv_quantized_smoke tools/quantized_smoke.cpp)
  target_include_directories(rwkv_quantized_smoke PRIVATE include cuda src quant/modeling quant/gemmv)
  target_link_libraries(rwkv_quantized_smoke PRIVATE rwkv7_fast_backend_core CUDA::cublas CUDA::cublasLt)
  target_compile_options(rwkv_quantized_smoke PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
  add_executable(rwkv_cmix_stats tools/calibration/cmix_stats.cpp)
  target_include_directories(rwkv_cmix_stats PRIVATE include cuda src quant/modeling quant/gemmv)
  target_link_libraries(rwkv_cmix_stats PRIVATE rwkv7_fast_backend_core CUDA::cublas CUDA::cublasLt)
  target_compile_options(rwkv_cmix_stats PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
  add_executable(rwkv_profile_model tools/calibration/profile_model.cpp)
  target_include_directories(rwkv_profile_model PRIVATE include cuda src quant/modeling quant/gemmv)
  target_link_libraries(rwkv_profile_model PRIVATE rwkv7_fast_backend_core CUDA::cublas CUDA::cublasLt)
  target_compile_options(rwkv_profile_model PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
  add_executable(rwkv_prefill_bench tools/calibration/prefill_bench.cpp)
  target_include_directories(rwkv_prefill_bench PRIVATE include cuda src quant/modeling quant/gemmv)
  target_link_libraries(rwkv_prefill_bench PRIVATE rwkv7_fast_backend_core CUDA::cublas CUDA::cublasLt)
  target_compile_options(rwkv_prefill_bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
  add_executable(rwkv_model_bench tools/bench_model.cpp)
  target_include_directories(rwkv_model_bench PRIVATE include cuda src quant/modeling quant/gemmv)
  target_link_libraries(rwkv_model_bench PRIVATE rwkv7_fast_backend_core CUDA::cublas CUDA::cublasLt)
  target_compile_options(rwkv_model_bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
  add_executable(rwkv_logit_mse_curve tools/calibration/logit_mse_curve.cpp)
  target_include_directories(rwkv_logit_mse_curve PRIVATE include cuda src quant/modeling quant/gemmv)
  target_link_libraries(rwkv_logit_mse_curve PRIVATE rwkv7_fast_backend_core CUDA::cublas CUDA::cublasLt)
  target_compile_options(rwkv_logit_mse_curve PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
  add_executable(rwkv_calibration_bandwidth tools/calibration/bandwidth.cu)
  target_link_libraries(rwkv_calibration_bandwidth PRIVATE CUDA::cudart)
  set_property(TARGET rwkv_calibration_bandwidth PROPERTY CUDA_RUNTIME_LIBRARY Shared)
endif()

if(RWKV_GPU_BACKEND STREQUAL "HIP")
  target_include_directories(rwkv7_fast_backend_core PUBLIC hip include src src/utils quant/modeling quant/gemmv)
  target_compile_definitions(rwkv7_fast_backend_core PUBLIC RWKV_USE_HIP)
  target_link_libraries(rwkv7_fast_backend_core PUBLIC hip::host roc::hipblas roc::hipblaslt hip::hiprand ${RWKV_SQLITE_TARGET})
else()
  target_include_directories(rwkv7_fast_backend_core PUBLIC cuda include src quant/modeling quant/gemmv)
  target_link_libraries(rwkv7_fast_backend_core PUBLIC CUDA::cublas CUDA::cublasLt ${RWKV_SQLITE_TARGET})
  set_property(TARGET rwkv7_fast_backend_core PROPERTY CUDA_RUNTIME_LIBRARY Shared)
endif()
target_compile_definitions(rwkv7_fast_backend_core PRIVATE _IO_FP16_ RWKV_SAMPLING_NO_TORCH)

if(RWKV7_FAST_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

if(MSVC)
  target_compile_definitions(rwkv7_fast_backend_core PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
  target_compile_options(rwkv7_fast_backend_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8 /EHsc>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:Debug>>>:/O2>
  )
  rwkv_configure_msvc_optimized_target(rwkv7_fast_backend_core)
else()
  target_compile_options(rwkv7_fast_backend_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
  )
  if(RWKV7_FAST_V4_GC_SECTIONS)
    target_compile_options(rwkv7_fast_backend_core PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections -fdata-sections>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffunction-sections,-fdata-sections>
    )
    target_link_options(rwkv7_fast_backend_core PRIVATE -Wl,--gc-sections)
  endif()
endif()

target_compile_options(rwkv7_fast_backend_core PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-O3 --use_fast_math --extra-device-vectorization --ptxas-options=-O3>
  $<$<AND:$<COMPILE_LANGUAGE:HIP>,$<NOT:$<CONFIG:Debug>>>:-O3>
)

if(Drogon_FOUND)
  add_executable(rwkv_lighting_cuda
    src/rwkv_fast_server.cpp
    src/rwkv_api_service.cpp
    src/rwkv_model_router.cpp
  )
  target_link_libraries(rwkv_lighting_cuda PRIVATE rwkv7_fast_backend_core Drogon::Drogon)
  if(RWKV_GPU_BACKEND STREQUAL "HIP")
    target_include_directories(rwkv_lighting_cuda PRIVATE hip include src)
  else()
    target_include_directories(rwkv_lighting_cuda PRIVATE cuda include src)
  endif()
  if(MSVC)
    target_compile_definitions(rwkv_lighting_cuda PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    target_compile_options(rwkv_lighting_cuda PRIVATE /O2 /utf-8 /EHsc)
  else()
    target_compile_options(rwkv_lighting_cuda PRIVATE -O3)
  endif()
  rwkv_configure_runtime_bundle(rwkv_lighting_cuda)
else()
  message(WARNING "Drogon not found. 'rwkv_lighting_cuda' target will not be built in this environment.")
endif()

if(RWKV7_FAST_V4_AUTO_STRIP AND CMAKE_STRIP AND NOT MSVC)
  if(TARGET rwkv_lighting_cuda)
    add_custom_command(TARGET rwkv_lighting_cuda POST_BUILD
      COMMAND "$<$<CONFIG:Release>:${CMAKE_STRIP}>"
              "$<$<CONFIG:Release>:--strip-unneeded>"
              "$<$<CONFIG:Release>:$<TARGET_FILE:rwkv_lighting_cuda>>"
      VERBATIM
      COMMAND_EXPAND_LISTS
    )
  endif()
endif()
