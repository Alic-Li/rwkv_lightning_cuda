cmake_minimum_required(VERSION 3.24)
project(rwkv7_fast_v4 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(Drogon CONFIG QUIET)
if(NOT Drogon_FOUND)
  find_package(Drogon QUIET)
endif()
option(RWKV7_FAST_V4_STATIC_LINK "Link CUDA and C++ runtimes statically when possible" ON)
option(RWKV7_FAST_V4_GC_SECTIONS "Enable section garbage collection to reduce binary size" ON)
option(RWKV7_FAST_V4_AUTO_STRIP "Strip the release binary after linking when possible" ON)
option(RWKV7_FAST_BUILD_TESTS "Build unit tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(RWKV7_FAST_V4_STATIC_LINK)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
else()
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
endif()

set(RWKV7_FAST_V4_SOURCES
  src/rwkv7_fast_v4_org.cu
  src/pth_archive.cpp
  src/pth_tensor.cpp
  cuda/rwkv7_v3a_ops.cu
  cuda/rwkv7_fast_ops_fp16.cu
  cuda/rwkv7_wkv_fp16_v2.cu
  cuda/rwkv7_wkv_fp32_v2.cu
)

set(RWKV7_FAST_SERVER_CORE_SOURCES
  src/rwkv7_fast_v4.cu
  src/pth_archive.cpp
  src/pth_tensor.cpp
  src/rwkv_tokenizer.cpp
  src/utils/tokenizer.cpp
  src/rwkv_sampler.cpp
  src/rwkv_inference_engine.cpp
  src/rwkv_state_cache.cpp
  src/utils/sampling.cu
  cuda/rwkv7_v3a_ops.cu
  cuda/rwkv7_fast_ops_fp16.cu
  cuda/rwkv7_wkv_fp16_v2.cu
  cuda/rwkv7_wkv_fp32_v2.cu
)

add_library(rwkv7_fast_backend_core STATIC
  ${RWKV7_FAST_SERVER_CORE_SOURCES}
)

add_executable(benchmark
  ${RWKV7_FAST_V4_SOURCES}
)

target_include_directories(rwkv7_fast_backend_core PUBLIC cuda include src)
target_include_directories(benchmark PRIVATE cuda include src)
target_compile_definitions(rwkv7_fast_backend_core PRIVATE _IO_FP16_ RWKV_SAMPLING_NO_TORCH)
target_compile_definitions(benchmark PRIVATE _IO_FP16_)
if(RWKV7_FAST_V4_STATIC_LINK)
  target_link_libraries(rwkv7_fast_backend_core PUBLIC
    CUDA::cudart_static
    CUDA::cublas_static
    CUDA::cublasLt_static
    SQLite3::SQLite3
  )
  target_link_libraries(benchmark PRIVATE
    CUDA::cudart_static
    CUDA::cublas_static
    CUDA::cublasLt_static
  )
  set_property(TARGET rwkv7_fast_backend_core PROPERTY CUDA_RUNTIME_LIBRARY Static)
  set_property(TARGET benchmark PROPERTY CUDA_RUNTIME_LIBRARY Static)
else()
  target_link_libraries(rwkv7_fast_backend_core PUBLIC CUDA::cublas CUDA::cublasLt SQLite3::SQLite3)
  target_link_libraries(benchmark PRIVATE CUDA::cublas CUDA::cublasLt)
  set_property(TARGET rwkv7_fast_backend_core PROPERTY CUDA_RUNTIME_LIBRARY Shared)
  set_property(TARGET benchmark PROPERTY CUDA_RUNTIME_LIBRARY Shared)
endif()

if(RWKV7_FAST_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

if(MSVC)
  target_compile_definitions(rwkv7_fast_backend_core PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
  target_compile_definitions(benchmark PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
  target_compile_options(rwkv7_fast_backend_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/O2 /utf-8 /EHsc>
  )
  target_compile_options(benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/O2 /utf-8 /EHsc>
  )
else()
  target_compile_options(rwkv7_fast_backend_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
  )
  target_compile_options(benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
  )
  if(RWKV7_FAST_V4_GC_SECTIONS)
    target_compile_options(rwkv7_fast_backend_core PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections -fdata-sections>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffunction-sections,-fdata-sections>
    )
    target_compile_options(benchmark PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections -fdata-sections>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffunction-sections,-fdata-sections>
    )
    target_link_options(rwkv7_fast_backend_core PRIVATE -Wl,--gc-sections)
    target_link_options(benchmark PRIVATE -Wl,--gc-sections)
  endif()
  if(RWKV7_FAST_V4_STATIC_LINK)
    target_link_options(rwkv7_fast_backend_core PRIVATE -static-libstdc++ -static-libgcc)
    target_link_options(benchmark PRIVATE -static-libstdc++ -static-libgcc)
  endif()
endif()

target_compile_options(rwkv7_fast_backend_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --extra-device-vectorization --ptxas-options=-O3>
)
target_compile_options(benchmark PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --extra-device-vectorization --ptxas-options=-O3>
)

if(Drogon_FOUND)
  add_executable(rwkv_lighting_cuda
    src/rwkv_fast_server.cpp
    src/rwkv_api_service.cpp
  )
  target_link_libraries(rwkv_lighting_cuda PRIVATE rwkv7_fast_backend_core Drogon::Drogon)
  target_include_directories(rwkv_lighting_cuda PRIVATE cuda include src)
  if(MSVC)
    target_compile_options(rwkv_lighting_cuda PRIVATE /O2 /utf-8 /EHsc)
  else()
    target_compile_options(rwkv_lighting_cuda PRIVATE -O3)
  endif()
else()
  message(WARNING "Drogon not found. 'rwkv_lighting_cuda' target will not be built in this environment.")
endif()

if(RWKV7_FAST_V4_AUTO_STRIP AND CMAKE_STRIP AND NOT MSVC)
  add_custom_command(TARGET benchmark POST_BUILD
    COMMAND "$<$<CONFIG:Release>:${CMAKE_STRIP}>"
            "$<$<CONFIG:Release>:--strip-unneeded>"
            "$<$<CONFIG:Release>:$<TARGET_FILE:benchmark>>"
    VERBATIM
    COMMAND_EXPAND_LISTS
  )
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
