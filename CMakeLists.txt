cmake_minimum_required(VERSION 3.24)

set(RWKV_GPU_BACKEND "CUDA" CACHE STRING "GPU backend (CUDA or HIP)")
set_property(CACHE RWKV_GPU_BACKEND PROPERTY STRINGS CUDA HIP)
string(TOUPPER "${RWKV_GPU_BACKEND}" RWKV_GPU_BACKEND)

if(RWKV_GPU_BACKEND STREQUAL "HIP")
  project(rwkv7_fast_v4 LANGUAGES CXX HIP)
elseif(RWKV_GPU_BACKEND STREQUAL "CUDA")
  project(rwkv7_fast_v4 LANGUAGES CXX CUDA)
else()
  message(FATAL_ERROR "RWKV_GPU_BACKEND must be CUDA or HIP")
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(RwkvOptions)
include(RwkvDependencies)
include(RwkvCompilerOptions)
include(RwkvPackaging)

add_subdirectory(src)
add_subdirectory(tools)

if(RWKV7_FAST_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

get_property(RWKV_BUNDLE_TARGETS GLOBAL PROPERTY RWKV_BUNDLE_TARGETS)
if(RWKV_BUNDLE_TARGETS)
  add_custom_target(bundle_all DEPENDS ${RWKV_BUNDLE_TARGETS})
endif()
