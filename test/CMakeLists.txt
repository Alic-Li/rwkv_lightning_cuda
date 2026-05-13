set(RWKV_TEST_COMMON_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/test_common.hpp
)

function(rwkv_add_unit_test target_name source_file)
  add_executable(${target_name} ${source_file})
  target_link_libraries(${target_name} PRIVATE rwkv7_fast_backend_core)
  target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${target_name} PRIVATE RWKV_TEST_REPO_ROOT="${PROJECT_SOURCE_DIR}")
  if(MSVC)
    target_compile_options(${target_name} PRIVATE /O2 /utf-8 /EHsc)
  else()
    target_compile_options(${target_name} PRIVATE -O2)
  endif()
  add_test(NAME ${target_name} COMMAND ${target_name})
endfunction()

rwkv_add_unit_test(rwkv_tokenizer_test ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer_test.cpp)
rwkv_add_unit_test(rwkv_sampler_test ${CMAKE_CURRENT_SOURCE_DIR}/sampler_test.cpp)
rwkv_add_unit_test(rwkv_state_cache_test ${CMAKE_CURRENT_SOURCE_DIR}/state_cache_test.cpp)
rwkv_add_unit_test(rwkv_inference_engine_test ${CMAKE_CURRENT_SOURCE_DIR}/inference_engine_test.cpp)
