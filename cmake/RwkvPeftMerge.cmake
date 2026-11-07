function(rwkv_add_peft_merge)
  find_package(OpenSSL REQUIRED COMPONENTS Crypto)
  find_package(ZLIB REQUIRED)
  find_package(jsoncpp CONFIG QUIET)
  if(TARGET JsonCpp::JsonCpp)
    set(merge_json JsonCpp::JsonCpp)
  elseif(TARGET jsoncpp_lib)
    set(merge_json jsoncpp_lib)
  else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(MERGE_JSON REQUIRED IMPORTED_TARGET jsoncpp)
    set(merge_json PkgConfig::MERGE_JSON)
  endif()
  find_package(OpenMP QUIET COMPONENTS CXX)
  add_executable(rwkv_peft_weight_merge
    ${PROJECT_SOURCE_DIR}/src/rwkv_peft_weight_merge.cpp
    ${PROJECT_SOURCE_DIR}/src/io/pth_archive.cpp
    ${PROJECT_SOURCE_DIR}/src/io/pth_tensor.cpp
    ${PROJECT_SOURCE_DIR}/src/io/fingerprint.cpp
  )
  target_compile_features(rwkv_peft_weight_merge PRIVATE cxx_std_17)
  target_include_directories(rwkv_peft_weight_merge PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(rwkv_peft_weight_merge PRIVATE ${merge_json} OpenSSL::Crypto ZLIB::ZLIB rwkv::version)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(rwkv_peft_weight_merge PRIVATE OpenMP::OpenMP_CXX)
  endif()
  if(MSVC)
    target_compile_options(rwkv_peft_weight_merge PRIVATE /utf-8 /EHsc)
    target_compile_definitions(rwkv_peft_weight_merge PRIVATE NOMINMAX)
  endif()
  set_target_properties(rwkv_peft_weight_merge PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
endfunction()
