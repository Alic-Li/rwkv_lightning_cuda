include_guard(GLOBAL)

function(rwkv_configure_runtime_bundle target_name)
  set(options)
  set(one_value_args)
  set(multi_value_args ASSETS)
  cmake_parse_arguments(RWKV_BUNDLE "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(APPLE)
    set(rwkv_origin "@loader_path")
  elseif(UNIX)
    set(rwkv_origin "$ORIGIN")
  endif()

  if(rwkv_origin)
    set_target_properties(${target_name} PROPERTIES
      BUILD_RPATH "${rwkv_origin};${rwkv_origin}/lib"
      INSTALL_RPATH "${rwkv_origin};${rwkv_origin}/lib"
      BUILD_RPATH_USE_ORIGIN ON
    )
    if(UNIX AND NOT APPLE)
      target_link_options(${target_name} PRIVATE -Wl,--disable-new-dtags)
    endif()
  endif()

  string(JOIN "|" rwkv_asset_files ${RWKV_BUNDLE_ASSETS})
  add_custom_target(bundle_${target_name}
    COMMAND ${CMAKE_COMMAND}
      -DINPUT_FILE=$<TARGET_FILE:${target_name}>
      -DOUTPUT_DIR=${PROJECT_BINARY_DIR}/bundle/${target_name}
      -DLIB_OUTPUT_DIR=${PROJECT_BINARY_DIR}/bundle/${target_name}/lib
      -DASSET_FILES=${rwkv_asset_files}
      -DCOPY_BINARY=ON
      -P ${PROJECT_SOURCE_DIR}/cmake/packaging/CopyRuntimeDependencies.cmake
    DEPENDS ${target_name}
    VERBATIM
  )
  set_property(GLOBAL APPEND PROPERTY RWKV_BUNDLE_TARGETS bundle_${target_name})
endfunction()
