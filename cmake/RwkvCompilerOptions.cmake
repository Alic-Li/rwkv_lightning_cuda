include_guard(GLOBAL)

function(rwkv_optimize_target target_name)
  target_compile_definitions(${target_name} PRIVATE
    $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
  )

  if(MSVC)
    target_compile_definitions(${target_name} PRIVATE
      NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS
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
  else()
    target_compile_options(${target_name} PRIVATE
      $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:Debug>>>:-O3>
      $<$<AND:$<COMPILE_LANGUAGE:HIP>,$<NOT:$<CONFIG:Debug>>>:-O3>
    )
  endif()
endfunction()

function(rwkv_optimize_backend_target target_name)
  rwkv_optimize_target(${target_name})

  if(NOT MSVC AND RWKV7_FAST_V4_GC_SECTIONS)
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections -fdata-sections>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffunction-sections,-fdata-sections>
    )
    target_link_options(${target_name} PRIVATE -Wl,--gc-sections)
  endif()

  target_compile_options(${target_name} PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-O3 --use_fast_math --extra-device-vectorization --ptxas-options=-O3>
  )
endfunction()

function(rwkv_strip_release_binary target_name)
  if(RWKV7_FAST_V4_AUTO_STRIP AND CMAKE_STRIP AND NOT MSVC)
    add_custom_command(TARGET ${target_name} POST_BUILD
      COMMAND "$<$<CONFIG:Release>:${CMAKE_STRIP}>"
              "$<$<CONFIG:Release>:--strip-unneeded>"
              "$<$<CONFIG:Release>:$<TARGET_FILE:${target_name}>>"
      VERBATIM
      COMMAND_EXPAND_LISTS
    )
  endif()
endfunction()
