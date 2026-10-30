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
