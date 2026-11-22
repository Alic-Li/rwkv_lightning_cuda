add_executable(rwkv_nora_tune_server main.cpp)
target_link_libraries(rwkv_nora_tune_server PRIVATE rwkv_online_rl)
rwkv_optimize_target(rwkv_nora_tune_server)
rwkv_configure_runtime_bundle(rwkv_nora_tune_server
  ASSETS ${PROJECT_SOURCE_DIR}/assets/rwkv_vocab_v20230424.txt)
if(TARGET rwkv_online_rl_dp)
  add_executable(rwkv_nora_tune_server_dp main.cpp)
  target_link_libraries(rwkv_nora_tune_server_dp PRIVATE rwkv_online_rl_dp)
  rwkv_optimize_target(rwkv_nora_tune_server_dp)
  rwkv_configure_runtime_bundle(rwkv_nora_tune_server_dp
    ASSETS ${PROJECT_SOURCE_DIR}/assets/rwkv_vocab_v20230424.txt)
endif()
