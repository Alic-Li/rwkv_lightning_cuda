#pragma once
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <json/json.h>

namespace rwkv_bf16_miss {
inline bool checkpoint_config_matches(Json::Value saved,
                                      const Json::Value &expected) {
  // Older online-RL writers inserted these absent CLI-only fields while
  // saving the preceding checkpoint. Accept only that null-key artifact;
  // real seed/schedule values and every other config field must still match.
  if (saved.isObject() && expected["kind"] == "online_rl") {
    for (const char *key : {"seed", "planned_steps"})
      if (!expected.isMember(key) && saved.isMember(key) && saved[key].isNull())
        saved.removeMember(key);
  }
  const char *reshape_env = std::getenv("RWKV_ALLOW_BATCH_RESHAPE");
  const bool allow_batch_reshape =
      reshape_env && std::strcmp(reshape_env, "1") == 0;
  // Allow a NCCL data-parallel run to resume on a different GPU count, micro-batch,
  // or chunk size when global batch_size, data order, and training config match
  // (or when RWKV_ALLOW_BATCH_RESHAPE=1 explicitly authorizes elastic batch scaling).
  Json::Value want = expected;
  if (saved.isObject() && want.isObject() &&
      (allow_batch_reshape || saved["batch_size"] == want["batch_size"]) &&
      (saved.isMember("dp_world_size") || want.isMember("dp_world_size"))) {
    for (const char *key :
         {"dp_world_size", "dp_micro_bsz", "dp_partition", "chunk", "gemm_mode"}) {
      saved.removeMember(key);
      want.removeMember(key);
    }
    if (allow_batch_reshape) {
      for (const char *key :
           {"batch_size", "planned_steps", "lr", "lr_final", "warmup_steps"}) {
        saved.removeMember(key);
        want.removeMember(key);
      }
    }
  }
  return Json::writeString(Json::StreamWriterBuilder{}, saved) ==
         Json::writeString(Json::StreamWriterBuilder{}, want);
}
} // namespace rwkv_bf16_miss
