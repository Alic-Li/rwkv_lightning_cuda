#pragma once
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
  // Allow a NCCL data-parallel run to resume on a different GPU count when the
  // global batch_size, data order, and every other training config field match.
  Json::Value want = expected;
  if (saved.isObject() && want.isObject() && saved["batch_size"] == want["batch_size"] &&
      saved.isMember("dp_world_size") && want.isMember("dp_world_size")) {
    for (const char *key : {"dp_world_size", "dp_micro_bsz"}) {
      saved.removeMember(key);
      want.removeMember(key);
    }
  }
  return Json::writeString(Json::StreamWriterBuilder{}, saved) ==
         Json::writeString(Json::StreamWriterBuilder{}, want);
}
} // namespace rwkv_bf16_miss
