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
  return Json::writeString(Json::StreamWriterBuilder{}, saved) ==
         Json::writeString(Json::StreamWriterBuilder{}, expected);
}
} // namespace rwkv_bf16_miss
