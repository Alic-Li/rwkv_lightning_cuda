#pragma once

#include <optional>
#include <string>

namespace rwkv7_server {

class InferenceEngine;

void register_api_routes(
    InferenceEngine& engine,
    const std::optional<std::string>& password);

}  // namespace rwkv7_server
