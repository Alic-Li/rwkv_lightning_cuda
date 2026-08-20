#pragma once

#include <optional>
#include <string>

namespace rwkv7_server {

class ModelRouter;

void register_api_routes(
    ModelRouter& models,
    const std::optional<std::string>& password);

}  // namespace rwkv7_server
