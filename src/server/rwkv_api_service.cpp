#include "rwkv/server/rwkv_api_service.hpp"

#include <drogon/drogon.h>

#include "rwkv/server/rwkv_model_router.hpp"
#include "rwkv/server/router/routes.hpp"

namespace rwkv7_server {

void register_api_routes(
    ModelRouter& models,
    const std::optional<std::string>& password) {
  auto& app = drogon::app();
  if (!models.dynamic_loading_enabled()) {
    auto lease = models.acquire();
    router::register_legacy_routes(app, lease.engine(), password);
    return;
  }
  router::register_dynamic_routes(app, models, password);
}

}  // namespace rwkv7_server
