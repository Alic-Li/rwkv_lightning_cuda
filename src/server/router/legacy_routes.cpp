#include "rwkv/server/router/route_helpers.hpp"
#include "rwkv/server/router/routes.hpp"

namespace rwkv7_server::router {

using namespace drogon;

void register_legacy_routes(
    HttpAppFramework& app, InferenceEngine& engine,
    const std::optional<std::string>& password) {
  auto admission = std::make_shared<PrefillAdmissionController>([&engine]() {
    return engine.model()->query_prefill_capacity(engine.prefill_chunk_size());
  });
  app.registerPostHandlingAdvice([](const HttpRequestPtr&, const HttpResponsePtr& resp) {
    resp->addHeader("Access-Control-Allow-Origin", "*");
    resp->addHeader("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
    resp->addHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, X-RWKV-Session-Id, X-RWKV-State-Id");
  });

  auto handle_options = [](const HttpRequestPtr&, std::function<void(const HttpResponsePtr&)>&& cb) {
    auto resp = HttpResponse::newHttpResponse();
    resp->setStatusCode(k204NoContent);
    cb(resp);
  };

  for (const auto& path : {
           "/v1/batch/completions",
           "/translate/v1/batch-translate",
           "/state/chat/completions",
           "/state/status",
           "/state/delete",
           "/v1/server/status",
           "/v1/server/stop",
           "/v1/server/pause",
           "/v1/server/resume",
           "/v1/tokens/count",
           "/v1/chat/completions",
           "/v1/models"}) {
    app.registerHandler(path, handle_options, {Options});
  }
  register_uploaded_state_routes(app, password);
  register_adapter_routes(app, password);

  register_server_routes(app, engine, password, admission);
  register_token_routes(app, engine, password);
  register_batch_routes(app, engine, password, admission);
  register_translation_routes(app, engine, admission);
  register_session_routes(app, engine, password, admission);
  register_chat_routes(app, engine, password, admission);
  register_model_routes(app, engine, password);
}

}  // namespace rwkv7_server::router
