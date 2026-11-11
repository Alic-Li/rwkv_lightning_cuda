#include "rwkv/server/router/route_helpers.hpp"
#include "rwkv/server/router/routes.hpp"

#include <cstdio>
#include <exception>
#include <string>

namespace rwkv7_server::router {

using namespace drogon;

void register_legacy_routes(
    HttpAppFramework& app, InferenceEngine& engine,
    const std::optional<std::string>& password) {
  // The admission controller fixes its hard limit from the free VRAM it sees
  // when constructed. The first generation permanently claims a few hundred MiB
  // (cuBLASLt workspace, lazily loaded CUDA modules), so measuring before it
  // over-promises: on a 16 GiB card with a 7B BF16 model the limit read 22
  // while at most 16 ever fit. Run one single-row generation over a full
  // prefill chunk first. A wider warmup is not safe: the estimate is itself
  // optimistic near its limit. Whatever drift remains (more kernel variants at
  // larger batches) is handled by the controller's idle-stall rejection.
  try {
    std::string prompt;
    for (int i = 0; i < engine.prefill_chunk_size(); ++i) {
      prompt += " hello";
    }
    GenerateOptions warmup;
    warmup.max_tokens = 2;
    warmup.stop_tokens.clear();
    engine.batch_generate({prompt}, warmup);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "[PrefillQueue] warmup generation failed: %s\n", error.what());
  }
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
