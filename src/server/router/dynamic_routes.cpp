#include "rwkv/server/router/route_helpers.hpp"
#include "rwkv/server/router/routes.hpp"

namespace rwkv7_server::router {

using namespace drogon;

Json::Value build_models_response(const ModelRouter& models) {
  Json::Value resp;
  resp["object"] = "list";
  resp["data"] = Json::arrayValue;

  const std::string loaded_id = models.current_model_id();
  resp["loaded"] = loaded_id.empty() ? Json::Value(Json::nullValue) : Json::Value(loaded_id);
  resp["available"] = Json::arrayValue;
  for (const auto& id : models.model_ids()) {
    resp["available"].append(id);
    Json::Value model;
    model["id"] = id;
    model["object"] = "model";
    model["created"] =
        static_cast<Json::Int64>(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    model["owned_by"] = "rwkv_lighting_cuda";
    model["status"] = id == loaded_id ? "loaded" : "available";
    resp["data"].append(model);
  }
  return resp;
}

void register_dynamic_routes(
    HttpAppFramework& app, ModelRouter& models,
    const std::optional<std::string>& password) {
  app.registerPostHandlingAdvice([](const HttpRequestPtr&, const HttpResponsePtr& resp) {
    resp->addHeader("Access-Control-Allow-Origin", "*");
    resp->addHeader("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
    resp->addHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, X-RWKV-Session-Id, X-RWKV-State-Id");
  });
  const auto options_handler = [](const HttpRequestPtr&, std::function<void(const HttpResponsePtr&)>&& cb) {
    auto resp = HttpResponse::newHttpResponse();
    resp->setStatusCode(k204NoContent);
    cb(resp);
  };
  for (const auto& path : {"/v1/models", "/v1/model/load", "/v1/chat/completions", "/v1/batch/completions", "/v1/tokens/count", "/v1/server/status"}) {
    app.registerHandler(path, options_handler, {Options});
  }
  register_uploaded_state_routes(app, password);
  register_adapter_routes(app, password);

  app.registerHandler("/v1/models", [&models, password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
    Json::Value body = Json::objectValue;
    HttpResponsePtr auth;
    if (!check_password(req, body, password, auth)) { cb(auth); return; }
    cb(json_response(build_models_response(models)));
  }, {Get});

  app.registerHandler("/v1/model/load", [&models, password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
    const auto json = req->getJsonObject();
    if (!json) { cb(json_response(make_error("Invalid JSON"), k400BadRequest)); return; }
    HttpResponsePtr auth;
    if (!check_password(req, *json, password, auth)) { cb(auth); return; }
    try {
      models.load((*json).get("model", "").asString());
      Json::Value resp;
      resp["object"] = "model";
      resp["id"] = models.current_model_id();
      resp["loaded"] = true;
      cb(json_response(std::move(resp)));
    } catch (const std::exception& e) { cb(json_response(make_error(e.what()), k400BadRequest)); }
  }, {Post});

  app.registerHandler("/v1/server/status", [&models](const HttpRequestPtr&, std::function<void(const HttpResponsePtr&)>&& cb) {
    Json::Value resp;
    resp["status"] = "running";
    resp["dynamic_loading"] = models.dynamic_loading_enabled();
    const auto id = models.current_model_id();
    resp["model"]["id"] = id.empty() ? Json::Value(Json::nullValue) : Json::Value(id);
    resp["model"]["loaded"] = !id.empty();
    cb(json_response(std::move(resp)));
  }, {Get});

  app.registerHandler("/v1/tokens/count", [&models, password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
    const auto json = req->getJsonObject();
    if (!json) { cb(json_response(make_error("Invalid JSON"), k400BadRequest)); return; }
    HttpResponsePtr auth;
    if (!check_password(req, *json, password, auth)) { cb(auth); return; }
    try {
      auto lease = models.acquire();
      auto& engine = lease.engine();
      std::string text = (*json).get("text", "").asString();
      if (text.empty() && (*json).isMember("messages")) text = format_openai_prompt(*json, engine);
      if (text.empty() && !(*json).isMember("text") && !(*json).isMember("messages")) {
        cb(json_response(make_error("Missing text or messages"), k400BadRequest)); return;
      }
      Json::Value resp; resp["tokens"] = engine.count_tokens(text); cb(json_response(std::move(resp)));
    } catch (const std::exception& e) { cb(json_response(make_error(e.what()), k400BadRequest)); }
  }, {Post});

  auto generate = [&models, password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb, bool batch) {
    const auto json = req->getJsonObject();
    if (!json) { cb(json_response(make_error("Invalid JSON"), k400BadRequest)); return; }
    HttpResponsePtr auth;
    if (!check_password(req, *json, password, auth)) { cb(auth); return; }
    try {
      auto lease = std::make_shared<ModelRouter::Lease>(models.acquire());
      auto& engine = lease->engine();
      const std::string state_id = state_id_from_request(req, *json);
      const std::string model = engine.model_name();
      const auto prompts = batch
          ? parse_contents(*json)
          : std::vector<std::string>{format_openai_prompt(*json, engine, !state_id.empty())};
      if (prompts.empty()) { cb(json_response(make_error("Empty prompts list"), k400BadRequest)); return; }
      auto options = parse_options(*json);
      if (!batch && ((*json).isMember("think_type") || (*json).isMember("think") || (*json).isMember("enable_think"))) {
        options.force_reasoning = force_reasoning_for_think_type(parse_think_type(*json));
      }
      const int chunk_size = parse_chunk_size(*json, batch ? 8 : 2);
      if ((*json).get("stream", false).asBool()) {
        cb(make_sse_response([lease, prompts, state_id, options, model, chunk_size, batch](ResponseStreamPtr stream) {
          auto matched_stops = std::make_shared<std::vector<std::string>>(prompts.size());
          start_streaming_task(std::move(stream), "chatcmpl-rwkv-fast", model, static_cast<int>(prompts.size()),
            [lease, prompts, state_id, options, chunk_size, batch, matched_stops](const InferenceEngine::StreamCallback& emit) {
              auto& engine = lease->engine();
              const auto on_row_stop = [matched_stops](const InferenceEngine::GenerationStats& stats) {
                *matched_stops = stats.matched_stops;
              };
              if (batch) {
                auto state = create_request_state(engine, state_id, static_cast<int>(prompts.size()));
                return engine.batch_generate_stream_with_state(
                    prompts, state, options, chunk_size, emit, {}, {}, state_id, on_row_stop);
              }
              if (!state_id.empty()) {
                auto state = create_request_state(
                    engine, state_id, static_cast<int>(prompts.size()));
                if (prompts.size() == 1) {
                  return engine.batch_generate_state_stream(prompts, state, options, chunk_size, emit);
                } else {
                  return engine.batch_generate_stream_with_state(
                      prompts, state, options, chunk_size, emit, {}, {}, std::nullopt, on_row_stop);
                }
              } else {
                auto state = engine.model()->create_state(static_cast<int>(prompts.size()));
                return engine.batch_generate_stream_with_state(
                    prompts, state, options, chunk_size, emit, {}, {}, std::nullopt, on_row_stop);
              }
            }, [] {}, matched_stops);
        }));
        return;
      }
      std::vector<std::string> results;
      InferenceEngine::GenerationStats stats;
      if (batch) {
        auto state = create_request_state(engine, state_id, static_cast<int>(prompts.size()));
        results.resize(prompts.size());
        stats = engine.batch_generate_stream_with_state(
            prompts, state, options, 0,
            [&](int index, const std::string& chunk) {
              results.at(static_cast<size_t>(index)) += chunk;
              return true;
            }, {}, {}, state_id);
      } else if (!state_id.empty()) {
        auto state = create_request_state(
            engine, state_id, static_cast<int>(prompts.size()));
        results = prompts.size() == 1
            ? engine.batch_generate_state(prompts, state, options, &stats)
            : engine.batch_generate_with_state(prompts, state, options, &stats);
      } else {
        results = engine.batch_generate(prompts, options, &stats);
      }
      if (batch) {
        Json::Value resp; resp["id"] = "rwkv7-fast-batch"; resp["object"] = "chat.completion"; resp["model"] = model; resp["choices"] = build_choices(results, stats);
        cb(json_response(std::move(resp)));
      } else {
        auto resp = build_openai_response(engine, *json, prompts.front(), results.front(), stats);
        resp["model"] = model;
        cb(json_response(std::move(resp)));
      }
    } catch (const std::exception& e) { cb(json_response(make_error(e.what()), k400BadRequest)); }
  };
  app.registerHandler("/v1/chat/completions", [generate](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) { generate(req, std::move(cb), false); }, {Post});
  app.registerHandler("/v1/batch/completions", [generate](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) { generate(req, std::move(cb), true); }, {Post});
}

}  // namespace rwkv7_server::router
