#include "rwkv/server/router/route_helpers.hpp"
#include "rwkv/server/router/routes.hpp"

#include "rwkv/server/rwkv_state_cache.hpp"

namespace rwkv7_server::router {

using namespace drogon;

Json::Value build_status_response(
    const InferenceEngine& engine,
    PrefillAdmissionController& admission) {
  Json::Value resp;
  resp["status"] = "running";
  resp["api_version"] = "1.3";
  resp["engine_version"] = "albatross-1.3.0";
  resp["model"]["id"] = engine.model_name();
  resp["model"]["name"] = engine.model_name();
  resp["model"]["path"] = engine.model()->model_path();
  resp["capabilities"]["chat_messages"] = true;
  resp["capabilities"]["completion"] = true;
  resp["capabilities"]["batch_completion"] = true;
  resp["capabilities"]["stream"] = true;
  resp["capabilities"]["stop"] = true;
  resp["capabilities"]["token_count"] = true;
  resp["capabilities"]["metrics"] = true;
  resp["capabilities"]["session_cache"] = true;
  resp["capabilities"]["pause_resume"] = true;
  resp["capabilities"]["think_type"] = true;
  resp["capabilities"]["concurrent_generation"] = true;
  resp["capabilities"]["chunk_prefill"] = true;
  resp["prefill_chunk_size"] = engine.prefill_chunk_size();
  const auto queue = admission.snapshot();
  resp["prefill_queue"]["hard_max_bsz"] = queue.hard_max_batch_size;
  resp["prefill_queue"]["dynamic_max_bsz"] = queue.dynamic_max_batch_size;
  resp["prefill_queue"]["reserved_bsz"] = queue.reserved_batch_size;
  resp["prefill_queue"]["available_bsz"] =
      std::max(0, queue.dynamic_max_batch_size - queue.reserved_batch_size);
  resp["prefill_queue"]["queued_requests"] = static_cast<Json::UInt64>(queue.queued_requests);
  resp["prefill_queue"]["free_vram_bytes"] = static_cast<Json::UInt64>(queue.capacity.free_vram_bytes);
  resp["prefill_queue"]["total_vram_bytes"] = static_cast<Json::UInt64>(queue.capacity.total_vram_bytes);
  resp["prefill_queue"]["reserve_vram_bytes"] = static_cast<Json::UInt64>(queue.capacity.reserve_vram_bytes);
  resp["prefill_queue"]["bytes_per_bsz"] = static_cast<Json::UInt64>(queue.capacity.bytes_per_batch);
  resp["active_request"] = RequestRegistry::instance().active_json();
  resp["active_requests"] = RequestRegistry::instance().active_requests_json();
  resp["last_request"] = RequestRegistry::instance().last_json();
  resp["paused_requests"] = RequestRegistry::instance().paused_json();
  return resp;
}

void register_server_routes(
    HttpAppFramework& app, InferenceEngine& engine,
    const std::optional<std::string>& password,
    const std::shared_ptr<PrefillAdmissionController>& admission) {
  app.registerHandler(
      "/v1/server/status",
      [&engine, admission](const HttpRequestPtr&, std::function<void(const HttpResponsePtr&)>&& cb) {
        cb(json_response(build_status_response(engine, *admission)));
      },
      {Get});

  app.registerHandler(
      "/v1/server/stop",
      [password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
        auto json = req->getJsonObject();
        Json::Value body = json ? *json : Json::Value(Json::objectValue);
        HttpResponsePtr auth_resp;
        if (!check_password(req, body, password, auth_resp)) {
          cb(auth_resp);
          return;
        }

        const auto request_id = body.get("request_id", "").asString();
        const bool stopped = RequestRegistry::instance().stop_active(request_id);
        Json::Value resp;
        resp["ok"] = true;
        resp["stopped"] = stopped;
        resp["active_request"] = RequestRegistry::instance().active_json();
        cb(json_response(std::move(resp)));
      },
      {Post});

  app.registerHandler(
      "/v1/server/pause",
      [password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
        auto json = req->getJsonObject();
        Json::Value body = json ? *json : Json::Value(Json::objectValue);
        HttpResponsePtr auth_resp;
        if (!check_password(req, body, password, auth_resp)) {
          cb(auth_resp);
          return;
        }

        const auto target_id = body.get("request_id", "").asString();
        const auto request_id = RequestRegistry::instance().pause_active(target_id);
        Json::Value resp;
        resp["ok"] = true;
        resp["paused"] = request_id.has_value();
        if (request_id.has_value()) {
          resp["request_id"] = *request_id;
        }
        resp["active_request"] = RequestRegistry::instance().active_json();
        cb(json_response(std::move(resp)));
      },
      {Post});

  app.registerHandler(
      "/v1/server/resume",
      [&engine, password, admission](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
        auto json = req->getJsonObject();
        if (!json) {
          cb(json_response(make_error("Invalid JSON"), k400BadRequest));
          return;
        }
        HttpResponsePtr auth_resp;
        if (!check_password(req, *json, password, auth_resp)) {
          cb(auth_resp);
          return;
        }

        const auto request_id = (*json).get("request_id", (*json).get("session_id", "").asString()).asString();
        if (request_id.empty()) {
          cb(json_response(make_error("Missing request_id"), k400BadRequest));
          return;
        }

        auto paused = RequestRegistry::instance().take_paused(request_id);
        if (!paused.has_value()) {
          cb(json_response(make_error("Paused request not found"), k404NotFound));
          return;
        }
        if (!paused->state || !paused->logits) {
          cb(json_response(make_error("Paused request state is incomplete"), k500InternalServerError));
          return;
        }

        GenerateOptions options = paused->options;
        options.force_reasoning_token_offset = paused->generated_tokens;
        if (!(*json).isMember("max_tokens")) {
          options.max_tokens = std::max(0, paused->options.max_tokens - paused->generated_tokens);
        }
        try {
          options = parse_options(*json, options);
          bind_adapter(*paused->state, options.adapter);
        } catch (const std::exception &e) {
          RequestRegistry::instance().put_paused(std::move(*paused));
          cb(json_response(make_error(e.what()), k400BadRequest));
          return;
        }
        paused->state->request_started = std::chrono::steady_clock::now();
        paused->state->adapter_cold = false;
        paused->state->adapter_h2d_ms = 0;
        const int chunk_size = parse_chunk_size(*json, paused->chunk_size);
        const std::string model = (*json).get("model", paused->model).asString();

        auto active = RequestRegistry::instance().start(
            "server.resume",
            model,
            options.max_tokens,
            0,
            request_id,
            request_id);
        active->generated_tokens.store(paused->generated_tokens);

        auto state_ptr = paused->state;
        auto logits_ptr = paused->logits;
        const auto saved_generated_tokens = paused->generated_tokens;
        cb(make_sse_response([&engine, admission, active, state_ptr, logits_ptr, options, chunk_size, request_id, model,
                              saved_generated_tokens](ResponseStreamPtr stream) {
          start_streaming_task(
              std::move(stream),
              request_id,
              model,
              1,
              [&, admission, active, state_ptr, logits_ptr, options, chunk_size, saved_generated_tokens](
                  const InferenceEngine::StreamCallback& emit) {
                auto permit = admission->acquire(1, "server.resume", [active]() {
                  return active->stop_requested.load() || active->pause_requested.load();
                });
                if (!permit.has_value()) {
                  return cancelled_generation(1);
                }
                const auto stats = engine.generate_from_logits_stream(
                    *state_ptr,
                    *logits_ptr,
                    options,
                    chunk_size,
                    [&](int index, const std::string& chunk) {
                      const int tokens = engine.count_tokens(chunk);
                      active->generated_tokens.fetch_add(tokens);
                      return emit(index, chunk);
                    },
                    [active]() {
                      return active->stop_requested.load() || active->pause_requested.load();
                    });
                if (stats.decode_seconds > 0.0) {
                  active->decode_speed.store(stats.generated_tokens / stats.decode_seconds);
                }
                if (active->pause_requested.load()) {
                  RequestRegistry::PausedRequest paused_again;
                  paused_again.id = active->id;
                  paused_again.endpoint = active->endpoint;
                  paused_again.model = active->model;
                  paused_again.created = active->created;
                  paused_again.generated_tokens = active->generated_tokens.load();
                  paused_again.chunk_size = chunk_size;
                  paused_again.options = options;
                  paused_again.options.max_tokens += saved_generated_tokens;
                  state_ptr->adapter_gpu.reset();
                  rwkv7_miss::adapter_cache().trim();
                  paused_again.state = state_ptr;
                  paused_again.logits = logits_ptr;
                  StateCacheManager::instance().put_state(paused_again.id, *state_ptr);
                  RequestRegistry::instance().put_paused(std::move(paused_again));
                }
                return stats;
              },
              [active]() {
                RequestRegistry::instance().finish(active);
              });
        }));
      },
      {Post});
}

}  // namespace rwkv7_server::router
