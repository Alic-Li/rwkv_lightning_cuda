#include "rwkv/server/router/route_helpers.hpp"
#include "rwkv/server/router/routes.hpp"

#include <algorithm>
#include <cctype>

#include "rwkv/server/rwkv_state_cache.hpp"

namespace rwkv7_server::router {

using namespace drogon;


void register_session_routes(
    HttpAppFramework& app, InferenceEngine& engine,
    const std::optional<std::string>& password,
    const std::shared_ptr<PrefillAdmissionController>& admission) {
  app.registerHandler(
      "/state/chat/completions",
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

        std::string session_id;
        try {
          session_id = session_id_from_request(req, *json);
        } catch (const std::exception& error) {
          cb(json_response(make_error(error.what()), k400BadRequest));
          return;
        }
        std::string state_id;
        try {
          state_id = state_id_from_request(req, *json);
        } catch (const std::exception& error) {
          cb(json_response(make_error(error.what()), k400BadRequest));
          return;
        }
        const auto prompts = parse_contents(*json);
        if (session_id.empty()) {
          cb(json_response(make_error("Missing session_id"), k400BadRequest));
          return;
        }
        if (prompts.size() != 1) {
          cb(json_response(make_error("Request must contain exactly one prompt"), k400BadRequest));
          return;
        }

        GenerateOptions options;
        try {
          options = parse_options(*json);
        } catch (const std::exception &e) {
          cb(json_response(make_error(e.what()), k400BadRequest));
          return;
        }
        session_id += ":" + rwkv7_miss::effective_model_key(
                                engine.model()->runtime_identity(),
                                options.adapter.get(), "session", false);
        const int prompt_tokens = engine.count_tokens(prompts.front());
        const auto model = (*json).get("model", engine.model_name()).asString();
        auto active = RequestRegistry::instance().start(
            "state.chat.completions",
            model,
            options.max_tokens,
            prompt_tokens,
            session_id);
        if ((*json).get("stream", false).asBool()) {
          cb(make_sse_response([&engine, admission, session_id, state_id, prompts, options,
                                active, model, chunk_size = parse_chunk_size(*json, 8)](
                                   ResponseStreamPtr stream) {
            start_streaming_task(
                std::move(stream),
                active->id,
                model,
                1,
                [&, admission, active, session_id, state_id, prompts, options, chunk_size](
                    const InferenceEngine::StreamCallback& emit) {
                  auto permit = admission->acquire(1, "state.chat.completions", [active]() {
                    return active->stop_requested.load() || active->pause_requested.load();
                  });
                  if (!permit.has_value()) {
                    return cancelled_generation(1);
                  }
                  auto cached_state = state_id.empty()
                      ? StateCacheManager::instance().get_state(session_id)
                      : std::optional<GenerationState>{};
                  auto state = !state_id.empty()
                      ? create_request_state(engine, state_id)
                      : (cached_state.has_value() ? std::move(*cached_state)
                                                  : engine.model()->create_state(1));
                  auto state_ptr = std::make_shared<GenerationState>(std::move(state));
                  const auto stats = engine.batch_generate_state_stream(
                      prompts,
                      *state_ptr,
                      options,
                      chunk_size,
                      [&](int index, const std::string& chunk) {
                        active->generated_tokens.fetch_add(engine.count_tokens(chunk));
                        return emit(index, chunk);
                      },
                      [active]() {
                        return active->stop_requested.load() || active->pause_requested.load();
                      });
                  StateCacheManager::instance().put_state(session_id, *state_ptr);
                  return stats;
                },
                [active]() {
                  RequestRegistry::instance().finish(active);
                });
          }));
          return;
        }

        std::optional<PrefillAdmissionController::Permit> permit;
        try {
          permit = admission->acquire(1, "state.chat.completions", [active]() {
            return active->stop_requested.load() || active->pause_requested.load();
          });
        } catch (const PrefillBatchLimitExceeded& error) {
          RequestRegistry::instance().finish(active);
          cb(json_response(prefill_limit_error(error), k400BadRequest));
          return;
        }
        if (!permit.has_value()) {
          RequestRegistry::instance().finish(active);
          cb(json_response(make_error("Request cancelled"), k409Conflict));
          return;
        }
        auto& manager = StateCacheManager::instance();
        GenerationState state;
        try {
          auto cached_state = state_id.empty() ? manager.get_state(session_id) : std::optional<GenerationState>{};
          state = !state_id.empty()
              ? create_request_state(engine, state_id)
              : (cached_state.has_value() ? std::move(*cached_state) : engine.model()->create_state(1));
        } catch (const std::exception& error) {
          RequestRegistry::instance().finish(active);
          cb(json_response(make_error(error.what()), k400BadRequest));
          return;
        }
        std::vector<std::string> texts(1);
        const auto stats = engine.batch_generate_state_stream(
            prompts,
            state,
            options,
            0,
            [&](int index, const std::string& chunk) {
              texts.at(static_cast<size_t>(index)) += chunk;
              return true;
            },
            [active]() {
              return active->stop_requested.load() || active->pause_requested.load();
            });
        manager.put_state(session_id, state);
        Json::Value resp;
        resp["id"] = "rwkv7-fast-state";
        resp["object"] = "chat.completion";
        resp["model"] = model;
        resp["choices"] = build_choices(texts, stats);
        RequestRegistry::instance().finish(active);
        cb(json_response(std::move(resp)));
      },
      {Post});

  app.registerHandler(
      "/state/status",
      [password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
        auto json = req->getJsonObject();
        Json::Value body = json ? *json : Json::Value(Json::objectValue);
        HttpResponsePtr auth_resp;
        if (!check_password(req, body, password, auth_resp)) {
          cb(auth_resp);
          return;
        }

        const auto summary = StateCacheManager::instance().list_all_states();
        Json::Value resp;
        resp["status"] = "success";
        resp["l1_cache_count"] = static_cast<int>(summary.l1_cache.size());
        resp["l2_cache_count"] = static_cast<int>(summary.l2_cache.size());
        resp["database_count"] = static_cast<int>(summary.database.size());
        resp["total_sessions"] = static_cast<int>(
            summary.l1_cache.size() + summary.l2_cache.size() + summary.database.size());
        resp["sessions"] = Json::arrayValue;

        auto append = [&](const std::vector<std::string>& ids, const std::string& level) {
          for (const auto& id : ids) {
            Json::Value item;
            item["session_id"] = id;
            item["cache_level"] = level;
            if (level == "Database (Disk)") {
              if (auto ts = StateCacheManager::instance().get_db_timestamp(id); ts.has_value()) {
                item["timestamp"] = *ts;
              }
            }
            resp["sessions"].append(item);
          }
        };

        append(summary.l1_cache, "L1 (VRAM)");
        append(summary.l2_cache, "L2 (RAM)");
        append(summary.database, "Database (Disk)");
        cb(json_response(std::move(resp)));
      },
      {Post});

  app.registerHandler(
      "/state/delete",
      [password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
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

        std::string session_id;
        try {
          session_id = session_id_from_request(req, *json);
        } catch (const std::exception& error) {
          cb(json_response(make_error(error.what()), k400BadRequest));
          return;
        }
        if (session_id.empty()) {
          cb(json_response(make_error("Missing session_id"), k400BadRequest));
          return;
        }
        auto &state_cache = StateCacheManager::instance();
        bool ok = state_cache.delete_state_from_any_level(session_id);
        const auto namespaces = state_cache.list_all_states();
        const auto prefix = session_id + ":";
        for (const auto *level :
             {&namespaces.l1_cache, &namespaces.l2_cache, &namespaces.database})
          for (const auto &key : *level)
            if (key.size() == prefix.size() + 64 &&
                key.compare(0, prefix.size(), prefix) == 0 &&
                std::all_of(key.begin() + prefix.size(), key.end(),
                            [](unsigned char c) { return std::isxdigit(c); }))
              ok = state_cache.delete_state_from_any_level(key) || ok;
        Json::Value resp;
        resp["status"] = ok ? "success" : "not_found";
        resp["message"] = ok ? ("Session " + session_id + " deleted successfully")
                             : ("Session " + session_id + " not found");
        cb(json_response(std::move(resp), ok ? k200OK : k404NotFound));
      },
      {Post});
}

}  // namespace rwkv7_server::router
