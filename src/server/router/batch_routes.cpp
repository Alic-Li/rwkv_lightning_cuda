#include "rwkv/server/router/route_helpers.hpp"
#include "rwkv/server/router/routes.hpp"
namespace rwkv7_server::router {

using namespace drogon;


void register_batch_routes(
    HttpAppFramework& app, InferenceEngine& engine,
    const std::optional<std::string>& password,
    const std::shared_ptr<PrefillAdmissionController>& admission) {
  app.registerHandler(
      "/v1/batch/completions",
      asynchronous_generation([&engine, password, admission](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
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
        std::string state_id;
        try {
          state_id = state_id_from_request(req, *json);
        } catch (const std::exception& error) {
          cb(json_response(make_error(error.what()), k400BadRequest));
          return;
        }

        const auto prompts = parse_contents(*json);
        if (prompts.empty()) {
          cb(json_response(make_error("Empty prompts list"), k400BadRequest));
          return;
        }
        GenerateOptions options;
        try {
          options = parse_options(*json);
        } catch (const std::exception &e) {
          cb(json_response(make_error(e.what()), k400BadRequest));
          return;
        }
        const bool metrics_requested = parse_metrics_requested(*json);
        int prompt_tokens = 0;
        if (metrics_requested) {
          for (const auto& prompt : prompts) {
            prompt_tokens += engine.count_tokens(prompt);
          }
        }
        const auto model = (*json).get("model", engine.model_name()).asString();
        auto active = RequestRegistry::instance().start(
            "batch.completions",
            model,
            options.max_tokens,
            prompt_tokens);
        if ((*json).get("stream", false).asBool()) {
          cb(make_sse_response([&engine, admission, active, prompts, state_id, options, model, metrics_requested,
                                chunk_size = parse_chunk_size(*json, 8)](
                                   ResponseStreamPtr stream) {
            auto matched_stops = std::make_shared<std::vector<std::string>>(prompts.size());
            start_streaming_task(
                std::move(stream),
                active->id,
                model,
                static_cast<int>(prompts.size()),
                [&, admission, active, prompts, state_id, options, chunk_size, metrics_requested, matched_stops](const InferenceEngine::StreamCallback& emit) {
                  auto permit = admission->acquire(
                      static_cast<int>(prompts.size()),
                      "batch.completions",
                      [active]() {
                        return active->stop_requested.load() || active->pause_requested.load();
                      });
                  if (!permit.has_value()) {
                    return cancelled_generation(static_cast<int>(prompts.size()));
                  }
                  InferenceEngine::StatsCallback on_prefill_complete;
                  if (metrics_requested) {
                    on_prefill_complete = [active](const InferenceEngine::GenerationStats& stats) {
                      record_generation_stats(active, stats);
                    };
                  }
                  auto state = create_request_state(
                      engine, state_id, static_cast<int>(prompts.size()));
                  const auto stats = engine.batch_generate_stream_with_state(
                      prompts,
                      state,
                      options,
                      chunk_size,
                      [&, metrics_requested](int index, const std::string& chunk) {
                        if (metrics_requested) {
                          active->generated_tokens.fetch_add(engine.count_tokens(chunk));
                        }
                        return emit(index, chunk);
                      },
                      [active]() {
                        return active->stop_requested.load() || active->pause_requested.load();
                      },
                      on_prefill_complete, state_id,
                      [matched_stops](const InferenceEngine::GenerationStats& stats) {
                        *matched_stops = stats.matched_stops;
                      });
                  if (metrics_requested) {
                    record_generation_stats(active, stats);
                  }
                  return stats;
                },
                [active]() {
                  RequestRegistry::instance().finish(active);
                }, matched_stops);
          }));
          return;
        }

        Json::Value resp;
        resp["id"] = "rwkv7-fast-batch";
        resp["object"] = "chat.completion";
        resp["model"] = model;
        std::optional<PrefillAdmissionController::Permit> permit;
        try {
          permit = admission->acquire(
              static_cast<int>(prompts.size()),
              "batch.completions",
              [active]() {
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
        std::vector<std::string> results(prompts.size());
        InferenceEngine::GenerationStats stats;
        try {
          auto state = create_request_state(
              engine, state_id, static_cast<int>(prompts.size()));
          stats = engine.batch_generate_stream_with_state(
              prompts,
              state,
              options,
              parse_chunk_size(*json, 0),
              [&](int index, const std::string& chunk) {
                if (index >= 0 && index < static_cast<int>(results.size())) {
                  results[static_cast<size_t>(index)] += chunk;
                }
                return true;
              },
              [active]() {
                return active->stop_requested.load() || active->pause_requested.load();
              }, {}, state_id);
        } catch (const std::exception& error) {
          RequestRegistry::instance().finish(active);
          cb(json_response(make_error(error.what()), k400BadRequest));
          return;
        }
        if (metrics_requested) {
          record_generation_stats(active, stats);
        }
        resp["choices"] = build_choices(results, stats);
        RequestRegistry::instance().finish(active);
        cb(json_response(std::move(resp)));
      }),
      {Post});
}

}  // namespace rwkv7_server::router
