#include "rwkv/server/router/route_helpers.hpp"
#include "rwkv/server/router/routes.hpp"

#include "rwkv/server/rwkv_state_cache.hpp"

namespace rwkv7_server::router {

using namespace drogon;


void register_chat_routes(
    HttpAppFramework& app, InferenceEngine& engine,
    const std::optional<std::string>& password,
    const std::shared_ptr<PrefillAdmissionController>& admission) {
  app.registerHandler(
      "/v1/chat/completions",
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
        const auto prompt = format_openai_prompt(*json, engine, !state_id.empty());
        // std::cout << prompt << std::endl; // Debug Prompt
        GenerateOptions options;
        try {
          options = parse_options(*json);
        } catch (const std::exception &e) {
          cb(json_response(make_error(e.what()), k400BadRequest));
          return;
        }
        if ((*json).isMember("think_type") || (*json).isMember("think") || (*json).isMember("enable_think")) {
          options.force_reasoning = force_reasoning_for_think_type(parse_think_type(*json));
        }
        const int prompt_tokens = engine.count_tokens(prompt);
        const auto model = (*json).get("model", engine.model_name()).asString();
        auto active = RequestRegistry::instance().start(
            "chat.completions",
            model,
            options.max_tokens,
            prompt_tokens);

        if ((*json).get("stream", false).asBool()) {
          cb(make_sse_response([&engine, admission, prompt, state_id, options, active, model,
                                chunk_size = parse_chunk_size(*json, 2)](
                                   ResponseStreamPtr stream) {
            start_streaming_task(
                std::move(stream),
                active->id,
                model,
                1,
                [&, admission, active, prompt, state_id, options, chunk_size](
                    const InferenceEngine::StreamCallback& emit) {
                  auto permit = admission->acquire(1, "chat.completions", [active]() {
                    return active->stop_requested.load() || active->pause_requested.load();
                  });
                  if (!permit.has_value()) {
                    return cancelled_generation(1);
                  }
                  auto state_ptr = std::make_shared<GenerationState>(
                      create_request_state(engine, state_id));
                  auto logits_ptr = std::make_shared<DeviceLogits>();
                  const auto prefill_begin = std::chrono::steady_clock::now();
                  bind_adapter(*state_ptr, options.adapter);
                  const int prefill_tokens = engine.prefill_prompt(prompt, *state_ptr, *logits_ptr);
                  const auto prefill_end = std::chrono::steady_clock::now();
                  record_prefill_metrics(active, prefill_tokens, prefill_begin, prefill_end);

                  const auto decode_begin = std::chrono::steady_clock::now();
                  const auto stats = engine.generate_from_logits_stream(
                      *state_ptr,
                      *logits_ptr,
                      options,
                      chunk_size,
                      [&](int index, const std::string& chunk) {
                        const int tokens = engine.count_tokens(chunk);
                        active->generated_tokens.fetch_add(tokens);
                        update_decode_metrics(active, decode_begin);
                        return emit(index, chunk);
                      },
                      [active]() {
                        return active->stop_requested.load() || active->pause_requested.load();
                      });
                  update_decode_metrics(active, stats.generated_tokens, stats.decode_seconds);
                  if (active->pause_requested.load()) {
                    RequestRegistry::PausedRequest paused;
                    paused.id = active->id;
                    paused.endpoint = active->endpoint;
                    paused.model = active->model;
                    paused.created = active->created;
                    paused.generated_tokens = active->generated_tokens.load();
                    paused.chunk_size = chunk_size;
                    paused.options = options;
                    state_ptr->adapter_gpu.reset();
                    rwkv7_miss::adapter_cache().trim();
                    paused.state = state_ptr;
                    paused.logits = logits_ptr;
                    StateCacheManager::instance().put_state(paused.id, *state_ptr);
                    RequestRegistry::instance().put_paused(std::move(paused));
                  }
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
          permit = admission->acquire(1, "chat.completions", [active]() {
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
        std::shared_ptr<GenerationState> state_ptr;
        try {
          state_ptr = std::make_shared<GenerationState>(
              create_request_state(engine, state_id));
        } catch (const std::exception& error) {
          RequestRegistry::instance().finish(active);
          cb(json_response(make_error(error.what()), k400BadRequest));
          return;
        }
        auto logits_ptr = std::make_shared<DeviceLogits>();
        std::string completion;

        const auto prefill_begin = std::chrono::steady_clock::now();
        bind_adapter(*state_ptr, options.adapter);
        const int prefill_tokens =
            engine.prefill_prompt(prompt, *state_ptr, *logits_ptr);
        const auto prefill_end = std::chrono::steady_clock::now();
        record_prefill_metrics(active, prefill_tokens, prefill_begin, prefill_end);

        const auto decode_begin = std::chrono::steady_clock::now();
        const auto stats = engine.generate_from_logits_stream(
            *state_ptr,
            *logits_ptr,
            options,
            parse_chunk_size(*json, 2),
            [&](int, const std::string& chunk) {
              completion += chunk;
              const int tokens = engine.count_tokens(chunk);
              active->generated_tokens.fetch_add(tokens);
              update_decode_metrics(active, decode_begin);
              return true;
            },
            [active]() {
              return active->stop_requested.load() || active->pause_requested.load();
            });
        update_decode_metrics(active, stats.generated_tokens, stats.decode_seconds);
        if (active->pause_requested.load()) {
          RequestRegistry::PausedRequest paused;
          paused.id = active->id;
          paused.endpoint = active->endpoint;
          paused.model = active->model;
          paused.created = active->created;
          paused.generated_tokens = active->generated_tokens.load();
          paused.chunk_size = parse_chunk_size(*json, 2);
          paused.options = options;
          state_ptr->adapter_gpu.reset();
          rwkv7_miss::adapter_cache().trim();
          paused.state = state_ptr;
          paused.logits = logits_ptr;
          StateCacheManager::instance().put_state(paused.id, *state_ptr);
          RequestRegistry::instance().put_paused(std::move(paused));
        }
        RequestRegistry::instance().finish(active);

        cb(json_response(build_openai_response(engine, *json, prompt, completion, stats)));
      }),
      {Post});
}

}  // namespace rwkv7_server::router
