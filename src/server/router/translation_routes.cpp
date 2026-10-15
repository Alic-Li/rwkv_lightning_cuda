#include "rwkv/server/router/route_helpers.hpp"
#include "rwkv/server/router/routes.hpp"
namespace rwkv7_server::router {

using namespace drogon;


void register_translation_routes(
    HttpAppFramework& app, InferenceEngine& engine,
    const std::shared_ptr<PrefillAdmissionController>& admission) {
  app.registerHandler(
      "/translate/v1/batch-translate",
      [&engine, admission](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
        auto json = req->getJsonObject();
        if (!json) {
          cb(json_response(make_error("Invalid JSON"), k400BadRequest));
          return;
        }
        std::string state_id;
        try {
          state_id = state_id_from_request(req, *json);
        } catch (const std::exception& error) {
          cb(json_response(make_error(error.what()), k400BadRequest));
          return;
        }
        const auto source_lang = (*json).get("source_lang", "auto").asString();
        const auto target_lang = (*json).get("target_lang", "").asString();
        if (target_lang.empty()) {
          cb(json_response(make_error("Missing target_lang"), k400BadRequest));
          return;
        }

        std::vector<std::string> prompts;
        if ((*json).isMember("text_list") && (*json)["text_list"].isArray()) {
          for (const auto& item : (*json)["text_list"]) {
            prompts.push_back(create_translation_prompt(source_lang, target_lang, item.asString()));
          }
        }
        if (prompts.empty()) {
          cb(json_response(make_error("Empty text_list"), k400BadRequest));
          return;
        }

        GenerateOptions options;
        options.max_tokens = 2048;
        options.temperature = 1.0;
        options.top_k = 1;
        options.top_p = 0.0;
        options.alpha_presence = 0.0;
        options.alpha_frequency = 0.0;
        options.stop_tokens = {0};

        std::optional<PrefillAdmissionController::Permit> permit;
        try {
          permit = admission->acquire(
              static_cast<int>(prompts.size()),
              "batch.translate");
        } catch (const PrefillBatchLimitExceeded& error) {
          cb(json_response(prefill_limit_error(error), k400BadRequest));
          return;
        }
        std::vector<std::string> results;
        try {
          auto state = create_request_state(
              engine, state_id, static_cast<int>(prompts.size()));
          results = engine.batch_generate_with_state(prompts, state, options);
        } catch (const std::exception& error) {
          cb(json_response(make_error(error.what()), k400BadRequest));
          return;
        }
        Json::Value resp;
        resp["translations"] = Json::arrayValue;
        for (const auto& text : results) {
          Json::Value item;
          item["detected_source_lang"] = source_lang == "auto" ? "auto" : source_lang;
          item["text"] = text;
          resp["translations"].append(item);
        }
        cb(json_response(std::move(resp)));
      },
      {Post});
}

}  // namespace rwkv7_server::router
