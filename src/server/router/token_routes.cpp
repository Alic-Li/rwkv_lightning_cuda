#include "rwkv/server/router/route_helpers.hpp"
#include "rwkv/server/router/routes.hpp"
namespace rwkv7_server::router {

using namespace drogon;


void register_token_routes(
    HttpAppFramework& app, InferenceEngine& engine,
    const std::optional<std::string>& password) {
  app.registerHandler(
      "/v1/tokens/count",
      [&engine, password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
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

        std::string text;
        if ((*json).isMember("text")) {
          text = (*json).get("text", "").asString();
        } else if ((*json).isMember("messages")) {
          text = format_openai_prompt(*json, engine);
        } else if ((*json).isMember("contents")) {
          const auto prompts = parse_contents(*json);
          for (const auto& prompt : prompts) {
            text += prompt;
          }
        } else {
          cb(json_response(make_error("Missing text or messages"), k400BadRequest));
          return;
        }

        Json::Value resp;
        resp["tokens"] = engine.count_tokens(text);
        cb(json_response(std::move(resp)));
      },
      {Post});
}

}  // namespace rwkv7_server::router
