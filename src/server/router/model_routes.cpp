#include "rwkv/server/router/route_helpers.hpp"
#include "rwkv/server/router/routes.hpp"
namespace rwkv7_server::router {

using namespace drogon;

Json::Value build_models_response(const InferenceEngine& engine) {
  Json::Value resp;
  resp["object"] = "list";
  resp["data"] = Json::arrayValue;
  Json::Value model;
  model["id"] = engine.model_name();
  model["object"] = "model";
  model["created"] = static_cast<Json::Int64>(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
  model["owned_by"] = "rwkv_lighting_cuda";
  resp["data"].append(model);
  return resp;
}

void register_model_routes(
    HttpAppFramework& app, InferenceEngine& engine,
    const std::optional<std::string>& password) {
  app.registerHandler(
      "/v1/models",
      [&engine, password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
        Json::Value body = Json::objectValue;
        HttpResponsePtr auth_resp;
        if (!check_password(req, body, password, auth_resp)) {
          cb(auth_resp);
          return;
        }
        cb(json_response(build_models_response(engine)));
      },
      {Get});
}

}  // namespace rwkv7_server::router
