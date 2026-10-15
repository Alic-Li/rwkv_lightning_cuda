#include "rwkv/server/router/route_helpers.hpp"
#include "rwkv/server/router/routes.hpp"

#include "rwkv/server/rwkv_uploaded_state_store.hpp"

namespace rwkv7_server::router {

using namespace drogon;

namespace {

Json::Value uploaded_state_json(const UploadedStateInfo& info) {
  Json::Value item;
  item["state_id"] = info.state_id;
  item["filename"] = info.filename;
  item["original_filename"] = info.original_filename;
  item["created_ms"] = static_cast<Json::Int64>(info.created_ms);
  item["uploaded_at"] = info.uploaded_at;
  item["layers"] = info.layers;
  item["heads"] = info.heads;
  item["head_size"] = info.head_size;
  item["size_bytes"] = static_cast<Json::UInt64>(info.size_bytes);
  item["tensor_count"] = info.tensor_count;
  item["created"] = static_cast<Json::Int64>(info.created);
  return item;
}

}  // namespace

void register_uploaded_state_routes(
    HttpAppFramework& app,
    const std::optional<std::string>& password) {
  const auto options_handler = [](const HttpRequestPtr&, std::function<void(const HttpResponsePtr&)>&& cb) {
    auto resp = HttpResponse::newHttpResponse();
    resp->setStatusCode(k204NoContent);
    cb(resp);
  };
  for (const auto& path : {"/v1/state/upload", "/v1/state/delete", "/v1/state/list"}) {
    app.registerHandler(path, options_handler, {Options});
  }

  app.registerHandler(
      "/v1/state/upload",
      [password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
        Json::Value auth_body = Json::objectValue;
        HttpResponsePtr auth_resp;
        if (!check_password(req, auth_body, password, auth_resp)) {
          cb(auth_resp);
          return;
        }
        MultiPartParser parser;
        if (parser.parse(req) != 0) {
          cb(json_response(make_error("Invalid multipart/form-data request"), k400BadRequest));
          return;
        }
        const auto& files = parser.getFiles();
        if (files.size() != 1) {
          cb(json_response(make_error("Request must contain exactly one state file"), k400BadRequest));
          return;
        }
        try {
          const auto& file = files.front();
          const auto info = UploadedStateStore::instance().upload(
              file.getFileName(), file.fileData(), file.fileLength(),
              req->getHeader("x-rwkv-state-upload-uuid"));
          Json::Value resp = uploaded_state_json(info);
          resp["object"] = "rwkv.state";
          cb(json_response(std::move(resp)));
        } catch (const std::exception& error) {
          cb(json_response(make_error(error.what()), k400BadRequest));
        }
      },
      {Post});

  app.registerHandler(
      "/v1/state/list",
      [password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
        const auto json = req->getJsonObject();
        Json::Value body = json ? *json : Json::Value(Json::objectValue);
        HttpResponsePtr auth_resp;
        if (!check_password(req, body, password, auth_resp)) {
          cb(auth_resp);
          return;
        }
        const auto states = UploadedStateStore::instance().list();
        Json::Value resp;
        resp["object"] = "list";
        resp["data"] = Json::arrayValue;
        for (const auto& state : states) {
          resp["data"].append(uploaded_state_json(state));
        }
        cb(json_response(std::move(resp)));
      },
      {Get, Post});

  app.registerHandler(
      "/v1/state/delete",
      [password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
        const auto json = req->getJsonObject();
        Json::Value body = json ? *json : Json::Value(Json::objectValue);
        HttpResponsePtr auth_resp;
        if (!check_password(req, body, password, auth_resp)) {
          cb(auth_resp);
          return;
        }
        try {
          std::string state_id = resolve_identifier(
              {{"request body", state_id_from_body(body)},
               {std::string(kStateIdHeader) + " header",
                request_header_value(req, kStateIdHeader)},
               {"query string", req->getParameter("state_id")}},
              "state_id");
          if (state_id.empty()) {
            cb(json_response(make_error("Missing state_id"), k400BadRequest));
            return;
          }
          const bool deleted = UploadedStateStore::instance().erase(state_id);
          Json::Value resp;
          resp["state_id"] = state_id;
          resp["deleted"] = deleted;
          cb(json_response(std::move(resp), deleted ? k200OK : k404NotFound));
        } catch (const std::exception& error) {
          cb(json_response(make_error(error.what()), k400BadRequest));
        }
      },
      {Delete, Post});
}

}  // namespace rwkv7_server::router
