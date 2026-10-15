#include "rwkv/server/router/route_helpers.hpp"
#include "rwkv/server/router/routes.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>

namespace rwkv7_server::router {

using namespace drogon;

void register_adapter_routes(drogon::HttpAppFramework &app,
                             const std::optional<std::string> &password) {
  app.registerHandler(
      "/v1/adapters",
      [password](const HttpRequestPtr &req,
                 std::function<void(const HttpResponsePtr &)> &&cb) {
        if (req->method() == Options) {
          auto resp = HttpResponse::newHttpResponse();
          resp->setStatusCode(k204NoContent);
          cb(resp);
          return;
        }
        auto body = req->getJsonObject();
        Json::Value json = body ? *body : Json::Value(Json::objectValue);
        HttpResponsePtr auth;
        if (!check_password(req, json, password, auth)) {
          cb(auth);
          return;
        }
        try {
          auto &cache = rwkv7_miss::adapter_cache();
          Json::Value out;
          if (req->method() == Post) {
            if (req->contentType() == CT_MULTIPART_FORM_DATA) {
              MultiPartParser parser;
              if (parser.parse(req) != 0 || parser.getFiles().empty() ||
                  parser.getFiles().size() > 2)
                throw std::runtime_error("upload requires one PTH and optional legacy checkpoint.json");
              auto id = parser.getParameter<std::string>("adapter_id");
              if (id.empty()) throw std::runtime_error("missing adapter_id");
              auto unique = rwkv7_miss::new_runtime_identity();
              std::replace(unique.begin(), unique.end(), ':', '_');
              const auto dir = std::filesystem::temp_directory_path() / ("rwkv-miss-" + unique);
              if (!std::filesystem::create_directory(dir))
                throw std::runtime_error("cannot create adapter upload staging directory");
              struct Cleanup {
                std::filesystem::path path;
                ~Cleanup() { std::error_code ec; std::filesystem::remove_all(path, ec); }
              } cleanup{dir};
              bool have_pth = false, have_meta = false;
              for (const auto &file : parser.getFiles()) {
                const bool meta = file.getFileName() == "checkpoint.json";
                if ((meta && have_meta) || (!meta && have_pth))
                  throw std::runtime_error("duplicate upload file");
                if (meta && file.fileLength() > (8ULL << 20))
                  throw std::runtime_error("checkpoint metadata too large");
                (meta ? have_meta : have_pth) = true;
                std::ofstream f(dir / (meta ? "checkpoint.json" : "training.pth"), std::ios::binary);
                f.write(file.fileData(), file.fileLength());
                f.close();
                if (!f) throw std::runtime_error("adapter upload write failed");
              }
              if (!have_pth) throw std::runtime_error("missing adapter PTH");
              out["version"] = cache.register_adapter(id, (dir / "training.pth").string());
              out["adapter_id"] = id;
            } else {
              out["version"] = cache.register_adapter(
                  json["adapter_id"].asString(), json["path"].asString());
              out["adapter_id"] = json["adapter_id"];
            }
          } else if (req->method() == Delete) {
            cache.erase(json["adapter_id"].asString(),
                        json.get("adapter_version", "").asString());
            cache.trim();
            out["deleted"] = true;
          } else {
            out["data"] = cache.list();
            auto stats = cache.stats();
            out["ram_hits"] = Json::UInt64(stats.ram_hits);
            out["gpu_hits"] = Json::UInt64(stats.gpu_hits);
            out["gpu_misses"] = Json::UInt64(stats.gpu_misses);
            out["uploads"] = Json::UInt64(stats.uploads);
            out["h2d_ms"] = stats.h2d_ms;
            out["ram_bytes"] = Json::UInt64(stats.ram_bytes);
            out["gpu_bytes"] = Json::UInt64(stats.gpu_bytes);
            out["gpu_peak_bytes"] = Json::UInt64(stats.gpu_peak_bytes);
          }
          cb(json_response(out));
        } catch (const std::exception &e) {
          cb(json_response(make_error(e.what()), k400BadRequest));
        }
      },
      {Get, Post, Delete, Options});
}

}  // namespace rwkv7_server::router
