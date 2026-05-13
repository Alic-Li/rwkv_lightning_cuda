#include "rwkv_api_service.hpp"

#include <chrono>
#include <cctype>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <drogon/drogon.h>

#include "rwkv_inference_engine.hpp"
#include "rwkv_state_cache.hpp"

namespace rwkv7_server {
namespace {

using namespace drogon;

Json::Value make_error(const std::string& message) {
  Json::Value out;
  out["error"] = message;
  return out;
}

HttpResponsePtr json_response(Json::Value payload, HttpStatusCode code = k200OK) {
  Json::StreamWriterBuilder builder;
  builder["emitUTF8"] = true;
  builder["indentation"] = "";
  auto resp = HttpResponse::newHttpResponse();
  resp->setContentTypeCode(CT_APPLICATION_JSON);
  resp->setBody(Json::writeString(builder, payload));
  resp->setStatusCode(code);
  return resp;
}

std::optional<std::string> bearer_token(const HttpRequestPtr& req) {
  auto auth = req->getHeader("authorization");
  if (auth.empty()) {
    auth = req->getHeader("Authorization");
  }
  if (auth.rfind("Bearer ", 0) != 0) {
    return std::nullopt;
  }
  return auth.substr(7);
}

bool check_password(
    const HttpRequestPtr& req,
    const Json::Value& body,
    const std::optional<std::string>& password,
    HttpResponsePtr& out) {
  if (!password.has_value()) {
    return true;
  }
  const auto token = bearer_token(req);
  if (body.get("password", "").asString() == *password ||
      (token.has_value() && *token == *password)) {
    return true;
  }
  out = json_response(make_error("Unauthorized: invalid or missing password"), k401Unauthorized);
  return false;
}

GenerateOptions parse_options(const Json::Value& body) {
  GenerateOptions options;
  options.max_tokens = body.get("max_tokens", 1024).asInt();
  options.temperature = body.get("temperature", 1.0).asDouble();
  options.top_k = body.get("top_k", 20).asInt();
  options.top_p = body.get("top_p", 0.6).asDouble();
  options.alpha_presence = body.get("alpha_presence", 1.0).asDouble();
  options.alpha_frequency = body.get("alpha_frequency", 0.1).asDouble();
  options.alpha_decay = body.get("alpha_decay", 0.996).asDouble();
  options.pad_zero = body.get("pad_zero", true).asBool();
  if (body.isMember("stop_tokens") && body["stop_tokens"].isArray()) {
    options.stop_tokens.clear();
    for (const auto& item : body["stop_tokens"]) {
      options.stop_tokens.push_back(item.asInt64());
    }
  }
  return options;
}

std::string normalize_content(const Json::Value& content) {
  if (content.isString()) {
    return content.asString();
  }
  if (content.isArray()) {
    std::string text;
    for (const auto& item : content) {
      if (item.isObject() && item.get("type", "").asString() == "text") {
        text += item.get("text", "").asString();
      } else if (item.isString()) {
        text += item.asString();
      }
    }
    return text;
  }
  if (content.isNull()) {
    return "";
  }
  return content.asString();
}

std::vector<std::string> parse_contents(const Json::Value& body) {
  std::vector<std::string> prompts;
  if (body.isMember("contents") && body["contents"].isArray()) {
    for (const auto& item : body["contents"]) {
      prompts.push_back(item.asString());
    }
  }
  return prompts;
}

std::string create_translation_prompt(
    const std::string& source_lang,
    const std::string& target_lang,
    const std::string& text) {
  return source_lang + ": " + text + "\n\n" + target_lang + ":";
}

Json::Value build_choices(const std::vector<std::string>& texts) {
  Json::Value choices = Json::arrayValue;
  for (size_t i = 0; i < texts.size(); ++i) {
    Json::Value choice;
    choice["index"] = static_cast<int>(i);
    choice["message"]["role"] = "assistant";
    choice["message"]["content"] = texts[i];
    choice["finish_reason"] = "stop";
    choices.append(choice);
  }
  return choices;
}

HttpResponsePtr make_sse_response(const std::function<void(ResponseStreamPtr)>& producer) {
  auto resp = HttpResponse::newAsyncStreamResponse(
      [producer](ResponseStreamPtr stream) { producer(std::move(stream)); },
      true);
  resp->setContentTypeString("text/event-stream; charset=utf-8");
  resp->addHeader("Cache-Control", "no-cache");
  resp->addHeader("X-Accel-Buffering", "no");
  return resp;
}

void start_streaming_task(
    ResponseStreamPtr stream,
    const std::function<void(const InferenceEngine::StreamCallback&)>& task) {
  std::thread([stream = std::move(stream), task]() mutable {
    auto emit = [&stream](int index, const std::string& chunk) -> bool {
      Json::Value payload;
      payload["object"] = "chat.completion.chunk";
      payload["choices"] = Json::arrayValue;
      Json::Value choice;
      choice["index"] = index;
      choice["delta"]["content"] = chunk;
      payload["choices"].append(choice);
      Json::StreamWriterBuilder builder;
      builder["emitUTF8"] = true;
      builder["indentation"] = "";
      return stream->send("data: " + Json::writeString(builder, payload) + "\n\n");
    };
    try {
      task(emit);
    } catch (const std::exception& e) {
      Json::Value err;
      err["error"] = e.what();
      Json::StreamWriterBuilder builder;
      builder["indentation"] = "";
      stream->send("data: " + Json::writeString(builder, err) + "\n\n");
    }
    stream->send("data: [DONE]\n\n");
    stream->close();
  }).detach();
}

std::string format_openai_prompt(const Json::Value& body, const InferenceEngine& engine) {
  std::string current_prompt;
  const auto contents = parse_contents(body);
  if (!contents.empty()) {
    current_prompt = contents.front();
  }

  std::vector<std::pair<std::string, std::string>> history_messages;
  std::vector<std::string> system_parts;
  const auto system_field = body.get("system", "").asString();
  if (!system_field.empty()) {
    system_parts.push_back(system_field);
  }

  if (body.isMember("messages") && body["messages"].isArray()) {
    for (const auto& msg : body["messages"]) {
      auto role = msg.get("role", "user").asString();
      auto content = normalize_content(msg["content"]);
      if (content.empty()) {
        continue;
      }
      if (role == "system") {
        system_parts.push_back(content);
        continue;
      }
      if (!role.empty()) {
        role[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(role[0])));
      }
      history_messages.emplace_back(role.empty() ? "User" : role, content);
    }
  }

  if (current_prompt.empty() && !history_messages.empty()) {
    current_prompt = history_messages.back().second;
  }
  if (!current_prompt.empty() && !history_messages.empty() &&
      history_messages.back().second == current_prompt) {
    history_messages.pop_back();
  }

  std::vector<std::pair<std::string, std::string>> messages;
  for (const auto& system : system_parts) {
    if (!system.empty()) {
      messages.emplace_back("System", system);
    }
  }
  for (const auto& item : history_messages) {
    messages.push_back(item);
  }
  if (!current_prompt.empty()) {
    messages.emplace_back("User", current_prompt);
  }
  if (messages.empty()) {
    messages.emplace_back("User", "");
  }

  std::string system;
  std::vector<std::pair<std::string, std::string>> dialogue_messages;
  for (const auto& [role, content] : messages) {
    if (role == "System") {
      if (!system.empty()) {
        system += "\n\n";
      }
      system += content;
    } else {
      dialogue_messages.emplace_back(role, content);
    }
  }
  return engine.format_openai_prompt(system, dialogue_messages, body.get("enable_think", false).asBool());
}

Json::Value build_openai_response(
    const InferenceEngine& engine,
    const Json::Value& body,
    const std::string& prompt,
    const std::string& completion) {
  Json::Value resp;
  resp["id"] = "chatcmpl-rwkv-fast";
  resp["object"] = "chat.completion";
  resp["created"] =
      static_cast<Json::Int64>(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
  resp["model"] = body.get("model", engine.model_name()).asString();
  resp["choices"] = Json::arrayValue;
  Json::Value choice;
  choice["index"] = 0;
  choice["message"]["role"] = "assistant";
  choice["message"]["content"] = completion;
  choice["finish_reason"] = "stop";
  resp["choices"].append(choice);
  resp["usage"]["prompt_tokens"] = engine.count_tokens(prompt);
  resp["usage"]["completion_tokens"] = engine.count_tokens(completion);
  resp["usage"]["total_tokens"] =
      resp["usage"]["prompt_tokens"].asInt() + resp["usage"]["completion_tokens"].asInt();
  return resp;
}

}  // namespace

void register_api_routes(
    InferenceEngine& engine,
    const std::optional<std::string>& password) {
  auto& app = drogon::app();
  app.registerPostHandlingAdvice([](const HttpRequestPtr&, const HttpResponsePtr& resp) {
    resp->addHeader("Access-Control-Allow-Origin", "*");
    resp->addHeader("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
    resp->addHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  });

  auto handle_options = [](const HttpRequestPtr&, std::function<void(const HttpResponsePtr&)>&& cb) {
    auto resp = HttpResponse::newHttpResponse();
    resp->setStatusCode(k204NoContent);
    cb(resp);
  };

  for (const auto& path : {
           "/v1/batch/completions",
           "/translate/v1/batch-translate",
           "/state/chat/completions",
           "/state/status",
           "/state/delete",
           "/v1/chat/completions"}) {
    app.registerHandler(path, handle_options, {Options});
  }

  app.registerHandler(
      "/v1/batch/completions",
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

        const auto prompts = parse_contents(*json);
        if (prompts.empty()) {
          cb(json_response(make_error("Empty prompts list"), k400BadRequest));
          return;
        }
        const auto options = parse_options(*json);
        if ((*json).get("stream", false).asBool()) {
          cb(make_sse_response([&engine, prompts, options, chunk_size = (*json).get("chunk_size", 8).asInt()](
                                   ResponseStreamPtr stream) {
            start_streaming_task(
                std::move(stream),
                [&, prompts, options, chunk_size](const InferenceEngine::StreamCallback& emit) {
                  engine.batch_generate_stream(prompts, options, chunk_size, emit);
                });
          }));
          return;
        }

        Json::Value resp;
        resp["id"] = "rwkv7-fast-batch";
        resp["object"] = "chat.completion";
        resp["model"] = (*json).get("model", engine.model_name()).asString();
        resp["choices"] = build_choices(engine.batch_generate(prompts, options));
        cb(json_response(std::move(resp)));
      },
      {Post});

  app.registerHandler(
      "/translate/v1/batch-translate",
      [&engine](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
        auto json = req->getJsonObject();
        if (!json) {
          cb(json_response(make_error("Invalid JSON"), k400BadRequest));
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

        const auto results = engine.batch_generate(prompts, options);
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

  app.registerHandler(
      "/state/chat/completions",
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

        const auto session_id = (*json).get("session_id", "").asString();
        const auto prompts = parse_contents(*json);
        if (session_id.empty()) {
          cb(json_response(make_error("Missing session_id"), k400BadRequest));
          return;
        }
        if (prompts.size() != 1) {
          cb(json_response(make_error("Request must contain exactly one prompt"), k400BadRequest));
          return;
        }

        auto& manager = StateCacheManager::instance();
        auto state = manager.get_state(session_id).value_or(engine.model()->create_state(1));
        const auto options = parse_options(*json);
        if ((*json).get("stream", false).asBool()) {
          auto state_ptr = std::make_shared<GenerationState>(std::move(state));
          cb(make_sse_response([&engine, &manager, session_id, state_ptr, prompts, options,
                                chunk_size = (*json).get("chunk_size", 8).asInt()](
                                   ResponseStreamPtr stream) {
            start_streaming_task(
                std::move(stream),
                [&, session_id, state_ptr, prompts, options, chunk_size](
                    const InferenceEngine::StreamCallback& emit) {
                  engine.batch_generate_state_stream(prompts, *state_ptr, options, chunk_size, emit);
                  manager.put_state(session_id, *state_ptr);
                });
          }));
          return;
        }

        auto texts = engine.batch_generate_state(prompts, state, options);
        manager.put_state(session_id, state);
        Json::Value resp;
        resp["id"] = "rwkv7-fast-state";
        resp["object"] = "chat.completion";
        resp["model"] = (*json).get("model", engine.model_name()).asString();
        resp["choices"] = build_choices(texts);
        cb(json_response(std::move(resp)));
      },
      {Post});

  app.registerHandler(
      "/state/status",
      [&password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
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
      [&password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
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

        const auto session_id = (*json).get("session_id", "").asString();
        if (session_id.empty()) {
          cb(json_response(make_error("Missing session_id"), k400BadRequest));
          return;
        }
        const bool ok = StateCacheManager::instance().delete_state_from_any_level(session_id);
        Json::Value resp;
        resp["status"] = ok ? "success" : "not_found";
        resp["message"] = ok ? ("Session " + session_id + " deleted successfully")
                             : ("Session " + session_id + " not found");
        cb(json_response(std::move(resp), ok ? k200OK : k404NotFound));
      },
      {Post});

  app.registerHandler(
      "/v1/chat/completions",
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

        const auto prompt = format_openai_prompt(*json, engine);
        const auto session_id = (*json).get("session_id", "").asString();
        const auto options = parse_options(*json);

        if ((*json).get("stream", false).asBool()) {
          if (!session_id.empty()) {
            auto& manager = StateCacheManager::instance();
            auto state = manager.get_state(session_id).value_or(engine.model()->create_state(1));
            auto state_ptr = std::make_shared<GenerationState>(std::move(state));
            cb(make_sse_response([&engine, &manager, session_id, state_ptr, prompt, options,
                                  chunk_size = (*json).get("chunk_size", 8).asInt()](
                                     ResponseStreamPtr stream) {
              start_streaming_task(
                  std::move(stream),
                  [&, session_id, state_ptr, prompt, options, chunk_size](
                      const InferenceEngine::StreamCallback& emit) {
                    engine.batch_generate_state_stream({prompt}, *state_ptr, options, chunk_size, emit);
                    manager.put_state(session_id, *state_ptr);
                  });
            }));
            return;
          }

          cb(make_sse_response([&engine, prompt, options, chunk_size = (*json).get("chunk_size", 8).asInt()](
                                   ResponseStreamPtr stream) {
            start_streaming_task(
                std::move(stream),
                [&, prompt, options, chunk_size](const InferenceEngine::StreamCallback& emit) {
                  engine.batch_generate_stream({prompt}, options, chunk_size, emit);
                });
          }));
          return;
        }

        std::vector<std::string> texts;
        if (!session_id.empty()) {
          auto& manager = StateCacheManager::instance();
          auto state = manager.get_state(session_id).value_or(engine.model()->create_state(1));
          texts = engine.batch_generate_state({prompt}, state, options);
          manager.put_state(session_id, state);
        } else {
          texts = engine.batch_generate({prompt}, options);
        }

        cb(json_response(build_openai_response(engine, *json, prompt, texts.empty() ? std::string{} : texts.front())));
      },
      {Post});
}

}  // namespace rwkv7_server
