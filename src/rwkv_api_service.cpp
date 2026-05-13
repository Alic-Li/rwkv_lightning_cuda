#include "rwkv_api_service.hpp"

#include <chrono>
#include <cctype>
#include <cstdio>
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
  options.max_tokens = body.get("max_tokens", options.max_tokens).asInt();
  options.temperature = body.get("temperature", options.temperature).asDouble();
  options.top_k = body.get("top_k", options.top_k).asInt();
  options.top_p = body.get("top_p", options.top_p).asDouble();
  options.alpha_presence = body.get("alpha_presence", options.alpha_presence).asDouble();
  options.alpha_frequency = body.get("alpha_frequency", options.alpha_frequency).asDouble();
  options.alpha_decay = body.get("alpha_decay", options.alpha_decay).asDouble();

  const auto& stop_tokens = body["stop_tokens"];
  if (stop_tokens.isArray()) {
    options.stop_tokens.clear();
    for (const auto& item : stop_tokens) {
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
  std::vector<std::pair<std::string, std::string>> dialogue_messages;
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
      dialogue_messages.emplace_back(role.empty() ? "User" : role, content);
    }
  }

  const auto contents = parse_contents(body);
  if (!contents.empty()) {
    dialogue_messages.emplace_back("User", contents.front());
  }
  if (dialogue_messages.empty()) {
    dialogue_messages.emplace_back("User", "");
  }

  std::string system;
  for (const auto& part : system_parts) {
    if (!part.empty()) {
      if (!system.empty()) {
        system += '\n';
      }
      system += part;
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

Json::Value build_models_response(const InferenceEngine& engine) {
  Json::Value resp;
  resp["object"] = "list";
  resp["data"] = Json::arrayValue;

  Json::Value model;
  model["id"] = engine.model_name();
  model["object"] = "model";
  model["created"] =
      static_cast<Json::Int64>(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
  model["owned_by"] = "rwkv_lighting_cuda";
  resp["data"].append(model);
  return resp;
}

Json::Value build_options_debug(const InferenceEngine& engine, const GenerateOptions& options) {
  Json::Value out;
  out["max_tokens"] = options.max_tokens;
  out["temperature"] = options.temperature;
  out["top_k"] = options.top_k;
  out["top_p"] = options.top_p;
  out["alpha_presence"] = options.alpha_presence;
  out["alpha_frequency"] = options.alpha_frequency;
  out["alpha_decay"] = options.alpha_decay;
  out["stop_tokens"] = Json::arrayValue;
  out["stop_texts"] = Json::arrayValue;
  for (const auto token : options.stop_tokens) {
    out["stop_tokens"].append(static_cast<Json::Int64>(token));
    out["stop_texts"].append(engine.tokenizer()->decode(static_cast<int>(token)));
  }
  return out;
}

void print_chat_context_debug(
    const char* endpoint,
    const std::string& session_id,
    const std::string& prompt,
    int prompt_tokens,
    const GenerateOptions& options) {
  std::printf(
      "[debug_context] endpoint=%s session_id=%s prompt_tokens=%d max_tokens=%d temperature=%.4f top_k=%d top_p=%.4f alpha_presence=%.4f alpha_frequency=%.4f alpha_decay=%.4f stop_tokens=",
      endpoint,
      session_id.empty() ? "<none>" : session_id.c_str(),
      prompt_tokens,
      options.max_tokens,
      options.temperature,
      options.top_k,
      options.top_p,
      options.alpha_presence,
      options.alpha_frequency,
      options.alpha_decay);
  if (options.stop_tokens.empty()) {
    std::printf("[]\n");
  } else {
    for (size_t i = 0; i < options.stop_tokens.size(); ++i) {
      std::printf("%s%lld", i == 0 ? "[" : ",", static_cast<long long>(options.stop_tokens[i]));
    }
    std::printf("]\n");
  }
  std::printf("[debug_context] prompt_begin\n%s\n[debug_context] prompt_end\n", prompt.c_str());
  std::fflush(stdout);
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
           "/v1/chat/completions",
           "/v1/models"}) {
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
        const auto options = parse_options(*json);

        if ((*json).get("stream", false).asBool()) {
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

        const auto texts = engine.batch_generate({prompt}, options);

        cb(json_response(build_openai_response(engine, *json, prompt, texts.empty() ? std::string{} : texts.front())));
      },
      {Post});

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

}  // namespace rwkv7_server
