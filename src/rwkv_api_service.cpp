#include "rwkv_api_service.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <drogon/drogon.h>

#include "rwkv_inference_engine.hpp"
#include "rwkv_state_cache.hpp"

namespace rwkv7_server {
namespace {

using namespace drogon;

class RequestRegistry {
 public:
  struct ActiveRequest {
    std::string id;
    std::string endpoint;
    std::string model;
    std::string state_key;
    Json::Int64 created = 0;
    int prompt_tokens = 0;
    int max_tokens = 0;
    std::atomic<int> generated_tokens{0};
    std::atomic<bool> stop_requested{false};
    std::atomic<bool> pause_requested{false};
    std::atomic<int> prefilled_tokens{0};
    std::atomic<double> prefill_progress{0.0};
    std::atomic<double> prefill_speed{0.0};
    std::atomic<double> decode_speed{0.0};
  };

  struct RequestSnapshot {
    std::string id;
    std::string endpoint;
    std::string model;
    std::string state_key;
    Json::Int64 created = 0;
    Json::Int64 finished = 0;
    int prompt_tokens = 0;
    int prefilled_tokens = 0;
    int generated_tokens = 0;
    int max_tokens = 0;
    bool stop_requested = false;
    bool pause_requested = false;
    double prefill_progress = 0.0;
    double prefill_speed = 0.0;
    double decode_speed = 0.0;
  };

  struct PausedRequest {
    std::string id;
    std::string endpoint;
    std::string model;
    Json::Int64 created = 0;
    int generated_tokens = 0;
    int chunk_size = 1;
    GenerateOptions options;
    std::shared_ptr<GenerationState> state;
    std::shared_ptr<DeviceLogits> logits;
  };

  static RequestRegistry& instance() {
    static RequestRegistry registry;
    return registry;
  }

  std::shared_ptr<ActiveRequest> start(
      std::string endpoint,
      std::string model,
      int max_tokens,
      int prompt_tokens = 0,
      std::string state_key = {}) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_) {
      return nullptr;
    }

    auto active = std::make_shared<ActiveRequest>();
    active->id = make_id_locked();
    active->endpoint = std::move(endpoint);
    active->model = std::move(model);
    active->state_key = std::move(state_key);
    active->created = now_seconds();
    active->prompt_tokens = prompt_tokens;
    active->max_tokens = max_tokens;
    active_ = active;
    return active;
  }

  std::shared_ptr<ActiveRequest> active() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_;
  }

  bool stop_active() {
    const auto active = this->active();
    if (!active) {
      return false;
    }
    active->stop_requested.store(true);
    return true;
  }

  std::optional<std::string> pause_active() {
    const auto active = this->active();
    if (!active) {
      return std::nullopt;
    }
    active->pause_requested.store(true);
    return active->id;
  }

  void finish(const std::shared_ptr<ActiveRequest>& active) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_ == active) {
      last_ = snapshot_request(*active);
      last_->finished = now_seconds();
      active_.reset();
    }
  }

  void put_paused(PausedRequest paused) {
    std::lock_guard<std::mutex> lock(mutex_);
    paused_[paused.id] = std::move(paused);
  }

  std::optional<PausedRequest> take_paused(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = paused_.find(id);
    if (it == paused_.end()) {
      return std::nullopt;
    }
    auto paused = std::move(it->second);
    paused_.erase(it);
    return paused;
  }

  Json::Value active_json() const {
    const auto active = this->active();
    if (!active) {
      return Json::Value(Json::nullValue);
    }
    Json::Value out;
    out["id"] = active->id;
    out["endpoint"] = active->endpoint;
    out["model"] = active->model;
    out["created"] = active->created;
    out["prompt_tokens"] = active->prompt_tokens;
    out["prefilled_tokens"] = active->prefilled_tokens.load();
    out["generated_tokens"] = active->generated_tokens.load();
    out["max_tokens"] = active->max_tokens;
    out["stop_requested"] = active->stop_requested.load();
    out["pause_requested"] = active->pause_requested.load();
    out["prefill_progress"] = active->prefill_progress.load();
    out["prefill_speed"] = active->prefill_speed.load();
    out["decode_speed"] = active->decode_speed.load();
    if (!active->state_key.empty()) {
      out["state_key"] = active->state_key;
    }
    return out;
  }

  Json::Value last_json() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!last_.has_value()) {
      return Json::Value(Json::nullValue);
    }
    return snapshot_json(*last_);
  }

  Json::Value paused_json() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Json::Value out = Json::arrayValue;
    for (const auto& [id, paused] : paused_) {
      Json::Value item;
      item["id"] = id;
      item["endpoint"] = paused.endpoint;
      item["model"] = paused.model;
      item["created"] = paused.created;
      item["generated_tokens"] = paused.generated_tokens;
      out.append(item);
    }
    return out;
  }

 private:
  static Json::Int64 now_seconds() {
    return static_cast<Json::Int64>(
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
  }

  std::string make_id_locked() {
    std::ostringstream oss;
    oss << "req-" << now_seconds() << "-" << ++next_id_;
    return oss.str();
  }

  static RequestSnapshot snapshot_request(const ActiveRequest& active) {
    RequestSnapshot snapshot;
    snapshot.id = active.id;
    snapshot.endpoint = active.endpoint;
    snapshot.model = active.model;
    snapshot.state_key = active.state_key;
    snapshot.created = active.created;
    snapshot.prompt_tokens = active.prompt_tokens;
    snapshot.prefilled_tokens = active.prefilled_tokens.load();
    snapshot.generated_tokens = active.generated_tokens.load();
    snapshot.max_tokens = active.max_tokens;
    snapshot.stop_requested = active.stop_requested.load();
    snapshot.pause_requested = active.pause_requested.load();
    snapshot.prefill_progress = active.prefill_progress.load();
    snapshot.prefill_speed = active.prefill_speed.load();
    snapshot.decode_speed = active.decode_speed.load();
    return snapshot;
  }

  static Json::Value snapshot_json(const RequestSnapshot& snapshot) {
    Json::Value out;
    out["id"] = snapshot.id;
    out["endpoint"] = snapshot.endpoint;
    out["model"] = snapshot.model;
    out["created"] = snapshot.created;
    if (snapshot.finished > 0) {
      out["finished"] = snapshot.finished;
    }
    out["prompt_tokens"] = snapshot.prompt_tokens;
    out["prefilled_tokens"] = snapshot.prefilled_tokens;
    out["generated_tokens"] = snapshot.generated_tokens;
    out["max_tokens"] = snapshot.max_tokens;
    out["stop_requested"] = snapshot.stop_requested;
    out["pause_requested"] = snapshot.pause_requested;
    out["prefill_progress"] = snapshot.prefill_progress;
    out["prefill_speed"] = snapshot.prefill_speed;
    out["decode_speed"] = snapshot.decode_speed;
    if (!snapshot.state_key.empty()) {
      out["state_key"] = snapshot.state_key;
    }
    return out;
  }

  mutable std::mutex mutex_;
  std::shared_ptr<ActiveRequest> active_;
  std::optional<RequestSnapshot> last_;
  std::unordered_map<std::string, PausedRequest> paused_;
  Json::UInt64 next_id_ = 0;
};

void record_prefill_metrics(
    const std::shared_ptr<RequestRegistry::ActiveRequest>& active,
    int prefill_tokens,
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end) {
  active->prefilled_tokens.store(prefill_tokens);
  active->prefill_progress.store(1.0);
  const double prefill_seconds = std::chrono::duration<double>(end - begin).count();
  if (prefill_seconds > 0.0) {
    active->prefill_speed.store(prefill_tokens / prefill_seconds);
  }
}

void update_decode_metrics(
    const std::shared_ptr<RequestRegistry::ActiveRequest>& active,
    int generated_tokens,
    double decode_seconds) {
  if (generated_tokens > 0) {
    active->generated_tokens.store(generated_tokens);
  }
  if (decode_seconds > 0.0 && active->generated_tokens.load() > 0) {
    active->decode_speed.store(active->generated_tokens.load() / decode_seconds);
  }
}

void update_decode_metrics(
    const std::shared_ptr<RequestRegistry::ActiveRequest>& active,
    std::chrono::steady_clock::time_point begin) {
  const double decode_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  if (decode_seconds > 0.0 && active->generated_tokens.load() > 0) {
    active->decode_speed.store(active->generated_tokens.load() / decode_seconds);
  }
}

void record_generation_stats(
    const std::shared_ptr<RequestRegistry::ActiveRequest>& active,
    const InferenceEngine::GenerationStats& stats) {
  active->prefilled_tokens.store(stats.prompt_tokens);
  active->prefill_progress.store(1.0);
  if (stats.prefill_seconds > 0.0) {
    active->prefill_speed.store(stats.prompt_tokens / stats.prefill_seconds);
  }
  update_decode_metrics(active, stats.generated_tokens, stats.decode_seconds);
}

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

ThinkType parse_think_type(const Json::Value& body) {
  if (body.isMember("think_type")) {
    const auto value = body.get("think_type", "").asString();
    if (value == "fast") {
      return ThinkType::Fast;
    }
    if (value == "free") {
      return ThinkType::Free;
    }
    if (value == "preferChinese" || value == "prefer_chinese") {
      return ThinkType::PreferChinese;
    }
    if (value == "en") {
      return ThinkType::En;
    }
    if (value == "enShort" || value == "en_short") {
      return ThinkType::EnShort;
    }
    if (value == "enLong" || value == "en_long") {
      return ThinkType::EnLong;
    }
  }
  if (body.isMember("think") ? body.get("think", false).asBool()
                             : body.get("enable_think", false).asBool()) {
    return ThinkType::Free;
  }
  return ThinkType::Fast;
}

bool force_reasoning_for_think_type(ThinkType think_type) {
  return think_type != ThinkType::Fast;
}

GenerateOptions parse_options(const Json::Value& body, GenerateOptions options = {}) {
  options.max_tokens = body.get("max_tokens", options.max_tokens).asInt();
  options.temperature = body.get("temperature", options.temperature).asDouble();
  options.top_k = body.get("top_k", options.top_k).asInt();
  options.top_p = body.get("top_p", options.top_p).asDouble();
  options.alpha_presence = body.get("alpha_presence", options.alpha_presence).asDouble();
  options.alpha_frequency = body.get("alpha_frequency", options.alpha_frequency).asDouble();
  options.alpha_decay = body.get("alpha_decay", options.alpha_decay).asDouble();
  options.force_reasoning = body.get("force_reasoning", options.force_reasoning).asBool();

  const auto& stop_tokens = body["stop_tokens"];
  if (stop_tokens.isArray()) {
    options.stop_tokens.clear();
    for (const auto& item : stop_tokens) {
      options.stop_tokens.push_back(item.asInt64());
    }
  }
  return options;
}

int parse_chunk_size(const Json::Value& body, int fallback) {
  return std::max(1, body.get("chunk_size", fallback).asInt());
}

bool parse_metrics_requested(const Json::Value& body) {
  const auto read_flag = [&body](const char* key) {
    const auto& value = body[key];
    return value.isBool() && value.asBool();
  };
  return read_flag("metrics") || read_flag("report_metrics") || read_flag("include_metrics");
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

bool send_finish_chunk(
    const ResponseStreamPtr& stream,
    const std::string& id,
    const std::string& model,
    int choice_count,
    const std::string& finish_reason = "stop") {
  Json::Value payload;
  if (!id.empty()) {
    payload["id"] = id;
  }
  payload["object"] = "chat.completion.chunk";
  payload["created"] =
      static_cast<Json::Int64>(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
  if (!model.empty()) {
    payload["model"] = model;
  }
  payload["choices"] = Json::arrayValue;
  for (int i = 0; i < choice_count; ++i) {
    Json::Value choice;
    choice["index"] = i;
    choice["delta"] = Json::objectValue;
    choice["finish_reason"] = finish_reason;
    payload["choices"].append(choice);
  }
  Json::StreamWriterBuilder builder;
  builder["emitUTF8"] = true;
  builder["indentation"] = "";
  return stream->send("data: " + Json::writeString(builder, payload) + "\n\n");
}

void start_streaming_task(
    ResponseStreamPtr stream,
    std::string id,
    std::string model,
    int choice_count,
    const std::function<void(const InferenceEngine::StreamCallback&)>& task,
    const std::function<void()>& on_done = {}) {
  std::thread([stream = std::move(stream),
               id = std::move(id),
               model = std::move(model),
               choice_count,
               task,
               on_done]() mutable {
    auto emit = [&stream, &id, &model](int index, const std::string& chunk) -> bool {
      Json::Value payload;
      if (!id.empty()) {
        payload["id"] = id;
      }
      payload["object"] = "chat.completion.chunk";
      payload["created"] =
          static_cast<Json::Int64>(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
      if (!model.empty()) {
        payload["model"] = model;
      }
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
      send_finish_chunk(stream, id, model, std::max(1, choice_count));
    } catch (const std::exception& e) {
      Json::Value err;
      err["error"] = e.what();
      Json::StreamWriterBuilder builder;
      builder["indentation"] = "";
      stream->send("data: " + Json::writeString(builder, err) + "\n\n");
    }
    stream->send("data: [DONE]\n\n");
    stream->close();
    if (on_done) {
      on_done();
    }
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
  return engine.format_openai_prompt(system, dialogue_messages, parse_think_type(body));
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

Json::Value build_status_response(const InferenceEngine& engine) {
  Json::Value resp;
  resp["status"] = "running";
  resp["api_version"] = "1.0";
  resp["engine_version"] = "albatross-1.0.1";
  resp["model"]["id"] = engine.model_name();
  resp["model"]["name"] = engine.model_name();
  resp["model"]["path"] = engine.model()->model_path();
  resp["capabilities"]["chat_messages"] = true;
  resp["capabilities"]["completion"] = true;
  resp["capabilities"]["batch_completion"] = true;
  resp["capabilities"]["stream"] = true;
  resp["capabilities"]["stop"] = true;
  resp["capabilities"]["token_count"] = true;
  resp["capabilities"]["metrics"] = true;
  resp["capabilities"]["session_cache"] = true;
  resp["capabilities"]["pause_resume"] = true;
  resp["capabilities"]["think_type"] = true;
  resp["active_request"] = RequestRegistry::instance().active_json();
  resp["last_request"] = RequestRegistry::instance().last_json();
  resp["paused_requests"] = RequestRegistry::instance().paused_json();
  return resp;
}

Json::Value active_conflict_response() {
  return make_error("Another generation is active");
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
  out["force_reasoning"] = options.force_reasoning;
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
      "[debug_context] endpoint=%s session_id=%s prompt_tokens=%d max_tokens=%d temperature=%.4f top_k=%d top_p=%.4f alpha_presence=%.4f alpha_frequency=%.4f alpha_decay=%.4f force_reasoning=%s stop_tokens=",
      endpoint,
      session_id.empty() ? "<none>" : session_id.c_str(),
      prompt_tokens,
      options.max_tokens,
      options.temperature,
      options.top_k,
      options.top_p,
      options.alpha_presence,
      options.alpha_frequency,
      options.alpha_decay,
      options.force_reasoning ? "true" : "false");
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
           "/v1/server/status",
           "/v1/server/stop",
           "/v1/server/pause",
           "/v1/server/resume",
           "/v1/tokens/count",
           "/v1/chat/completions",
           "/v1/models"}) {
    app.registerHandler(path, handle_options, {Options});
  }

  app.registerHandler(
      "/v1/server/status",
      [&engine](const HttpRequestPtr&, std::function<void(const HttpResponsePtr&)>&& cb) {
        cb(json_response(build_status_response(engine)));
      },
      {Get});

  app.registerHandler(
      "/v1/server/stop",
      [&password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
        auto json = req->getJsonObject();
        Json::Value body = json ? *json : Json::Value(Json::objectValue);
        HttpResponsePtr auth_resp;
        if (!check_password(req, body, password, auth_resp)) {
          cb(auth_resp);
          return;
        }

        const bool stopped = RequestRegistry::instance().stop_active();
        Json::Value resp;
        resp["ok"] = true;
        resp["stopped"] = stopped;
        resp["active_request"] = RequestRegistry::instance().active_json();
        cb(json_response(std::move(resp)));
      },
      {Post});

  app.registerHandler(
      "/v1/server/pause",
      [&password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
        auto json = req->getJsonObject();
        Json::Value body = json ? *json : Json::Value(Json::objectValue);
        HttpResponsePtr auth_resp;
        if (!check_password(req, body, password, auth_resp)) {
          cb(auth_resp);
          return;
        }

        const auto request_id = RequestRegistry::instance().pause_active();
        Json::Value resp;
        resp["ok"] = true;
        resp["paused"] = request_id.has_value();
        if (request_id.has_value()) {
          resp["request_id"] = *request_id;
        }
        resp["active_request"] = RequestRegistry::instance().active_json();
        cb(json_response(std::move(resp)));
      },
      {Post});

  app.registerHandler(
      "/v1/server/resume",
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

        const auto request_id = (*json).get("request_id", (*json).get("session_id", "").asString()).asString();
        if (request_id.empty()) {
          cb(json_response(make_error("Missing request_id"), k400BadRequest));
          return;
        }

        auto paused = RequestRegistry::instance().take_paused(request_id);
        if (!paused.has_value()) {
          cb(json_response(make_error("Paused request not found"), k404NotFound));
          return;
        }
        if (!paused->state || !paused->logits) {
          cb(json_response(make_error("Paused request state is incomplete"), k500InternalServerError));
          return;
        }

        GenerateOptions options = paused->options;
        options.force_reasoning_token_offset = paused->generated_tokens;
        if (!(*json).isMember("max_tokens")) {
          options.max_tokens = std::max(0, paused->options.max_tokens - paused->generated_tokens);
        }
        options = parse_options(*json, options);
        const int chunk_size = parse_chunk_size(*json, paused->chunk_size);
        const std::string model = (*json).get("model", paused->model).asString();

        auto active = RequestRegistry::instance().start(
            "server.resume",
            model,
            options.max_tokens,
            0,
            request_id);
        if (!active) {
          RequestRegistry::instance().put_paused(std::move(*paused));
          cb(json_response(active_conflict_response(), k409Conflict));
          return;
        }
        active->id = request_id;
        active->generated_tokens.store(paused->generated_tokens);

        auto state_ptr = paused->state;
        auto logits_ptr = paused->logits;
        const auto saved_generated_tokens = paused->generated_tokens;
        cb(make_sse_response([&engine, active, state_ptr, logits_ptr, options, chunk_size, request_id, model,
                              saved_generated_tokens](ResponseStreamPtr stream) {
          start_streaming_task(
              std::move(stream),
              request_id,
              model,
              1,
              [&, active, state_ptr, logits_ptr, options, chunk_size, saved_generated_tokens](
                  const InferenceEngine::StreamCallback& emit) {
                const auto stats = engine.generate_from_logits_stream(
                    *state_ptr,
                    *logits_ptr,
                    options,
                    chunk_size,
                    [&](int index, const std::string& chunk) {
                      const int tokens = engine.count_tokens(chunk);
                      active->generated_tokens.fetch_add(tokens);
                      return emit(index, chunk);
                    },
                    [active]() {
                      return active->stop_requested.load() || active->pause_requested.load();
                    });
                if (stats.decode_seconds > 0.0) {
                  active->decode_speed.store(stats.generated_tokens / stats.decode_seconds);
                }
                if (active->pause_requested.load()) {
                  RequestRegistry::PausedRequest paused_again;
                  paused_again.id = active->id;
                  paused_again.endpoint = active->endpoint;
                  paused_again.model = active->model;
                  paused_again.created = active->created;
                  paused_again.generated_tokens = active->generated_tokens.load();
                  paused_again.chunk_size = chunk_size;
                  paused_again.options = options;
                  paused_again.options.max_tokens += saved_generated_tokens;
                  paused_again.state = state_ptr;
                  paused_again.logits = logits_ptr;
                  StateCacheManager::instance().put_state(paused_again.id, *state_ptr);
                  RequestRegistry::instance().put_paused(std::move(paused_again));
                }
              },
              [active]() {
                RequestRegistry::instance().finish(active);
              });
        }));
      },
      {Post});

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
        if (!active) {
          cb(json_response(active_conflict_response(), k409Conflict));
          return;
        }
        if ((*json).get("stream", false).asBool()) {
          cb(make_sse_response([&engine, active, prompts, options, model, metrics_requested,
                                chunk_size = parse_chunk_size(*json, 8)](
                                   ResponseStreamPtr stream) {
            start_streaming_task(
                std::move(stream),
                active->id,
                model,
                static_cast<int>(prompts.size()),
                [&, active, prompts, options, chunk_size, metrics_requested](const InferenceEngine::StreamCallback& emit) {
                  InferenceEngine::StatsCallback on_prefill_complete;
                  if (metrics_requested) {
                    on_prefill_complete = [active](const InferenceEngine::GenerationStats& stats) {
                      record_generation_stats(active, stats);
                    };
                  }
                  const auto stats = engine.batch_generate_stream(
                      prompts,
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
                      on_prefill_complete);
                  if (metrics_requested) {
                    record_generation_stats(active, stats);
                  }
                },
                [active]() {
                  RequestRegistry::instance().finish(active);
                });
          }));
          return;
        }

        Json::Value resp;
        resp["id"] = "rwkv7-fast-batch";
        resp["object"] = "chat.completion";
        resp["model"] = model;
        std::vector<std::string> results(prompts.size());
        const auto stats = engine.batch_generate_stream(
            prompts,
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
            });
        if (metrics_requested) {
          record_generation_stats(active, stats);
        }
        resp["choices"] = build_choices(results);
        RequestRegistry::instance().finish(active);
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
        const int prompt_tokens = engine.count_tokens(prompts.front());
        const auto model = (*json).get("model", engine.model_name()).asString();
        auto active = RequestRegistry::instance().start(
            "state.chat.completions",
            model,
            options.max_tokens,
            prompt_tokens,
            session_id);
        if (!active) {
          cb(json_response(active_conflict_response(), k409Conflict));
          return;
        }
        if ((*json).get("stream", false).asBool()) {
          auto state_ptr = std::make_shared<GenerationState>(std::move(state));
          cb(make_sse_response([&engine, session_id, state_ptr, prompts, options,
                                active, model, chunk_size = parse_chunk_size(*json, 8)](
                                   ResponseStreamPtr stream) {
            start_streaming_task(
                std::move(stream),
                active->id,
                model,
                1,
                [&, active, session_id, state_ptr, prompts, options, chunk_size](
                    const InferenceEngine::StreamCallback& emit) {
                  engine.batch_generate_state_stream(
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
                },
                [active]() {
                  RequestRegistry::instance().finish(active);
                });
          }));
          return;
        }

        auto texts = engine.batch_generate_state(prompts, state, options);
        manager.put_state(session_id, state);
        Json::Value resp;
        resp["id"] = "rwkv7-fast-state";
        resp["object"] = "chat.completion";
        resp["model"] = model;
        resp["choices"] = build_choices(texts);
        RequestRegistry::instance().finish(active);
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
        // std::cout << prompt << std::endl; // Debug Prompt
        auto options = parse_options(*json);
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
        if (!active) {
          cb(json_response(active_conflict_response(), k409Conflict));
          return;
        }

        if ((*json).get("stream", false).asBool()) {
          cb(make_sse_response([&engine, prompt, options, active, model,
                                chunk_size = parse_chunk_size(*json, 2)](
                                   ResponseStreamPtr stream) {
            auto state_ptr = std::make_shared<GenerationState>(engine.model()->create_state(1));
            auto logits_ptr = std::make_shared<DeviceLogits>();
            start_streaming_task(
                std::move(stream),
                active->id,
                model,
                1,
                [&, active, state_ptr, logits_ptr, prompt, options, chunk_size](
                    const InferenceEngine::StreamCallback& emit) {
                  const auto prefill_begin = std::chrono::steady_clock::now();
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
                    paused.state = state_ptr;
                    paused.logits = logits_ptr;
                    StateCacheManager::instance().put_state(paused.id, *state_ptr);
                    RequestRegistry::instance().put_paused(std::move(paused));
                  }
                },
                [active]() {
                  RequestRegistry::instance().finish(active);
                });
          }));
          return;
        }

        auto state_ptr = std::make_shared<GenerationState>(engine.model()->create_state(1));
        auto logits_ptr = std::make_shared<DeviceLogits>();
        std::string completion;

        const auto prefill_begin = std::chrono::steady_clock::now();
        const int prefill_tokens = engine.prefill_prompt(prompt, *state_ptr, *logits_ptr);
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
          paused.state = state_ptr;
          paused.logits = logits_ptr;
          StateCacheManager::instance().put_state(paused.id, *state_ptr);
          RequestRegistry::instance().put_paused(std::move(paused));
        }
        RequestRegistry::instance().finish(active);

        cb(json_response(build_openai_response(engine, *json, prompt, completion)));
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
