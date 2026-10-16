#include "rwkv/server/router/route_helpers.hpp"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <stdexcept>
#include <thread>

#include "rwkv/server/rwkv_uploaded_state_store.hpp"

namespace rwkv7_server::router {

using namespace drogon;

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

ThinkType parse_think_type(const Json::Value& body) {
  if (body.isMember("think_type")) {
    const auto value = body.get("think_type", "").asString();
    if (value == "none" || value == "off" || value == "disabled") {
      return ThinkType::None;
    }
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
  return think_type != ThinkType::None && think_type != ThinkType::Fast;
}

bool has_explicit_think_type(const Json::Value& body) {
  return body.isMember("think_type") || body.isMember("think") ||
         body.isMember("enable_think");
}

GenerateOptions parse_options(const Json::Value& body, GenerateOptions options) {
  if (body.isMember("stop") && body.isMember("stop_tokens"))
    throw std::invalid_argument("stop and stop_tokens are mutually exclusive");
  if (body.isMember("stop")) {
    if (!body["stop"].isArray())
      throw std::invalid_argument("stop must be an array of non-empty strings");
    options.stop.emplace();
    options.stop_tokens.clear();
    for (const auto& item : body["stop"]) {
      if (!item.isString() || item.asString().empty())
        throw std::invalid_argument("stop must be an array of non-empty strings");
      options.stop->push_back(item.asString());
    }
  }
  if (body.isMember("adapter_scale") && !body["adapter_scale"].isNumeric())
    throw std::invalid_argument("adapter_scale must be numeric");
  if (body.isMember("adapter_id")) {
    const auto id = body["adapter_id"].asString();
    options.adapter_id = id;
    if (id.empty())
      options.adapter.reset();
    else {
      float scale = body.get("adapter_scale", 0).asFloat();
      options.adapter = rwkv7_miss::adapter_cache().resolve(
          id, body.get("adapter_version", "").asString(),
          body.isMember("adapter_scale") ? &scale : nullptr);
    }
  } else if (body.isMember("adapter_version") ||
             body.isMember("adapter_scale")) {
    throw std::invalid_argument(
        "adapter_id is required with adapter_version/adapter_scale");
  }
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
    options.stop.reset();
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

GenerationState create_request_state(
    const InferenceEngine& engine,
    const std::string& state_id,
    int batch_size) {
  if (state_id.empty()) {
    return engine.model()->create_state(batch_size);
  }
  auto uploaded = UploadedStateStore::instance().acquire(state_id);
  if (!uploaded.has_value()) {
    throw std::runtime_error("uploaded state not found: " + state_id);
  }
  return engine.model()->load_state_from_pth(uploaded->path.string(), batch_size);
}

std::string create_translation_prompt(
    const std::string& source_lang,
    const std::string& target_lang,
    const std::string& text) {
  return source_lang + ": " + text + "\n\n" + target_lang + ":";
}

Json::Value build_choices(const std::vector<std::string>& texts,
                          const InferenceEngine::GenerationStats& stats) {
  Json::Value choices = Json::arrayValue;
  for (size_t i = 0; i < texts.size(); ++i) {
    Json::Value choice;
    choice["index"] = static_cast<int>(i);
    choice["message"]["role"] = "assistant";
    choice["message"]["content"] = texts[i];
    choice["finish_reason"] = stats.finish_reasons.at(i);
    if (i < stats.matched_stops.size() && !stats.matched_stops[i].empty())
      choice["matched_stop"] = stats.matched_stops[i];
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
    const InferenceEngine::GenerationStats& stats,
    const std::vector<bool>& finished) {
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
  for (size_t i = 0; i < stats.finish_reasons.size(); ++i) {
    if (finished[i]) continue;
    Json::Value choice;
    choice["index"] = static_cast<int>(i);
    choice["delta"] = Json::objectValue;
    choice["finish_reason"] = stats.finish_reasons[i];
    if (i < stats.matched_stops.size() && !stats.matched_stops[i].empty())
      choice["matched_stop"] = stats.matched_stops[i];
    payload["choices"].append(choice);
  }
  if (payload["choices"].empty()) return true;
  Json::StreamWriterBuilder builder;
  builder["emitUTF8"] = true;
  builder["indentation"] = "";
  return stream->send("data: " + Json::writeString(builder, payload) + "\n\n");
}

InferenceEngine::GenerationStats cancelled_generation(int choices) {
  InferenceEngine::GenerationStats stats;
  stats.stopped = true;
  stats.finish_reasons.assign(static_cast<size_t>(choices), "stop");
  return stats;
}

void start_streaming_task(
    ResponseStreamPtr stream,
    std::string id,
    std::string model,
    int choice_count,
    const std::function<InferenceEngine::GenerationStats(const InferenceEngine::StreamCallback&)>& task,
    const std::function<void()>& on_done,
    std::shared_ptr<std::vector<std::string>> matched_stops) {
  std::thread([stream = std::move(stream),
               id = std::move(id),
               model = std::move(model),
               choice_count,
               task,
               on_done, matched_stops]() mutable {
    std::vector<bool> finished(static_cast<size_t>(choice_count), false);
    auto emit = [&stream, &id, &model, &finished, &matched_stops](int index, const std::string& chunk) -> bool {
      if (index < 0 || static_cast<size_t>(index) >= finished.size()) {
        throw std::runtime_error("generation returned invalid choice index");
      }
      if (finished[index]) return true;
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
      // The engine reserves empty chunks for per-row stop events.
      choice["delta"] = Json::objectValue;
      if (chunk.empty()) {
        choice["finish_reason"] = "stop";
        if (matched_stops && !matched_stops->at(index).empty())
          choice["matched_stop"] = matched_stops->at(index);
      } else {
        choice["delta"]["content"] = chunk;
        choice["finish_reason"] = Json::nullValue;
      }
      payload["choices"].append(choice);
      Json::StreamWriterBuilder builder;
      builder["emitUTF8"] = true;
      builder["indentation"] = "";
      const bool sent = stream->send("data: " + Json::writeString(builder, payload) + "\n\n");
      if (sent && chunk.empty()) finished[index] = true;
      return sent;
    };
    try {
      const auto stats = task(emit);
      if (stats.finish_reasons.size() != static_cast<size_t>(choice_count)) {
        throw std::runtime_error("generation returned incomplete finish reasons");
      }
      send_finish_chunk(stream, id, model, stats, finished);
    } catch (const PrefillBatchLimitExceeded& e) {
      Json::StreamWriterBuilder builder;
      builder["indentation"] = "";
      stream->send("data: " + Json::writeString(builder, prefill_limit_error(e)) + "\n\n");
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

std::string format_openai_prompt(
    const Json::Value& body,
    const InferenceEngine& engine,
    bool uploaded_state) {
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
  if (dialogue_messages.empty() && system_parts.empty() && !contents.empty() &&
      (contents.front().rfind("User:", 0) == 0 ||
       contents.front().find("\n\nAssistant:") != std::string::npos)) {
    // Compatibility clients may send an already formatted prompt through
    // contents. Do not wrap it in a second User/Assistant envelope.
    return contents.front();
  }
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
  const auto think_type = uploaded_state && !has_explicit_think_type(body)
      ? ThinkType::None
      : parse_think_type(body);
  return engine.format_openai_prompt(system, dialogue_messages, think_type);
}

Json::Value build_openai_response(
    const InferenceEngine& engine,
    const Json::Value& body,
    const std::string& prompt,
    const std::string& completion,
    const InferenceEngine::GenerationStats& stats) {
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
  choice["finish_reason"] = stats.finish_reasons.at(0);
  if (!stats.matched_stops.empty() && !stats.matched_stops[0].empty())
    choice["matched_stop"] = stats.matched_stops[0];
  resp["choices"].append(choice);
  resp["usage"]["prompt_tokens"] = engine.count_tokens(prompt);
  resp["usage"]["completion_tokens"] = engine.count_tokens(completion);
  resp["usage"]["total_tokens"] =
      resp["usage"]["prompt_tokens"].asInt() + resp["usage"]["completion_tokens"].asInt();
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
  out["force_reasoning"] = options.force_reasoning;
  if (options.stop) {
    out["stop"] = Json::arrayValue;
    for (const auto& stop : *options.stop) out["stop"].append(stop);
    return out;
  }
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

}  // namespace rwkv7_server::router
