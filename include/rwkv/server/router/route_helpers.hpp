#pragma once

#include <chrono>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <drogon/drogon.h>

#include "rwkv/inference/rwkv_inference_engine.hpp"
#include "rwkv/server/rwkv_model_router.hpp"
#include "rwkv/server/rwkv_prefill_admission.hpp"
#include "rwkv/server/router/request_registry.hpp"

namespace rwkv7_server::router {

using namespace drogon;

using GenerationHttpHandler = std::function<void(
    const HttpRequestPtr&, std::function<void(const HttpResponsePtr&)>&&)>;
// Synchronous generation must not occupy Drogon's small I/O thread pool.
void configure_generation_workers(size_t count);
void shutdown_generation_workers();
GenerationHttpHandler asynchronous_generation(GenerationHttpHandler handler);

inline constexpr const char* kSessionIdHeader = "X-RWKV-Session-Id";
inline constexpr const char* kStateIdHeader = "X-RWKV-State-Id";

Json::Value make_error(const std::string& message);

Json::Value prefill_limit_error(const PrefillBatchLimitExceeded& error);

HttpResponsePtr json_response(Json::Value payload, HttpStatusCode code = k200OK);

std::optional<std::string> bearer_token(const HttpRequestPtr& req);

bool check_password(
    const HttpRequestPtr& req,
    const Json::Value& body,
    const std::optional<std::string>& password,
    HttpResponsePtr& out);

std::string trim(const std::string& input);

std::string request_header_value(const HttpRequestPtr& req, const char* name);

std::string state_id_from_body(const Json::Value& body);

std::string session_id_from_body(const Json::Value& body);

std::string resolve_identifier(
    std::initializer_list<std::pair<std::string, std::string>> channels,
    const std::string& field);

std::string session_id_from_request(const HttpRequestPtr& req, const Json::Value& body);

std::string state_id_from_request(const HttpRequestPtr& req, const Json::Value& body);

void record_prefill_metrics(
    const std::shared_ptr<RequestRegistry::ActiveRequest>& active,
    int prefill_tokens,
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end);

void update_decode_metrics(
    const std::shared_ptr<RequestRegistry::ActiveRequest>& active,
    int generated_tokens,
    double decode_seconds);

void update_decode_metrics(
    const std::shared_ptr<RequestRegistry::ActiveRequest>& active,
    std::chrono::steady_clock::time_point begin);

void record_generation_stats(
    const std::shared_ptr<RequestRegistry::ActiveRequest>& active,
    const InferenceEngine::GenerationStats& stats);

ThinkType parse_think_type(const Json::Value& body);

bool force_reasoning_for_think_type(ThinkType think_type);

bool has_explicit_think_type(const Json::Value& body);

GenerateOptions parse_options(const Json::Value& body, GenerateOptions options = {});

int parse_chunk_size(const Json::Value& body, int fallback);

bool parse_metrics_requested(const Json::Value& body);

std::string normalize_content(const Json::Value& content);

std::vector<std::string> parse_contents(const Json::Value& body);

GenerationState create_request_state(
    const InferenceEngine& engine,
    const std::string& state_id,
    int batch_size = 1);

std::string create_translation_prompt(
    const std::string& source_lang,
    const std::string& target_lang,
    const std::string& text);

Json::Value build_choices(const std::vector<std::string>& texts,
                          const InferenceEngine::GenerationStats& stats);

HttpResponsePtr make_sse_response(const std::function<void(ResponseStreamPtr)>& producer);

bool send_finish_chunk(
    const ResponseStreamPtr& stream,
    const std::string& id,
    const std::string& model,
    const InferenceEngine::GenerationStats& stats,
    const std::vector<bool>& finished);

InferenceEngine::GenerationStats cancelled_generation(int choices);

void start_streaming_task(
    ResponseStreamPtr stream,
    std::string id,
    std::string model,
    int choice_count,
    const std::function<InferenceEngine::GenerationStats(const InferenceEngine::StreamCallback&)>& task,
    const std::function<void()>& on_done = {},
    std::shared_ptr<std::vector<std::string>> matched_stops = {});

std::string format_openai_prompt(
    const Json::Value& body,
    const InferenceEngine& engine,
    bool uploaded_state = false);

Json::Value build_openai_response(
    const InferenceEngine& engine,
    const Json::Value& body,
    const std::string& prompt,
    const std::string& completion,
    const InferenceEngine::GenerationStats& stats);

Json::Value build_options_debug(const InferenceEngine& engine, const GenerateOptions& options);

void print_chat_context_debug(
    const char* endpoint,
    const std::string& session_id,
    const std::string& prompt,
    int prompt_tokens,
    const GenerateOptions& options);

Json::Value build_models_response(const ModelRouter& models);

Json::Value build_models_response(const InferenceEngine& engine);

Json::Value build_status_response(
    const InferenceEngine& engine,
    PrefillAdmissionController& admission);

}  // namespace rwkv7_server::router
