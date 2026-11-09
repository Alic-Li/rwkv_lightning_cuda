#pragma once

#include <memory>
#include <optional>
#include <string>

namespace drogon { class HttpAppFramework; }
namespace rwkv7_server {
class InferenceEngine;
class ModelRouter;
class PrefillAdmissionController;

namespace router {

void register_legacy_routes(
    drogon::HttpAppFramework& app, InferenceEngine& engine,
    const std::optional<std::string>& password);
void register_dynamic_routes(
    drogon::HttpAppFramework& app, ModelRouter& models,
    const std::optional<std::string>& password);
void register_adapter_routes(
    drogon::HttpAppFramework& app, const std::optional<std::string>& password);
void register_uploaded_state_routes(
    drogon::HttpAppFramework& app, const std::optional<std::string>& password);

void register_server_routes(
    drogon::HttpAppFramework& app, InferenceEngine& engine,
    const std::optional<std::string>& password,
    const std::shared_ptr<PrefillAdmissionController>& admission);

void register_token_routes(
    drogon::HttpAppFramework& app, InferenceEngine& engine,
    const std::optional<std::string>& password);

void register_batch_routes(
    drogon::HttpAppFramework& app, InferenceEngine& engine,
    const std::optional<std::string>& password,
    const std::shared_ptr<PrefillAdmissionController>& admission);

void register_translation_routes(
    drogon::HttpAppFramework& app, InferenceEngine& engine,
    const std::shared_ptr<PrefillAdmissionController>& admission);

void register_session_routes(
    drogon::HttpAppFramework& app, InferenceEngine& engine,
    const std::optional<std::string>& password,
    const std::shared_ptr<PrefillAdmissionController>& admission);

void register_chat_routes(
    drogon::HttpAppFramework& app, InferenceEngine& engine,
    const std::optional<std::string>& password,
    const std::shared_ptr<PrefillAdmissionController>& admission);

void register_model_routes(
    drogon::HttpAppFramework& app, InferenceEngine& engine,
    const std::optional<std::string>& password);

}  // namespace router
}  // namespace rwkv7_server
