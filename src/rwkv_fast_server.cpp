#include <atomic>
#include <csignal>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <drogon/drogon.h>

#include "rwkv_api_service.hpp"
#include "rwkv_inference_engine.hpp"
#include "rwkv_server_backend.hpp"
#include "rwkv_state_cache.hpp"
#include "rwkv_tokenizer.hpp"

namespace {

std::atomic<bool> g_shutdown{false};

void print_server_banner(
    const rwkv7_server::ModelBackend& model,
    const std::string& vocab_path,
    const std::string& state_db_path,
    uint16_t port,
    const std::optional<std::string>& password) {
  const std::vector<std::string> endpoints{
      "/v1/batch/completions",
      "/translate/v1/batch-translate",
      "/state/chat/completions",
      "/state/status",
      "/state/delete",
      "/v1/chat/completions",
      "/v1/models"};

  std::cout << "rwkv_lighting_cuda model_name=" << model.model_name()
            << " model_path=" << model.model_path() << "\n";
  std::cout << "rwkv_lighting_cuda vocab_path=" << vocab_path
            << " state_db_path=" << state_db_path
            << " port=" << port
            << " password=" << (password.has_value() ? "enabled" : "disabled") << "\n";
  std::cout << "rwkv_lighting_cuda api_endpoints:\n";
  for (const auto& endpoint : endpoints) {
    std::cout << "  - http://0.0.0.0:" << port << endpoint << "\n";
  }
}

void handle_signal(int) {
  g_shutdown = true;
  try {
    rwkv7_server::StateCacheManager::instance().shutdown();
  } catch (...) {
  }
  drogon::app().quit();
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string model_path;
  std::string vocab_path;
  std::string state_db_path = "rwkv_sessions.db";
  uint16_t port = 8000;
  std::optional<std::string> password;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto require_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + name);
      }
      return argv[++i];
    };
    if (arg == "--model-path") {
      model_path = require_value(arg);
    } else if (arg == "--vocab-path") {
      vocab_path = require_value(arg);
    } else if (arg == "--state-db-path") {
      state_db_path = require_value(arg);
    } else if (arg == "--port") {
      port = static_cast<uint16_t>(std::stoi(require_value(arg)));
    } else if (arg == "--password") {
      password = require_value(arg);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (model_path.empty()) {
    throw std::runtime_error("--model-path is required");
  }
  if (vocab_path.empty()) {
    throw std::runtime_error("--vocab-path is required");
  }

  auto model = std::make_shared<rwkv7_server::ModelBackend>(model_path);
  auto tokenizer = std::make_shared<rwkv7_server::TrieTokenizer>();
  if (tokenizer->load(vocab_path) != rwkv7_server::kTokenizerSuccess) {
    throw std::runtime_error("failed to load tokenizer vocab: " + vocab_path);
  }

  rwkv7_server::InferenceEngine engine(model, tokenizer, model->model_name());
  rwkv7_server::StateCacheManager::instance().initialize(16, 32, state_db_path);

  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);

  rwkv7_server::register_api_routes(engine, password);
  print_server_banner(*model, vocab_path, state_db_path, port, password);

  drogon::app()
      .addListener("0.0.0.0", port)
      .setThreadNum(4)
      .run();

  rwkv7_server::StateCacheManager::instance().shutdown();
  return 0;
}
