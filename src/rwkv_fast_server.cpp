#include <atomic>
#include <cstdlib>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

#include <drogon/drogon.h>

#include "rwkv_api_service.hpp"
#include "rwkv_inference_engine.hpp"
#include "rwkv_server_backend.hpp"
#include "rwkv_state_cache.hpp"
#include "rwkv_tokenizer.hpp"

namespace {

std::atomic<bool> g_shutdown{false};

constexpr const char* kDefaultHost = "127.0.0.1";

void print_usage(const char* program) {
  std::cout
      << "Usage: " << program << " --model-path <path> --vocab-path <path> [options]\n"
      << "\n"
      << "Options:\n"
      << "  --model-path <path>     Path to the RWKV .pth model file. Required.\n"
      << "  --vocab-path <path>     Path to rwkv_vocab_v20230424.txt. Required.\n"
      << "  --host <addr>           Host/interface to bind. Default: 127.0.0.1.\n"
      << "                          Use 0.0.0.0 explicitly to listen on all IPv4 interfaces.\n"
      << "  --port <port>           TCP port to bind. Default: 8000.\n"
      << "  --state-db-path <path>  SQLite state cache path. Default: rwkv_sessions.db.\n"
      << "  --password <token>      Require a bearer token or JSON password field.\n"
      << "  --wkv32                 Use fp32 WKV state with fp16 IO.\n"
      << "  --help, -h              Show this help text.\n";
}

uint16_t parse_port(const std::string& value) {
  std::size_t parsed = 0;
  int port = 0;
  try {
    port = std::stoi(value, &parsed);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid value for --port: " + value);
  }
  if (parsed != value.size() || port < 1 || port > std::numeric_limits<uint16_t>::max()) {
    throw std::runtime_error("invalid value for --port: " + value);
  }
  return static_cast<uint16_t>(port);
}

std::string format_url_host(const std::string& host) {
  if (host.find(':') != std::string::npos &&
      !(host.size() >= 2 && host.front() == '[' && host.back() == ']')) {
    return "[" + host + "]";
  }
  return host;
}

void handle_signal(int) {
  g_shutdown = true;
  try {
    rwkv7_server::StateCacheManager::instance().shutdown();
  } catch (...) {
  }
  drogon::app().quit();
}

#ifdef _WIN32
void configure_windows_dll_search_path() {
  char exe_path[MAX_PATH] = {0};
  const DWORD n = GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
  if (n == 0 || n >= MAX_PATH) {
    return;
  }

  std::filesystem::path lib_dir = std::filesystem::path(exe_path).parent_path() / "lib";
  if (!std::filesystem::exists(lib_dir)) {
    return;
  }

  // Ensure bundled DLLs in ./lib are discoverable regardless of the launch cwd.
  SetDllDirectoryA(lib_dir.string().c_str());
}
#endif

}  // namespace

int run_server(int argc, char* argv[]) {
#ifdef _WIN32
  configure_windows_dll_search_path();
#endif
  trantor::Logger::setLogLevel(trantor::Logger::kInfo);
  std::string model_path;
  std::string vocab_path;
  std::string host = kDefaultHost;
  std::string state_db_path = "rwkv_sessions.db";
  uint16_t port = 8000;
  bool use_wkv32 = false;
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
    } else if (arg == "--host") {
      host = require_value(arg);
    } else if (arg == "--state-db-path") {
      state_db_path = require_value(arg);
    } else if (arg == "--port") {
      port = parse_port(require_value(arg));
    } else if (arg == "--password") {
      password = require_value(arg);
    } else if (arg == "--wkv32") {
      use_wkv32 = true;
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
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
  if (host.empty()) {
    throw std::runtime_error("--host must not be empty");
  }

  auto model = std::make_shared<rwkv7_server::ModelBackend>(model_path, use_wkv32);
  auto tokenizer = std::make_shared<rwkv7_server::TrieTokenizer>();
  if (tokenizer->load(vocab_path) != rwkv7_server::kTokenizerSuccess) {
    throw std::runtime_error("failed to load tokenizer vocab: " + vocab_path);
  }

  rwkv7_server::InferenceEngine engine(model, tokenizer, model->model_name());
  rwkv7_server::StateCacheManager::instance().initialize(16, 32, state_db_path);

  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);

  rwkv7_server::register_api_routes(engine, password);
  std::cout << "rwkv_lighting_cuda model_name=" << model->model_name()
            << " model_path=" << model->model_path() << std::endl;
  std::cout << "rwkv_lighting_cuda vocab_path=" << vocab_path
            << " state_db_path=" << state_db_path
            << " host=" << host
            << " port=" << port
            << " wkv=" << (use_wkv32 ? "fp32io16" : "fp16")
            << " password=" << (password.has_value() ? "enabled" : "disabled") << std::endl;
  const std::string url_host = format_url_host(host);
  for (const std::string& endpoint : std::vector<std::string>{
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
      std::cout << "||      http://" << url_host << ":" << port << endpoint << std::endl;
  }
  std::cout << "listening on " << host << ":" << port << std::endl;
  std::cout << "Mamba Out!!! Nvidia Fuck You !!!" << std::endl;
  std::cout << "ROCm RWKV Is All You Need !!!" << std::endl;
  std::cout << "Boom Has Been Planted !!!" << std::endl;

  drogon::app()
      .addListener(host, port)
      .setThreadNum(4)
      .run();

  rwkv7_server::StateCacheManager::instance().shutdown();
  return 0;
}

int main(int argc, char* argv[]) {
  try {
    return run_server(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "rwkv_lighting_cuda: " << error.what() << std::endl;
    std::cerr << "Try '--help' for usage." << std::endl;
    return EXIT_FAILURE;
  }
}
