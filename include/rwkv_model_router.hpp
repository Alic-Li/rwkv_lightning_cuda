#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rwkv7_server {

class InferenceEngine;

// Owns the currently resident CUDA model.  A Lease pins it for a request, so a
// model switch can never destroy weights or states that are still in use.
class ModelRouter {
 public:
  class Lease {
   public:
    Lease() = default;
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& other) noexcept;
    Lease& operator=(Lease&& other) noexcept;
    ~Lease();

    InferenceEngine& engine() const;
    explicit operator bool() const { return static_cast<bool>(engine_); }

   private:
    friend class ModelRouter;
    Lease(ModelRouter* router, std::shared_ptr<InferenceEngine> engine);
    void release();
    ModelRouter* router_ = nullptr;
    std::shared_ptr<InferenceEngine> engine_;
  };

  // The static constructor preserves the legacy single-model startup path.
  explicit ModelRouter(std::shared_ptr<InferenceEngine> engine);
  // In dynamic mode model_directory contains the selectable .pth files.
  ModelRouter(std::filesystem::path model_directory, std::shared_ptr<class TrieTokenizer> tokenizer,
              int prefill_chunk_size, bool use_wkv32, bool chunk_load = false);

  // Pins the currently loaded model for a request. Request model IDs are not
  // routing directives: switching is exclusively done by load().
  Lease acquire();
  void load(const std::string& model_id);
  std::vector<std::string> model_ids() const;
  bool dynamic_loading_enabled() const { return dynamic_; }
  std::string current_model_id() const;

 private:
  void release_lease();
  std::string resolve_id_or_throw(const std::string& requested_model) const;

  bool dynamic_ = false;
  std::shared_ptr<class TrieTokenizer> tokenizer_;
  int prefill_chunk_size_ = 128;
  bool use_wkv32_ = false;
  bool chunk_load_ = false;
  std::unordered_map<std::string, std::filesystem::path> model_paths_;

  mutable std::mutex mutex_;
  std::condition_variable changed_;
  std::shared_ptr<InferenceEngine> current_engine_;
  std::string current_id_;
  std::size_t active_leases_ = 0;
  bool loading_ = false;
  std::deque<std::uint64_t> load_queue_;
  std::uint64_t next_load_ticket_ = 0;
};

}  // namespace rwkv7_server
