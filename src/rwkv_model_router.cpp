#include "rwkv_model_router.hpp"

#include <algorithm>
#include <stdexcept>

#include "rwkv_inference_engine.hpp"
#include "rwkv_server_backend.hpp"
#include "rwkv_tokenizer.hpp"

namespace rwkv7_server {

ModelRouter::Lease::Lease(ModelRouter* router, std::shared_ptr<InferenceEngine> engine)
    : router_(router), engine_(std::move(engine)) {}
ModelRouter::Lease::Lease(Lease&& other) noexcept
    : router_(other.router_), engine_(std::move(other.engine_)) { other.router_ = nullptr; }
ModelRouter::Lease& ModelRouter::Lease::operator=(Lease&& other) noexcept {
  if (this != &other) {
    release();
    router_ = other.router_;
    engine_ = std::move(other.engine_);
    other.router_ = nullptr;
  }
  return *this;
}
ModelRouter::Lease::~Lease() { release(); }
InferenceEngine& ModelRouter::Lease::engine() const { return *engine_; }
void ModelRouter::Lease::release() {
  if (router_) router_->release_lease();
  router_ = nullptr;
  engine_.reset();
}

ModelRouter::ModelRouter(std::shared_ptr<InferenceEngine> engine)
    : current_engine_(std::move(engine)), current_id_(current_engine_->model_name()) {}

ModelRouter::ModelRouter(std::filesystem::path model_directory,
                         std::shared_ptr<TrieTokenizer> tokenizer,
                         int prefill_chunk_size, bool use_wkv32, bool chunk_load)
    : dynamic_(true), tokenizer_(std::move(tokenizer)), prefill_chunk_size_(prefill_chunk_size),
      use_wkv32_(use_wkv32), chunk_load_(chunk_load) {
  if (!std::filesystem::is_directory(model_directory)) {
    throw std::runtime_error("--model-path must be a directory when --enable-dynamic-loading is set: " +
                             model_directory.string());
  }
  for (const auto& entry : std::filesystem::directory_iterator(model_directory)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".pth") continue;
    const std::string id = entry.path().stem().string();
    if (!model_paths_.emplace(id, entry.path()).second) {
      throw std::runtime_error("duplicate dynamic model id: " + id);
    }
  }
  if (model_paths_.empty()) {
    throw std::runtime_error("no .pth model files found in: " + model_directory.string());
  }
}

std::string ModelRouter::resolve_id_or_throw(const std::string& requested_model) const {
  if (!dynamic_) return current_id_;
  if (requested_model.empty()) {
    throw std::runtime_error("'model' is required");
  }
  if (model_paths_.find(requested_model) == model_paths_.end()) {
    throw std::runtime_error("unknown model: " + requested_model);
  }
  return requested_model;
}

ModelRouter::Lease ModelRouter::acquire() {
  std::unique_lock<std::mutex> lock(mutex_);
  changed_.wait(lock, [this] { return !loading_; });
  if (!current_engine_) {
    throw std::runtime_error("no model is loaded; call POST /v1/model/load first");
  }
  ++active_leases_;
  return Lease(this, current_engine_);
}

void ModelRouter::load(const std::string& requested_model) {
  const std::string id = resolve_id_or_throw(requested_model);
  if (!dynamic_) {
    if (id != current_id_) throw std::runtime_error("dynamic model loading is disabled");
    return;
  }
  std::unique_lock<std::mutex> lock(mutex_);
  const std::uint64_t ticket = next_load_ticket_++;
  load_queue_.push_back(ticket);
  changed_.wait(lock, [this, ticket] {
    return load_queue_.front() == ticket && !loading_ && active_leases_ == 0;
  });
  if (current_engine_ && current_id_ == id) {
    load_queue_.pop_front();
    changed_.notify_all();
    return;
  }
  loading_ = true;
  current_engine_.reset();  // release old model VRAM before loading the next one
  current_id_.clear();
  lock.unlock();
  std::shared_ptr<InferenceEngine> loaded;
  try {
    auto backend = std::make_shared<ModelBackend>(
        model_paths_.at(id).string(), use_wkv32_, chunk_load_);
    loaded = std::make_shared<InferenceEngine>(backend, tokenizer_, backend->model_name(), prefill_chunk_size_);
  } catch (...) {
    lock.lock();
    loading_ = false;
    load_queue_.pop_front();
    changed_.notify_all();
    throw;
  }
  lock.lock();
  current_engine_ = std::move(loaded);
  current_id_ = id;
  loading_ = false;
  load_queue_.pop_front();
  changed_.notify_all();
}

void ModelRouter::release_lease() {
  std::lock_guard<std::mutex> lock(mutex_);
  --active_leases_;
  changed_.notify_all();
}

std::vector<std::string> ModelRouter::model_ids() const {
  std::vector<std::string> ids;
  if (!dynamic_) return {current_id_};
  ids.reserve(model_paths_.size());
  for (const auto& item : model_paths_) ids.push_back(item.first);
  std::sort(ids.begin(), ids.end());
  return ids;
}

std::string ModelRouter::current_model_id() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return current_id_;
}

}  // namespace rwkv7_server
