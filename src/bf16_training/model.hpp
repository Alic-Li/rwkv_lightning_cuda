#pragma once
#include "training.hpp"
#include <map>
namespace rwkv_bf16_training {
class Model {
  std::map<std::string, DeviceBuffer<bf16>> weights_;
  std::vector<uint16_t> embeddings_;
  FrozenModelView view_;
  std::string fingerprint_;

public:
  explicit Model(const std::string &path);
  FrozenModelView state_tuning_model_view() const { return view_; }
  const std::string &base_fingerprint() const { return fingerprint_; }
  DeviceBuffer<float> load_state(const std::string &path) const;
};
void transpose_bf16(int rows, int cols, const bf16 *input, bf16 *output);
} // namespace rwkv_bf16_training
