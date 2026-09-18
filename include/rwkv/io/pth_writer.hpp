#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace llm_infer {
struct WriteTensor {
  std::string name;
  std::vector<int> shape;
  bool fp16 = false;
  std::vector<std::uint8_t> data;
  bool bf16 = false; // exclusive with fp16; defaults retain the existing ABI

};
// Atomic stored-ZIP PyTorch state dictionary. No Python runtime dependency.
void write_pth(const std::string &, const std::vector<WriteTensor> &,
               const std::string &miss_manifest = {});
} // namespace llm_infer
