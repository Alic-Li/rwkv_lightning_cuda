#pragma once
#include <string>
namespace rwkv7_miss {
std::string fingerprint_file(const std::string &);
// Quantized model sidecar binds the runtime archive digest to its source PTH.
std::string source_fingerprint(const std::string &);
void write_source_fingerprint(const std::string &source,
                              const std::string &output);
} // namespace rwkv7_miss
