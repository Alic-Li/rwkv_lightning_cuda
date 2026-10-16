#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include <utility>

namespace rwkv7_server {

// Tokenizer decode returns bytes, so retaining the longest stop's byte length
// also handles UTF-8 sequences split across tokens. Match generated text only.
class TextStopMatcher {
 public:
  explicit TextStopMatcher(std::vector<std::string> stops) : stops_(std::move(stops)) {
    for (const auto& stop : stops_) max_bytes_ = std::max(max_bytes_, stop.size());
  }

  std::string append(std::string& tail, const std::string& decoded) const {
    if (max_bytes_ == 0) return {};
    if (decoded.size() >= max_bytes_) {
      tail.assign(decoded, decoded.size() - max_bytes_, max_bytes_);
    } else {
      if (tail.size() + decoded.size() > max_bytes_)
        tail.erase(0, tail.size() + decoded.size() - max_bytes_);
      tail += decoded;
    }
    // Multiple matching suffixes resolve in request order.
    for (const auto& stop : stops_) {
      if (!stop.empty() && tail.size() >= stop.size() &&
          tail.compare(tail.size() - stop.size(), stop.size(), stop) == 0)
        return stop;
    }
    return {};
  }

 private:
  std::vector<std::string> stops_;
  size_t max_bytes_ = 0;
};

}  // namespace rwkv7_server
