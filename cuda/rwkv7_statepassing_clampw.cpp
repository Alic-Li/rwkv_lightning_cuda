#include "rwkv/runtime/rwkv_state_tuning.hpp"

#include <limits>
#include <stdexcept>

namespace rwkv7_state_tuning {

std::size_t wkv_tape_elements(const WkvShape &shape) {
  if (shape.batch <= 0 || shape.time <= 0 || shape.heads <= 0 ||
      shape.head_size != 64) {
    throw std::invalid_argument(
        "state-tuning WKV requires B,T,H > 0 and head_size == 64");
  }
  const std::size_t values[] = {static_cast<std::size_t>(shape.batch),
                                static_cast<std::size_t>(shape.heads),
                                static_cast<std::size_t>(shape.time),
                                static_cast<std::size_t>(shape.head_size),
                                static_cast<std::size_t>(shape.head_size)};
  std::size_t result = 1;
  for (std::size_t value : values) {
    if (value > std::numeric_limits<std::size_t>::max() / result) {
      throw std::overflow_error("state-tuning WKV tape size overflow");
    }
    result *= value;
  }
  return result;
}

} // namespace rwkv7_state_tuning
