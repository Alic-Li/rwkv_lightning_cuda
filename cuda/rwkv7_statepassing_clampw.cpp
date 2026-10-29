#include "rwkv/runtime/rwkv_training_wkv.hpp"

#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace rwkv7_state_tuning {

namespace {

std::size_t product(std::initializer_list<std::size_t> values) {
  std::size_t result = 1;
  for (std::size_t value : values) {
    if (value > std::numeric_limits<std::size_t>::max() / result) {
      throw std::overflow_error("state-tuning WKV tape size overflow");
    }
    result *= value;
  }
  return result;
}

void check(const WkvShape &shape) {
  if (shape.batch <= 0 || shape.time <= 0 || shape.heads <= 0 ||
      shape.head_size != 64) {
    throw std::invalid_argument(
        "state-tuning WKV requires B,T,H > 0 and head_size == 64");
  }
}

} // namespace

std::size_t wkv_checkpoint_elements(const WkvShape &shape) {
  check(shape);
  // Packed shapes count rows independently of lane count. A lane's
  // checkpoints start at its first row, so slot t / interval + lane needs at
  // most rows / interval (rounded up) plus one extra slot per further lane.
  const std::size_t rows = product({static_cast<std::size_t>(
                                        shape.offsets ? 1 : shape.batch),
                                    static_cast<std::size_t>(shape.time)});
  const std::size_t slots =
      (rows + kWkvCheckpointInterval - 1) / kWkvCheckpointInterval +
      static_cast<std::size_t>(shape.batch) - 1;
  return product({slots, static_cast<std::size_t>(shape.heads),
                  static_cast<std::size_t>(shape.head_size),
                  static_cast<std::size_t>(shape.head_size)});
}

std::size_t wkv_tape_elements(const WkvShape &shape) {
  const std::size_t scratch = product(
      {static_cast<std::size_t>(shape.batch),
       static_cast<std::size_t>(shape.heads),
       static_cast<std::size_t>(kWkvCheckpointInterval),
       static_cast<std::size_t>(shape.head_size),
       static_cast<std::size_t>(shape.head_size)});
  const std::size_t checkpoints = wkv_checkpoint_elements(shape);
  if (checkpoints > std::numeric_limits<std::size_t>::max() - scratch)
    throw std::overflow_error("state-tuning WKV tape size overflow");
  return checkpoints + scratch;
}

} // namespace rwkv7_state_tuning
