#pragma once
#include <cstddef>
#include <string>
namespace rwkv_adapter_detail {
// One logical target owns D for MiSS, or B followed by A for NoRA.
template <class T, class Names, class Fn>
void components(const T &t, bool nora, const Names &names, Fn fn) {
  const auto prefix =
      "blocks." + std::to_string(t.layer) + "." + names[t.target];
  fn(prefix + (nora ? ".B" : ".D"), t.out, t.rank, t.offset);
  if (nora)
    fn(prefix + ".A", t.rank, t.in, t.a_offset);
}
} // namespace rwkv_adapter_detail
