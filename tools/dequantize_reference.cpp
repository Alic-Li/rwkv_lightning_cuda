#include "rwkv/runtime/rwkv7_fast_v4_common.hpp"
#include "rwkv_quantized.hpp"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

// Validation utility: expand integer tensors to FP16 in an RWKVQ archive.
// This provides an independent CPU dequantization reference for HIP inference.
int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: rwkv_dequantize_reference INPUT_RWKVQ OUTPUT_RWKVQ\n";
    return 2;
  }
  try {
    if (std::filesystem::exists(argv[2]))
      throw std::runtime_error("reference output already exists");
    auto archive = llm_infer::QuantizedArchive::open(argv[1]);
    if (!archive.ok())
      throw std::runtime_error(archive.status().message());
    llm_infer::QuantizedWriter writer(argv[2],
                                      archive.value().records().size());
    for (const auto &rec : archive.value().records()) {
      std::vector<std::uint8_t> data;
      auto status = archive.value().read_data(rec, &data);
      if (!status.ok_status())
        throw std::runtime_error(status.message());
      auto dtype = rec.dtype;
      if (llm_infer::asym_bits(dtype)) {
        std::vector<std::uint16_t> metadata;
        status = archive.value().read_scales(rec, &metadata);
        if (!status.ok_status())
          throw std::runtime_error(status.message());
        auto expanded = llm_infer::expand_asym_metadata(rec, metadata);
        if (!expanded.ok())
          throw std::runtime_error(expanded.status().message());
        const int bits = llm_infer::asym_bits(dtype);
        const std::size_t N = rec.shape[0], K = rec.shape[1],
                          groups = (K + 31) / 32, stride = 16 + 4 * (bits - 4);
        std::vector<std::uint8_t> output(N * K * 2);
        for (std::size_t n = 0; n < N; ++n)
          for (std::size_t k = 0; k < K; ++k) {
            const auto g = n * groups + k / 32, j = k % 32;
            const auto *p = data.data() + g * stride;
            int q = (p[j / 2] >> ((j & 1) * 4)) & 15;
            for (int b = 4; b < bits; ++b)
              q |= ((p[16 + (b - 4) * 4 + j / 8] >> (j % 8)) & 1) << b;
            const float s =
                llm_infer::f16_bits_to_float(expanded.value()[2 * g]);
            const float m =
                llm_infer::f16_bits_to_float(expanded.value()[2 * g + 1]);
            const half h = __float2half_rn(std::fma(float(q), s, m));
            std::uint16_t raw;
            std::memcpy(&raw, &h, sizeof(raw));
            output[(n * K + k) * 2] = raw;
            output[(n * K + k) * 2 + 1] = raw >> 8;
          }
        data = std::move(output);
        dtype = llm_infer::QuantizedDType::kFloat16;
      } else if (dtype == llm_infer::QuantizedDType::kInt8) {
        if (rec.shape.size() != 2)
          throw std::runtime_error("reference requires rank-2 integer weights");
        std::vector<std::uint16_t> scales;
        status = archive.value().read_scales(rec, &scales);
        if (!status.ok_status())
          throw std::runtime_error(status.message());
        const std::size_t N = rec.shape[0], K = rec.shape[1];
        std::vector<std::uint8_t> output(N * K * 2);
        for (std::size_t n = 0; n < N; ++n)
          for (std::size_t k = 0; k < K; ++k) {
            const int v = data[n * K + k];
            const int q = v < 128 ? v : v - 256;
            const auto h = llm_infer::float_to_f16_bits(
                q * llm_infer::f16_bits_to_float(scales[n]));
            output[(n * K + k) * 2] = h;
            output[(n * K + k) * 2 + 1] = h >> 8;
          }
        data = std::move(output);
        dtype = llm_infer::QuantizedDType::kFloat16;
      }
      status = writer.append(rec.name, dtype, rec.shape, data);
      if (!status.ok_status())
        throw std::runtime_error(status.message());
    }
    const auto status = writer.close();
    if (!status.ok_status())
      throw std::runtime_error(status.message());
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
