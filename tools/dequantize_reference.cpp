#include "rwkv/runtime/rwkv7_fast_v4_common.hpp"
#include "rwkv_quantized.hpp"
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
      if (dtype == llm_infer::QuantizedDType::kInt4 ||
          dtype == llm_infer::QuantizedDType::kInt8) {
        if (rec.shape.size() != 2)
          throw std::runtime_error("reference requires rank-2 integer weights");
        std::vector<std::uint16_t> scales;
        status = archive.value().read_scales(rec, &scales);
        if (!status.ok_status())
          throw std::runtime_error(status.message());
        const std::size_t N = rec.shape[0], K = rec.shape[1],
                          G = rec.quant_group_size;
        std::vector<std::uint8_t> output(N * K * 2);
        for (std::size_t n = 0; n < N; ++n)
          for (std::size_t k = 0; k < K; ++k) {
            int q;
            std::size_t si;
            if (dtype == llm_infer::QuantizedDType::kInt4) {
              const int v =
                  (data[n * ((K + 1) / 2) + k / 2] >> (4 * (k % 2))) & 15;
              q = (v ^ 8) - 8;
              si = n * ((K + G - 1) / G) + k / G;
            } else {
              const int v = data[n * K + k];
              q = v < 128 ? v : v - 256;
              si = n;
            }
            const auto h = llm_infer::float_to_f16_bits(
                q * llm_infer::f16_bits_to_float(scales[si]));
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
