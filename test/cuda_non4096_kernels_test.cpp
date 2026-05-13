#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "rwkv7_fast_v4_kernels.cuh"
#include "test_common.hpp"

namespace {

using rwkv7_fast_v4::DeviceBuffer;

constexpr int kChannels = 2560;
constexpr float kEps = 1.0e-5f;

float pattern_value(int index, float scale, float bias) {
  return std::sinf(static_cast<float>(index % 97) * 0.173f) * scale +
         std::cosf(static_cast<float>(index % 53) * 0.071f) * (scale * 0.5f) +
         bias;
}

std::uint16_t float_to_half_bits(float value) {
  const half h = __float2half(value);
  return *reinterpret_cast<const std::uint16_t*>(&h);
}

float half_bits_to_float(std::uint16_t bits) {
  const half h = *reinterpret_cast<const half*>(&bits);
  return __half2float(h);
}

std::uint16_t float_to_bf16_bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<std::uint16_t>(bits >> 16);
}

std::vector<float> from_half_bits(const std::vector<std::uint16_t>& bits) {
  std::vector<float> values(bits.size());
  for (std::size_t i = 0; i < bits.size(); ++i) {
    values[i] = half_bits_to_float(bits[i]);
  }
  return values;
}

std::vector<float> copy_half_buffer(const DeviceBuffer<std::uint16_t>& buffer, const char* label) {
  return from_half_bits(rwkv_test::copy_device_buffer(buffer, label));
}

void copy_half_to_device(
    const std::vector<float>& host,
    DeviceBuffer<std::uint16_t>& buffer,
    const char* resize_label,
    const char* copy_label) {
  rwkv_test::copy_host_to_device(rwkv_test::to_half_bits(host), buffer, resize_label, copy_label);
}

void copy_bf16_to_device(
    const std::vector<float>& host,
    DeviceBuffer<std::uint16_t>& buffer,
    const char* resize_label,
    const char* copy_label) {
  std::vector<std::uint16_t> bits(host.size());
  for (std::size_t i = 0; i < host.size(); ++i) {
    bits[i] = float_to_bf16_bits(host[i]);
  }
  rwkv_test::copy_host_to_device(bits, buffer, resize_label, copy_label);
}

void expect_vector_near(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float eps) {
  TEST_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    TEST_NEAR(actual[i], expected[i], eps);
  }
}

void cpu_add_layer_norm_cmix_mix(
    int rows,
    int channels,
    const std::vector<float>& x,
    const std::vector<float>& residual,
    const std::vector<float>& shift_state,
    const std::vector<float>& weight,
    const std::vector<float>& bias,
    const std::vector<float>& x_k,
    std::vector<float>& x_out,
    std::vector<float>& mixed,
    std::vector<float>& shift_out) {
  x_out.resize(static_cast<std::size_t>(rows) * channels);
  mixed.resize(static_cast<std::size_t>(rows) * channels);
  shift_out.resize(static_cast<std::size_t>(rows) * channels);
  for (int row = 0; row < rows; ++row) {
    const int64_t base = static_cast<int64_t>(row) * channels;
    float sum = 0.0f;
    for (int c = 0; c < channels; ++c) {
      sum += x[base + c] + residual[base + c];
    }
    const float mean = sum / static_cast<float>(channels);
    float sum_var = 0.0f;
    for (int c = 0; c < channels; ++c) {
      const float v = x[base + c] + residual[base + c];
      const float d = v - mean;
      sum_var += d * d;
    }
    const float rstd = 1.0f / std::sqrt(sum_var / static_cast<float>(channels) + kEps);
    for (int c = 0; c < channels; ++c) {
      const float v = x[base + c] + residual[base + c];
      const float yv = (v - mean) * rstd * weight[c] + bias[c];
      x_out[base + c] = v;
      mixed[base + c] = yv + (shift_state[base + c] - yv) * x_k[c];
      shift_out[base + c] = yv;
    }
  }
}

void cpu_add_layer_norm_tmix_mix6(
    int rows,
    int channels,
    const std::vector<float>& x,
    const std::vector<float>& residual,
    const std::vector<float>& shift_state,
    const std::vector<float>& weight,
    const std::vector<float>& bias,
    const std::vector<float>& x_r,
    const std::vector<float>& x_w,
    const std::vector<float>& x_k,
    const std::vector<float>& x_v,
    const std::vector<float>& x_a,
    const std::vector<float>& x_g,
    std::vector<float>& x_out,
    std::vector<float>& out_r,
    std::vector<float>& out_w,
    std::vector<float>& out_k,
    std::vector<float>& out_v,
    std::vector<float>& out_a,
    std::vector<float>& out_g,
    std::vector<float>& shift_out) {
  x_out.resize(static_cast<std::size_t>(rows) * channels);
  out_r.resize(static_cast<std::size_t>(rows) * channels);
  out_w.resize(static_cast<std::size_t>(rows) * channels);
  out_k.resize(static_cast<std::size_t>(rows) * channels);
  out_v.resize(static_cast<std::size_t>(rows) * channels);
  out_a.resize(static_cast<std::size_t>(rows) * channels);
  out_g.resize(static_cast<std::size_t>(rows) * channels);
  shift_out.resize(static_cast<std::size_t>(rows) * channels);
  for (int row = 0; row < rows; ++row) {
    const int64_t base = static_cast<int64_t>(row) * channels;
    float sum = 0.0f;
    for (int c = 0; c < channels; ++c) {
      sum += x[base + c] + residual[base + c];
    }
    const float mean = sum / static_cast<float>(channels);
    float sum_var = 0.0f;
    for (int c = 0; c < channels; ++c) {
      const float v = x[base + c] + residual[base + c];
      const float d = v - mean;
      sum_var += d * d;
    }
    const float rstd = 1.0f / std::sqrt(sum_var / static_cast<float>(channels) + kEps);
    for (int c = 0; c < channels; ++c) {
      const float v = x[base + c] + residual[base + c];
      const float yv = (v - mean) * rstd * weight[c] + bias[c];
      const float delta = shift_state[base + c] - yv;
      x_out[base + c] = v;
      out_r[base + c] = yv + delta * x_r[c];
      out_w[base + c] = yv + delta * x_w[c];
      out_k[base + c] = yv + delta * x_k[c];
      out_v[base + c] = yv + delta * x_v[c];
      out_a[base + c] = yv + delta * x_a[c];
      out_g[base + c] = yv + delta * x_g[c];
      shift_out[base + c] = yv;
    }
  }
}

void cpu_add_last_layer_norm(
    int batch,
    int steps,
    int channels,
    const std::vector<float>& x,
    const std::vector<float>& residual,
    const std::vector<float>& weight,
    const std::vector<float>& bias,
    std::vector<float>& y) {
  y.resize(static_cast<std::size_t>(batch) * channels);
  for (int b = 0; b < batch; ++b) {
    const int64_t src = static_cast<int64_t>(b * steps + (steps - 1)) * channels;
    const int64_t dst = static_cast<int64_t>(b) * channels;
    float sum = 0.0f;
    for (int c = 0; c < channels; ++c) {
      sum += x[src + c] + residual[src + c];
    }
    const float mean = sum / static_cast<float>(channels);
    float sum_var = 0.0f;
    for (int c = 0; c < channels; ++c) {
      const float v = x[src + c] + residual[src + c];
      const float d = v - mean;
      sum_var += d * d;
    }
    const float rstd = 1.0f / std::sqrt(sum_var / static_cast<float>(channels) + kEps);
    for (int c = 0; c < channels; ++c) {
      const float v = x[src + c] + residual[src + c];
      y[dst + c] = (v - mean) * rstd * weight[c] + bias[c];
    }
  }
}

void cpu_emb_ln0(
    int vocab,
    int channels,
    const std::vector<float>& emb,
    const std::vector<float>& weight,
    const std::vector<float>& bias,
    std::vector<float>& out) {
  out.resize(static_cast<std::size_t>(vocab) * channels);
  for (int tok = 0; tok < vocab; ++tok) {
    const int64_t base = static_cast<int64_t>(tok) * channels;
    float sum = 0.0f;
    for (int c = 0; c < channels; ++c) {
      sum += emb[base + c];
    }
    const float mean = sum / static_cast<float>(channels);
    float sum_var = 0.0f;
    for (int c = 0; c < channels; ++c) {
      const float d = emb[base + c] - mean;
      sum_var += d * d;
    }
    const float rstd = 1.0f / std::sqrt(sum_var / static_cast<float>(channels) + kEps);
    for (int c = 0; c < channels; ++c) {
      out[base + c] = (emb[base + c] - mean) * rstd * weight[c] + bias[c];
    }
  }
}

}  // namespace

int main() {
  try {
    if (!rwkv_test::cuda_device_available()) {
      std::cout << "rwkv_cuda_non4096_kernels_test skipped: no CUDA device available\n";
      return 0;
    }

    {
      const int rows = 3;
      std::vector<float> x(rows * kChannels);
      std::vector<float> residual(rows * kChannels);
      std::vector<float> shift_state(rows * kChannels);
      std::vector<float> weight(kChannels);
      std::vector<float> bias(kChannels);
      std::vector<float> x_k(kChannels);
      for (int i = 0; i < rows * kChannels; ++i) {
        x[i] = pattern_value(i, 0.2f, -0.05f);
        residual[i] = pattern_value(i + 17, 0.15f, 0.03f);
        shift_state[i] = pattern_value(i + 33, 0.25f, -0.02f);
      }
      for (int c = 0; c < kChannels; ++c) {
        weight[c] = pattern_value(c + 5, 0.08f, 1.0f);
        bias[c] = pattern_value(c + 9, 0.03f, 0.01f);
        x_k[c] = 0.5f + 0.4f * std::sinf(static_cast<float>(c) * 0.007f);
      }
      std::vector<float> expect_x_out;
      std::vector<float> expect_mixed;
      std::vector<float> expect_shift;
      cpu_add_layer_norm_cmix_mix(
          rows, kChannels, x, residual, shift_state, weight, bias, x_k,
          expect_x_out, expect_mixed, expect_shift);

      DeviceBuffer<std::uint16_t> x_dev;
      DeviceBuffer<std::uint16_t> residual_dev;
      DeviceBuffer<std::uint16_t> shift_dev;
      DeviceBuffer<std::uint16_t> weight_dev;
      DeviceBuffer<std::uint16_t> bias_dev;
      DeviceBuffer<std::uint16_t> xk_dev;
      DeviceBuffer<std::uint16_t> x_out_dev;
      DeviceBuffer<std::uint16_t> mixed_dev;
      copy_half_to_device(x, x_dev, "alloc x", "copy x");
      copy_half_to_device(residual, residual_dev, "alloc residual", "copy residual");
      copy_half_to_device(shift_state, shift_dev, "alloc shift", "copy shift");
      copy_half_to_device(weight, weight_dev, "alloc weight", "copy weight");
      copy_half_to_device(bias, bias_dev, "alloc bias", "copy bias");
      copy_half_to_device(x_k, xk_dev, "alloc x_k", "copy x_k");
      x_out_dev.resize(x.size(), "alloc x_out");
      mixed_dev.resize(x.size(), "alloc mixed");
      rwkv7_v3a_add_layer_norm_cmix_mix_f16_launch(
          nullptr, rows, kChannels,
          reinterpret_cast<const half*>(x_dev.p),
          reinterpret_cast<const half*>(residual_dev.p),
          reinterpret_cast<half*>(shift_dev.p),
          reinterpret_cast<const half*>(weight_dev.p),
          reinterpret_cast<const half*>(bias_dev.p),
          reinterpret_cast<const half*>(xk_dev.p),
          reinterpret_cast<half*>(x_out_dev.p),
          reinterpret_cast<half*>(mixed_dev.p),
          kEps);
      rwkv_test::require_cuda(cudaDeviceSynchronize(), "sync cmix non4096");
      expect_vector_near(copy_half_buffer(x_out_dev, "copy x_out"), expect_x_out, 2.5e-2f);
      expect_vector_near(copy_half_buffer(mixed_dev, "copy mixed"), expect_mixed, 2.5e-2f);
      expect_vector_near(copy_half_buffer(shift_dev, "copy shift_out"), expect_shift, 2.5e-2f);
    }

    {
      const int rows = 2;
      std::vector<float> x(rows * kChannels);
      std::vector<float> residual(rows * kChannels);
      std::vector<float> shift_state(rows * kChannels);
      std::vector<float> weight(kChannels);
      std::vector<float> bias(kChannels);
      std::vector<float> x_r(kChannels);
      std::vector<float> x_w(kChannels);
      std::vector<float> x_k(kChannels);
      std::vector<float> x_v(kChannels);
      std::vector<float> x_a(kChannels);
      std::vector<float> x_g(kChannels);
      for (int i = 0; i < rows * kChannels; ++i) {
        x[i] = pattern_value(i + 101, 0.18f, -0.04f);
        residual[i] = pattern_value(i + 203, 0.12f, 0.02f);
        shift_state[i] = pattern_value(i + 307, 0.22f, 0.01f);
      }
      for (int c = 0; c < kChannels; ++c) {
        weight[c] = pattern_value(c + 11, 0.06f, 1.0f);
        bias[c] = pattern_value(c + 19, 0.02f, 0.0f);
        x_r[c] = pattern_value(c + 23, 0.09f, 0.5f);
        x_w[c] = pattern_value(c + 29, 0.07f, 0.4f);
        x_k[c] = pattern_value(c + 31, 0.05f, 0.3f);
        x_v[c] = pattern_value(c + 37, 0.08f, 0.2f);
        x_a[c] = pattern_value(c + 41, 0.06f, 0.1f);
        x_g[c] = pattern_value(c + 43, 0.04f, 0.6f);
      }
      std::vector<float> expect_x_out;
      std::vector<float> expect_out_r;
      std::vector<float> expect_out_w;
      std::vector<float> expect_out_k;
      std::vector<float> expect_out_v;
      std::vector<float> expect_out_a;
      std::vector<float> expect_out_g;
      std::vector<float> expect_shift;
      cpu_add_layer_norm_tmix_mix6(
          rows, kChannels, x, residual, shift_state, weight, bias,
          x_r, x_w, x_k, x_v, x_a, x_g,
          expect_x_out, expect_out_r, expect_out_w, expect_out_k,
          expect_out_v, expect_out_a, expect_out_g, expect_shift);

      DeviceBuffer<std::uint16_t> x_dev;
      DeviceBuffer<std::uint16_t> residual_dev;
      DeviceBuffer<std::uint16_t> shift_dev;
      DeviceBuffer<std::uint16_t> weight_dev;
      DeviceBuffer<std::uint16_t> bias_dev;
      DeviceBuffer<std::uint16_t> xr_dev;
      DeviceBuffer<std::uint16_t> xw_dev;
      DeviceBuffer<std::uint16_t> xk_dev;
      DeviceBuffer<std::uint16_t> xv_dev;
      DeviceBuffer<std::uint16_t> xa_dev;
      DeviceBuffer<std::uint16_t> xg_dev;
      DeviceBuffer<std::uint16_t> x_out_dev;
      DeviceBuffer<std::uint16_t> out_r_dev;
      DeviceBuffer<std::uint16_t> out_w_dev;
      DeviceBuffer<std::uint16_t> out_k_dev;
      DeviceBuffer<std::uint16_t> out_v_dev;
      DeviceBuffer<std::uint16_t> out_a_dev;
      DeviceBuffer<std::uint16_t> out_g_dev;
      copy_half_to_device(x, x_dev, "alloc tmix x", "copy tmix x");
      copy_half_to_device(residual, residual_dev, "alloc tmix residual", "copy tmix residual");
      copy_half_to_device(shift_state, shift_dev, "alloc tmix shift", "copy tmix shift");
      copy_half_to_device(weight, weight_dev, "alloc tmix weight", "copy tmix weight");
      copy_half_to_device(bias, bias_dev, "alloc tmix bias", "copy tmix bias");
      copy_half_to_device(x_r, xr_dev, "alloc x_r", "copy x_r");
      copy_half_to_device(x_w, xw_dev, "alloc x_w", "copy x_w");
      copy_half_to_device(x_k, xk_dev, "alloc x_k", "copy x_k");
      copy_half_to_device(x_v, xv_dev, "alloc x_v", "copy x_v");
      copy_half_to_device(x_a, xa_dev, "alloc x_a", "copy x_a");
      copy_half_to_device(x_g, xg_dev, "alloc x_g", "copy x_g");
      x_out_dev.resize(x.size(), "alloc tmix x_out");
      out_r_dev.resize(x.size(), "alloc out_r");
      out_w_dev.resize(x.size(), "alloc out_w");
      out_k_dev.resize(x.size(), "alloc out_k");
      out_v_dev.resize(x.size(), "alloc out_v");
      out_a_dev.resize(x.size(), "alloc out_a");
      out_g_dev.resize(x.size(), "alloc out_g");
      rwkv7_v3a_add_layer_norm_tmix_mix6_f16_launch(
          nullptr, rows, kChannels,
          reinterpret_cast<const half*>(x_dev.p),
          reinterpret_cast<const half*>(residual_dev.p),
          reinterpret_cast<half*>(shift_dev.p),
          reinterpret_cast<const half*>(weight_dev.p),
          reinterpret_cast<const half*>(bias_dev.p),
          reinterpret_cast<const half*>(xr_dev.p),
          reinterpret_cast<const half*>(xw_dev.p),
          reinterpret_cast<const half*>(xk_dev.p),
          reinterpret_cast<const half*>(xv_dev.p),
          reinterpret_cast<const half*>(xa_dev.p),
          reinterpret_cast<const half*>(xg_dev.p),
          reinterpret_cast<half*>(x_out_dev.p),
          reinterpret_cast<half*>(out_r_dev.p),
          reinterpret_cast<half*>(out_w_dev.p),
          reinterpret_cast<half*>(out_k_dev.p),
          reinterpret_cast<half*>(out_v_dev.p),
          reinterpret_cast<half*>(out_a_dev.p),
          reinterpret_cast<half*>(out_g_dev.p),
          kEps);
      rwkv_test::require_cuda(cudaDeviceSynchronize(), "sync tmix non4096");
      expect_vector_near(copy_half_buffer(x_out_dev, "copy tmix x_out"), expect_x_out, 2.5e-2f);
      expect_vector_near(copy_half_buffer(out_r_dev, "copy out_r"), expect_out_r, 2.5e-2f);
      expect_vector_near(copy_half_buffer(out_w_dev, "copy out_w"), expect_out_w, 2.5e-2f);
      expect_vector_near(copy_half_buffer(out_k_dev, "copy out_k"), expect_out_k, 2.5e-2f);
      expect_vector_near(copy_half_buffer(out_v_dev, "copy out_v"), expect_out_v, 2.5e-2f);
      expect_vector_near(copy_half_buffer(out_a_dev, "copy out_a"), expect_out_a, 2.5e-2f);
      expect_vector_near(copy_half_buffer(out_g_dev, "copy out_g"), expect_out_g, 2.5e-2f);
      expect_vector_near(copy_half_buffer(shift_dev, "copy tmix shift"), expect_shift, 2.5e-2f);
    }

    {
      const int batch = 2;
      const int steps = 3;
      std::vector<float> x(batch * steps * kChannels);
      std::vector<float> residual(batch * steps * kChannels);
      std::vector<float> weight(kChannels);
      std::vector<float> bias(kChannels);
      for (int i = 0; i < batch * steps * kChannels; ++i) {
        x[i] = pattern_value(i + 401, 0.17f, -0.01f);
        residual[i] = pattern_value(i + 503, 0.11f, 0.05f);
      }
      for (int c = 0; c < kChannels; ++c) {
        weight[c] = pattern_value(c + 47, 0.05f, 1.0f);
        bias[c] = pattern_value(c + 59, 0.015f, -0.02f);
      }
      std::vector<float> expect_y;
      cpu_add_last_layer_norm(batch, steps, kChannels, x, residual, weight, bias, expect_y);

      DeviceBuffer<std::uint16_t> x_dev;
      DeviceBuffer<std::uint16_t> residual_dev;
      DeviceBuffer<std::uint16_t> weight_dev;
      DeviceBuffer<std::uint16_t> bias_dev;
      DeviceBuffer<std::uint16_t> y_dev;
      copy_half_to_device(x, x_dev, "alloc final x", "copy final x");
      copy_half_to_device(residual, residual_dev, "alloc final residual", "copy final residual");
      copy_half_to_device(weight, weight_dev, "alloc final weight", "copy final weight");
      copy_half_to_device(bias, bias_dev, "alloc final bias", "copy final bias");
      y_dev.resize(expect_y.size(), "alloc final y");
      rwkv7_v3a_add_last_layer_norm_f16_launch(
          nullptr, batch, steps, kChannels,
          reinterpret_cast<const half*>(x_dev.p),
          reinterpret_cast<const half*>(residual_dev.p),
          reinterpret_cast<const half*>(weight_dev.p),
          reinterpret_cast<const half*>(bias_dev.p),
          reinterpret_cast<half*>(y_dev.p),
          kEps);
      rwkv_test::require_cuda(cudaDeviceSynchronize(), "sync final ln non4096");
      expect_vector_near(copy_half_buffer(y_dev, "copy final y"), expect_y, 2.5e-2f);
    }

    {
      const int vocab = 4;
      std::vector<float> emb(vocab * kChannels);
      std::vector<float> weight(kChannels);
      std::vector<float> bias(kChannels);
      for (int i = 0; i < vocab * kChannels; ++i) {
        emb[i] = pattern_value(i + 601, 0.14f, 0.02f);
      }
      for (int c = 0; c < kChannels; ++c) {
        weight[c] = pattern_value(c + 71, 0.04f, 1.0f);
        bias[c] = pattern_value(c + 79, 0.02f, 0.0f);
      }
      std::vector<float> expect_out;
      cpu_emb_ln0(vocab, kChannels, emb, weight, bias, expect_out);

      DeviceBuffer<std::uint16_t> emb_dev;
      DeviceBuffer<std::uint16_t> weight_dev;
      DeviceBuffer<std::uint16_t> bias_dev;
      DeviceBuffer<std::uint16_t> out_dev;
      copy_bf16_to_device(emb, emb_dev, "alloc emb bf16", "copy emb bf16");
      copy_bf16_to_device(weight, weight_dev, "alloc ln0 weight bf16", "copy ln0 weight bf16");
      copy_bf16_to_device(bias, bias_dev, "alloc ln0 bias bf16", "copy ln0 bias bf16");
      out_dev.resize(expect_out.size(), "alloc emb out");
      rwkv7_v4_emb_ln0_bf16_to_f16_launch(
          nullptr, vocab, kChannels, emb_dev.p, weight_dev.p, bias_dev.p, out_dev.p, kEps);
      rwkv_test::require_cuda(cudaDeviceSynchronize(), "sync emb ln0 non4096");
      expect_vector_near(copy_half_buffer(out_dev, "copy emb out"), expect_out, 3.0e-2f);
    }

    std::cout << "rwkv_cuda_non4096_kernels_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "rwkv_cuda_non4096_kernels_test failed: " << e.what() << "\n";
    return 1;
  }
}
