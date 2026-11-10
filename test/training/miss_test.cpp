#include "rwkv/io/pth_tensor.hpp"
#include "rwkv/runtime/rwkv_adapter.hpp"
#include "rwkv/runtime/rwkv_server_backend.hpp"
#include "test_common.hpp"
#include <cmath>
#include <future>
#include <iostream>
#include <thread>
#include <utility>
using rwkv7_fast_v4::DeviceBuffer;
using namespace rwkv7_miss;
namespace {
void check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
template <class T> void upload(DeviceBuffer<T> &d, const std::vector<T> &h) {
  d.resize(h.size(), "test");
  check(
      cudaMemcpy(d.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}
template <class T> std::vector<T> read(const DeviceBuffer<T> &d) {
  std::vector<T> h(d.n);
  check(cudaMemcpy(h.data(), d.p, d.n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}
void near(double a, double b, double tol) {
  if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a - b) > tol)
    throw std::runtime_error("MiSS numerical mismatch: " + std::to_string(a) +
                             " / " + std::to_string(b));
}
void numerical(int rows, int in, int out, int rank) {
  std::vector<half> x(rows * in), d(out * rank), y(rows * out);
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = __float2half(.15f * std::sin(i * .11f));
  for (size_t i = 0; i < d.size(); ++i)
    d[i] = __float2half(.07f * std::cos(i * .21f));
  for (auto &v : y)
    v = __float2half(.01f);
  DeviceBuffer<half> X, D, Y, Yf;
  DeviceBuffer<float> S;
  upload(X, x);
  upload(D, d);
  upload(Y, y);
  upload(Yf, y);
  S.resize(rows * rank, "S");
  Linear w{in, out, rank, .7f, D.p};
  forward(nullptr, rows, w, X.p, Y.p, S.p);
  forward(nullptr, rows, w, X.p, Yf.p);
  auto a = read(Y), f = read(Yf);
  std::vector<double> s(rows * rank);
  for (int t = 0; t < rows; ++t)
    for (int k = 0; k < in; ++k)
      s[t * rank + k % rank] += __half2float(x[t * in + k]);
  for (int t = 0; t < rows; ++t)
    for (int o = 0; o < out; ++o) {
      double v = __half2float(y[t * out + o]);
      for (int j = 0; j < rank; ++j)
        v += .7f * s[t * rank + j] * __half2float(d[o * rank + j]);
      near(__half2float(a[t * out + o]), v, 2e-4);
      near(__half2float(f[t * out + o]), __half2float(a[t * out + o]), 0);
    }
  Linear none;
  auto before = read(Y);
  forward(nullptr, rows, none, nullptr, Y.p);
  auto after = read(Y);
  TEST_CHECK(std::memcmp(before.data(), after.data(), before.size() * 2) == 0);
}
void reference(const std::string &path) {
  auto a = llm_infer::PthArchive::open(path);
  TEST_CHECK(a.ok());
  auto rs = llm_infer::parse_pth_tensor_records(a.value());
  TEST_CHECK(rs.ok());
  std::map<std::string, llm_infer::TensorData> h;
  for (auto &r : rs.value()) {
    auto d = llm_infer::load_tensor_select(a.value(), r, false, true, true);
    TEST_CHECK(d.ok());
    h[r.name] = std::move(d.value());
  }
  int rows = h.at("X").shape[0], in = h.at("X").shape[1],
      out = h.at("D").shape[0], rank = h.at("D").shape[1];
  auto halfdata = [&](const std::string &n) {
    std::vector<half> v(h.at(n).values_f16.size());
    std::memcpy(v.data(), h.at(n).values_f16.data(), v.size() * 2);
    return v;
  };
  DeviceBuffer<half> X, D, Y;
  upload(X, halfdata("X"));
  upload(D, halfdata("D"));
  upload(Y, halfdata("base_Y"));
  Linear w{in, out, rank, 1, D.p};
  forward(nullptr, rows, w, X.p, Y.p);
  auto y = read(Y);
  for (size_t i = 0; i < y.size(); ++i)
    near(__half2float(y[i]), h.at("Y").values[i], .002);
}
void cache() {
  auto dir = rwkv_test::unique_temp_path("miss_cache");
  Json::Value m;
  m["rank"] = 4;
  m["alpha"] = 4;
  m["scale"] = 1;
  m["base_fingerprint"] = std::string(64, 'a');
  std::vector<HostTensor> ts{{0, 0, 8, 8, 4, 0}, {1, 5, 16, 8, 4, 32}};
  std::vector<uint16_t> d(64, 0x3000);
  save_package(dir.string(), m, ts, d);
  AdapterCache too_small(64, 128, 128);
  bool ram_denied = false;
  try {
    too_small.register_adapter("large", dir.string());
  } catch (const std::runtime_error &) {
    ram_denied = true;
  }
  TEST_CHECK(ram_denied);
  TEST_EQ(too_small.stats().ram_bytes, 0);
  AdapterCache staging_small(4096, 128, 64);
  staging_small.register_adapter("large", dir.string());
  auto staging_handle = staging_small.resolve("large");
  bool staging_denied = false;
  try {
    staging_small.acquire(*staging_handle);
  } catch (const std::runtime_error &) {
    staging_denied = true;
  }
  TEST_CHECK(staging_denied);
  TEST_EQ(staging_small.stats().uploads, 0);
  AdapterCache c(4096, 128, 128);
  auto v = c.register_adapter("one", dir.string());
  TEST_EQ(c.stats().uploads, 0);
  auto h = c.resolve("one");
  std::vector<std::future<std::shared_ptr<const GpuLease>>> jobs;
  for (int i = 0; i < 8; ++i)
    jobs.push_back(
        std::async(std::launch::async, [&] { return c.acquire(*h); }));
  auto lease = jobs[0].get();
  for (size_t i = 1; i < jobs.size(); ++i)
    TEST_CHECK(jobs[i].get().get() == lease.get());
  TEST_EQ(c.stats().uploads, 1);
  TEST_EQ(lease->blocks.size(), 2);
  auto dir2 = dir.string() + "-two";
  d[0] = 0x3400;
  save_package(dir2, m, ts, d);
  c.register_adapter("two", dir2);
  auto h2 = c.resolve("two");
  bool denied = false;
  try {
    c.acquire(*h2);
  } catch (const std::runtime_error &) {
    denied = true;
  }
  TEST_CHECK(denied);
  lease.reset();
  auto lease2 = c.acquire(*h2);
  TEST_EQ(c.stats().evictions, 1);
  lease2.reset();
  c.trim();
  TEST_EQ(c.stats().gpu_bytes, 0);
  auto resumed = c.acquire(*h);
  TEST_EQ(c.stats().uploads, 3);
  rwkv7_server::GenerationState state;
  rwkv7_server::bind_adapter(state, h);
  state.advanced = true;
  bool mismatch = false;
  try {
    rwkv7_server::bind_adapter(state, h2);
  } catch (...) {
    mismatch = true;
  }
  TEST_CHECK(mismatch);
  c.register_adapter("one", dir2);
  TEST_EQ(c.resolve("one")->package->version, h2->package->version);
  c.erase("one", h2->package->version);
  TEST_EQ(c.resolve("one")->package->version, h->package->version);
  c.erase("one");
  TEST_CHECK(c.acquire(*h).get() == resumed.get());
  TEST_CHECK(effective_model_key("base", h.get(), "zero", false) !=
             effective_model_key("base", h2.get(), "zero", false));
  std::filesystem::remove_all(dir);
  std::filesystem::remove_all(dir2);
}
// Packed decode: one forward_rows launch must equal per-row single-adapter
// launches (MiSS, NoRA, base rows, a slot missing the target), with relu2 on
// load equal to materializing relu(x)^2 first.
void packed_rows(int rows, int in, int out, bool relu2) {
  const int miss_rank = 16, nora_rank = 24;
  std::vector<half> x(rows * in), xa(rows * in), y(rows * out);
  std::vector<half> md(out * miss_rank), na(nora_rank * in), nb(out * nora_rank),
      nt(in * nora_rank);
  for (size_t i = 0; i < x.size(); ++i) {
    const float v = .3f * std::sin(i * .37f);
    x[i] = __float2half(v);
    xa[i] = __float2half(relu2 ? (v > 0 ? v * v : 0) : v);
  }
  for (size_t i = 0; i < md.size(); ++i)
    md[i] = __float2half(.05f * std::cos(i * .13f));
  for (size_t i = 0; i < na.size(); ++i)
    na[i] = __float2half(.04f * std::sin(i * .07f));
  for (int j = 0; j < nora_rank; ++j)
    for (int k = 0; k < in; ++k)
      nt[size_t(k) * nora_rank + j] = na[size_t(j) * in + k];
  for (size_t i = 0; i < nb.size(); ++i)
    nb[i] = __float2half(.03f * std::cos(i * .05f));
  for (size_t i = 0; i < y.size(); ++i)
    y[i] = __float2half(.02f * std::sin(i * .3f));
  DeviceBuffer<half> X, XA, MD, NA, NB, NT, Yp, Yr;
  upload(X, x);
  upload(XA, xa);
  upload(MD, md);
  upload(NA, na);
  upload(NB, nb);
  upload(NT, nt);
  upload(Yp, y);
  upload(Yr, y);
  const Linear miss{in, out, miss_rank, .6f, MD.p};
  const Linear nora{in, out, nora_rank, 1.f, NB.p, NA.p};
  // Slot 3 is the lease layout: only the transposed A (first 16-wide tile is
  // vectorized, the 8-wide tail is scalar). Slot 2 lacks this target (null
  // d) and must behave like a base row.
  const Linear nora_t{in, out, nora_rank, 1.f, NB.p, nullptr, NT.p};
  std::vector<Linear> table{miss, nora, Linear{}, nora_t};
  std::vector<int> slot(rows);
  for (int r = 0; r < rows; ++r)
    slot[r] = r % 5 == 4 ? -1 : r % 5;
  DeviceBuffer<Linear> T;
  DeviceBuffer<int> R;
  upload(T, table);
  upload(R, slot);
  DeviceBuffer<float> L;
  L.resize(forward_rows_scratch(rows, in, nora_rank), "latent");
  forward_rows(nullptr, rows, in, out, nora_rank, R.p, T.p, X.p, Yp.p, L.p,
               relu2);
  for (int r = 0; r < rows; ++r)
    if (slot[r] == 0 || slot[r] == 1 || slot[r] == 3)
      forward(nullptr, 1, slot[r] == 3 ? nora : table[slot[r]],
              XA.p + size_t(r) * in, Yr.p + size_t(r) * out);
  check(cudaDeviceSynchronize());
  const auto p = read(Yp), q = read(Yr);
  for (int r = 0; r < rows; ++r)
    for (int o = 0; o < out; ++o) {
      const size_t i = size_t(r) * out + o;
      if (slot[r] < 0 || slot[r] == 2)
        near(__half2float(p[i]), __half2float(y[i]), 0);
      else // NoRA accumulation order differs: allow ~1 FP16 ULP.
        near(__half2float(p[i]), __half2float(q[i]),
             2e-3 + 1.5e-3 * std::abs(__half2float(q[i])));
    }
  // Single-adapter prefill: null row_slot applies table[0] to every row.
  for (const auto &pair : {std::make_pair(miss, miss), std::make_pair(nora_t, nora)}) {
    const Linear &w = pair.first;
    DeviceBuffer<Linear> U;
    upload(U, std::vector<Linear>{w});
    DeviceBuffer<half> Yu, Yv;
    upload(Yu, y);
    upload(Yv, y);
    forward_rows(nullptr, rows, in, out, nora_rank, nullptr, U.p, X.p, Yu.p,
                 L.p, relu2);
    for (int r = 0; r < rows; ++r)
      forward(nullptr, 1, pair.second, XA.p + size_t(r) * in,
              Yv.p + size_t(r) * out);
    check(cudaDeviceSynchronize());
    const auto u = read(Yu), v = read(Yv);
    for (size_t i = 0; i < u.size(); ++i)
      near(__half2float(u[i]), __half2float(v[i]),
           2e-3 + 1.5e-3 * std::abs(__half2float(v[i])));
  }
}
} // namespace
int main(int argc, char **argv) {
  if (!rwkv_test::cuda_device_available())
    return 77;
  try {
    for (auto shape : std::vector<std::array<int, 4>>{{1, 64, 64, 16},
                                                      {7, 65, 33, 16},
                                                      {33, 128, 129, 32},
                                                      {65, 257, 513, 17},
                                                      {5, 7, 9, 16}})
      numerical(shape[0], shape[1], shape[2], shape[3]);
    {
      // Transposed-only NoRA must be rejected by the single-adapter kernel.
      bool rejected = false;
      try {
        forward(nullptr, 1, Linear{4, 4, 2, 1.f, nullptr, nullptr, nullptr}, nullptr, nullptr);
        Linear t{4, 4, 2, 1.f, reinterpret_cast<const half *>(16), nullptr,
                 reinterpret_cast<const half *>(16)};
        forward(nullptr, 1, t, reinterpret_cast<const half *>(16),
                reinterpret_cast<half *>(16));
      } catch (const std::invalid_argument &) {
        rejected = true;
      }
      TEST_CHECK(rejected);
    }
    for (bool relu2 : {false, true})
      for (auto shape : std::vector<std::array<int, 3>>{
               {1, 2048, 8192}, {8, 8192, 2048}, {13, 300, 257}, {64, 2048, 2048}})
        packed_rows(shape[0], shape[1], shape[2], relu2);
    cache();
    if (argc > 1)
      reference(argv[1]);
    std::cout << "MiSS numerical/cache tests passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
