#include "rwkv/io/pth_tensor.hpp"
#include "rwkv/runtime/rwkv_adapter.hpp"
#include "rwkv/runtime/rwkv_server_backend.hpp"
#include "test_common.hpp"
#include <cmath>
#include <future>
#include <iostream>
#include <thread>
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
  std::vector<half> x(rows * in), d(out * rank), g(rows * out), y(rows * out),
      dx(rows * in);
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = __float2half(.15f * std::sin(i * .11f));
  for (size_t i = 0; i < d.size(); ++i)
    d[i] = __float2half(.07f * std::cos(i * .21f));
  for (size_t i = 0; i < g.size(); ++i) {
    g[i] = __float2half(.2f * std::sin(i * .07f));
    y[i] = __float2half(.01f);
  }
  for (auto &v : dx)
    v = __float2half(.02f);
  DeviceBuffer<half> X, D, G, Y, Yf, DX;
  DeviceBuffer<float> S, DD;
  upload(X, x);
  upload(D, d);
  upload(G, g);
  upload(Y, y);
  upload(Yf, y);
  upload(DX, dx);
  S.resize(rows * rank, "S");
  DD.resize(out * rank, "dD");
  DD.zero("zero");
  Linear w{in, out, rank, .7f, D.p, DD.p};
  forward(nullptr, rows, w, X.p, Y.p, S.p);
  forward(nullptr, rows, w, X.p, Yf.p);
  backward(nullptr, rows, w, S.p, G.p, DX.p);
  auto a = read(Y), f = read(Yf), b = read(DX);
  auto c = read(DD);
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
  for (int o = 0; o < out; ++o)
    for (int j = 0; j < rank; ++j) {
      double v = 0;
      for (int t = 0; t < rows; ++t)
        v += .7f * __half2float(g[t * out + o]) * s[t * rank + j];
      near(c[o * rank + j], v, 2e-5);
    }
  for (int t = 0; t < rows; ++t)
    for (int k = 0; k < in; ++k) {
      double v = __half2float(dx[t * in + k]);
      for (int o = 0; o < out; ++o)
        v += .7f * __half2float(g[t * out + o]) *
             __half2float(d[o * rank + k % rank]);
      near(__half2float(b[t * in + k]), v, 2e-4);
    }
  backward(nullptr, rows, w, S.p, G.p, DX.p);
  auto twice = read(DD);
  for (size_t i = 0; i < c.size(); ++i)
    near(twice[i], 2 * c[i], 1e-6);
  // Finite differences of the continuous explicit shard expansion objective.
  auto objective = [&](double delta) {
    double v = 0;
    for (int t = 0; t < rows; ++t)
      v += .7f * __half2float(g[t * out]) * s[t * rank] *
           (double(__half2float(d[0])) + delta);
    return v;
  };
  near((objective(1e-5) - objective(-1e-5)) / 2e-5, c[0], 2e-5);
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
  DeviceBuffer<half> X, D, G, Y, DX;
  DeviceBuffer<float> S, DD;
  upload(X, halfdata("X"));
  upload(D, halfdata("D"));
  upload(G, halfdata("G"));
  upload(Y, halfdata("base_Y"));
  upload(DX, halfdata("base_dX"));
  S.resize(rows * rank, "S");
  DD.resize(out * rank, "DD");
  DD.zero("zero");
  Linear w{in, out, rank, 1, D.p, DD.p};
  forward(nullptr, rows, w, X.p, Y.p, S.p);
  backward(nullptr, rows, w, S.p, G.p, DX.p);
  auto y = read(Y), dx = read(DX);
  auto dd = read(DD);
  for (size_t i = 0; i < y.size(); ++i)
    near(__half2float(y[i]), h.at("Y").values[i], .002);
  for (size_t i = 0; i < dx.size(); ++i)
    near(__half2float(dx[i]), h.at("dX").values[i], .002);
  for (size_t i = 0; i < dd.size(); ++i)
    near(dd[i], h.at("dD").values[i], .0001);
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
