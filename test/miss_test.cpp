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
void package_roundtrip() {
  auto dir = rwkv_test::unique_temp_path("miss_cache");
  Json::Value m;
  m["rank"] = 4;
  m["alpha"] = 4;
  m["scale"] = 1;
  m["base_fingerprint"] = std::string(64, 'a');
  std::vector<HostTensor> ts{{0, 0, 8, 8, 4, 0}, {1, 5, 16, 8, 4, 32}};
  std::vector<uint16_t> d(64, 0x3000);
  save_package(dir.string(), m, ts, d);
  auto package = load_package(dir.string());
  TEST_EQ(package->data, d);
  TEST_EQ(package->tensors.size(), ts.size());
  TEST_EQ(package->version, content_version(package->manifest, d));
  bool denied = false;
  try { load_package(dir.string(), 64); }
  catch (const std::runtime_error&) { denied = true; }
  TEST_CHECK(denied);
  std::filesystem::remove_all(dir);
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
    package_roundtrip();
    if (argc > 1)
      reference(argv[1]);
    std::cout << "MiSS numerical/package tests passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
