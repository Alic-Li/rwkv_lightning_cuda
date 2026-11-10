#include <algorithm>
#include <cmath>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "rwkv7_fast_v4_kernels.cuh"
#include "utils/sampling.h"

namespace {
void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
template<class T> struct Buffer {
  T* p; std::size_t size;
  explicit Buffer(std::size_t n): size(n) { check(cudaMalloc(&p, n * sizeof(T))); }
  explicit Buffer(const std::vector<T>& v): Buffer(v.size()) { set(v); }
  ~Buffer() { cudaFree(p); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void set(const std::vector<T>& v) {
    require(v.size() == size, "buffer size");
    check(cudaMemcpy(p, v.data(), size * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> get() const {
    std::vector<T> v(size);
    check(cudaMemcpy(v.data(), p, size * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }
};
std::vector<half> pattern(std::size_t n, float scale = 0.02f) {
  std::vector<half> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = __float2half((int((i * 17 + i / 13) % 23) - 11) * scale);
  return v;
}
void near(const std::vector<half>& a, const std::vector<half>& b, float tol, const char* msg) {
  require(a.size() == b.size(), msg);
  for (std::size_t i = 0; i < a.size(); ++i)
    require(std::fabs(__half2float(a[i]) - __half2float(b[i])) <= tol, msg);
}
void sampling() {
  constexpr int B = 3, T = 5, V = 4100;
  std::vector<float> logits(B*T*V, -100.0f);
  for (int b = 0; b < B; ++b) logits[(b*T+T-1)*V+V-1-b] = 100.0f;
  Buffer<float> x(logits), probs(B*V), penalties(std::vector<float>(B*V, 0));
  Buffer<int> out(B);
  Buffer<unsigned char> rng(rwkv_sampling::rand_state_bytes(B));
  check(rwkv_sampling::setup_rand_raw(rng.p, 991, B, nullptr));
  for (int mode = 0; mode < 3; ++mode) {
    if (mode == 0)
      check(rwkv_sampling::batch_sampling_repetition_temperature_topk_topp_raw(
          x.p, penalties.p, out.p, rng.p, probs.p, B,T,V, 1,1,1, 0.8, 1,1, nullptr));
    else
      check(rwkv_sampling::batch_sampling_temperature_topk_topp_raw(
          x.p, out.p, rng.p, probs.p, B,T,V, mode == 1 ? 0.8 : 1, mode == 1 ? 1 : V,1, nullptr));
    const auto values = out.get();
    for (int b = 0; b < B; ++b) require(values[b] == V-1-b, "sampling last time step / scratch stride");
  }
}
void wkv(int B, int T) {
  constexpr int H = 2, C = H*64;
  const auto input = pattern(std::size_t(B)*T*C);
  const auto initial = pattern(std::size_t(B)*H*64*64, 0.001f);
  Buffer<half> x(input), state(initial), reference_state(initial), y(input.size()), reference_y(input.size());
  Buffer<int> elapsed(std::vector<int>(B, 0));
  rwkv7_wkv_fp16_seq_launch(nullptr,B,T,C,H,state.p,x.p,x.p,x.p,x.p,x.p,x.p,y.p,elapsed.p);
  // Independent sequence/decode launches must give the same recurrence.
  for (int b = 0; b < B; ++b) for (int t = 0; t < T; ++t) {
    elapsed.set(std::vector<int>(B, t));
    const auto offset = (std::size_t(b)*T+t)*C;
    rwkv7_wkv_fp16_one_launch(nullptr,1,C,H,reference_state.p+std::size_t(b)*H*64*64,
        x.p+offset,x.p+offset,x.p+offset,x.p+offset,x.p+offset,x.p+offset,reference_y.p+offset,elapsed.p);
  }
  near(y.get(), reference_y.get(), 0.002f, "FP16 WKV sequence/decode");
  near(state.get(), reference_state.get(), 0.002f, "FP16 WKV state");
  std::vector<float> initial32(initial.size());
  for (std::size_t i=0; i<initial.size(); ++i) initial32[i]=__half2float(initial[i]);
  Buffer<float> state32(initial32), ref32(initial32);
  rwkv7_wkv_fp32io16_launch(nullptr,B,T,C,H,1,ref32.p,x.p,x.p,x.p,x.p,x.p,x.p,reference_y.p);
  const auto ref_out = reference_y.get();
  const auto ref_state = ref32.get();
  for (int mode : {0,2,3}) {
    state32.set(initial32);
    rwkv7_wkv_fp32io16_launch(nullptr,B,T,C,H,mode,state32.p,x.p,x.p,x.p,x.p,x.p,x.p,y.p);
    near(y.get(), ref_out, 0.001f, "FP32 WKV modes");
    const auto got=state32.get();
    for (std::size_t i=0;i<got.size();++i) require(std::fabs(got[i]-ref_state[i])<1e-5f,"FP32 WKV state");
  }
}
void sparse(int C, int F, int rows) {
  auto input=pattern(std::size_t(rows)*F);
  auto weight=pattern(std::size_t(F)*C);
  Buffer<half> x(input), w(weight), y(std::size_t(rows)*C), ref(std::size_t(rows)*C);
  rwkv7_cmix_sparse_down_relu_rows_launch(nullptr,rows,1,C,F,x.p,w.p,ref.p);
  rwkv7_cmix_sparse_down_relu_rows_t512_launch(nullptr,rows,1,C,F,x.p,w.p,y.p);
  near(y.get(),ref.get(),0.004f,"sparse FP16 tile fallback");
  std::vector<std::int8_t> q(weight.size());
  for (std::size_t i=0;i<q.size();++i) q[i]=int(i%11)-5;
  Buffer<std::int8_t> qw(q);
  Buffer<half> scale(std::vector<half>(C,__float2half(0.01f)));
  rwkv7_cmix_sparse_down_relu_rows_i8_launch(nullptr,rows,1,C,F,x.p,qw.p,scale.p,ref.p);
  rwkv7_cmix_sparse_down_relu_rows_t512_i8_launch(nullptr,rows,1,C,F,x.p,qw.p,scale.p,y.p);
  near(y.get(),ref.get(),0.004f,"sparse INT8 tile fallback");
}
void odd_linear() {
  constexpr int M=3,K=65,N=7;
  const auto input=pattern(M*K), weights=pattern(N*K);
  Buffer<half> x(input),w(weights),y(M*N),ref(M*N);
  rwkv7_v3a_linear_f16_orig_launch(nullptr,M,K,N,x.p,w.p,ref.p);
  rwkv7_v3a_linear_t_f16_launch(nullptr,M,K,N,x.p,w.p,y.p);
  near(y.get(),ref.get(),0.002f,"odd K transposed linear");
  rwkv7_v3a_linear_orig_rows_cfg_f16_launch(nullptr,M,K,N,x.p,w.p,64,3,4,y.p);
  near(y.get(),ref.get(),0.002f,"odd K tiled projection");
  Buffer<half> y1(M*N),y2(M*N),y3(M*N);
  rwkv7_v3a_linear_wagv_rank_in_f16_launch(nullptr,M,K,N,N,N,N,x.p,x.p,x.p,x.p,w.p,w.p,w.p,w.p,
      y.p,y1.p,y2.p,y3.p);
  for (auto output : {&y,&y1,&y2,&y3}) near(output->get(),ref.get(),0.002f,"odd K rank projection");
}
void odd_elementwise() {
  const auto input=pattern(65);
  Buffer<half> x(input), y(65), v(std::vector<half>(5,__float2half(1.0f)));
  for (int op=0;op<4;++op) {
    if (op==0) rwkv7_relu_square_launch(nullptr,x.p,y.p,65);
    if (op==1) rwkv7_act_tanh_launch(nullptr,x.p,y.p,65);
    if (op==2) rwkv7_act_sigmoid_launch(nullptr,x.p,y.p,65);
    if (op==3) rwkv7_add_vec_launch(nullptr,5,x.p,v.p,y.p,65);
    const auto got=y.get();
    for (int i=0;i<65;++i) {
      const float f=__half2float(input[i]);
      const float expected=op==0 ? std::max(f,0.0f)*std::max(f,0.0f) :
          op==1 ? std::tanh(f) : op==2 ? 1.0f/(1.0f+std::exp(-f)) : f+1;
      require(std::fabs(__half2float(got[i])-expected)<0.001f,"odd elementwise tail");
    }
  }
}
void normalization_tiles() {
  constexpr int C=4096;
  const auto row=pattern(C);
  const auto weight=pattern(C,0.03f),bias=pattern(C,0.01f);
  double mean=0,variance=0;
  for (half v:row) mean+=__half2float(v);
  mean/=C;
  for (half v:row) { const double d=__half2float(v)-mean; variance+=d*d; }
  variance/=C;
  std::vector<half> expected(C);
  for (int c=0;c<C;++c) expected[c]=__float2half(
      (__half2float(row[c])-mean)/std::sqrt(variance+1e-5)*__half2float(weight[c])+__half2float(bias[c]));
  Buffer<half> w(weight),b(bias);
  for (int rows : {1,512,1024}) {
    std::vector<half> input(std::size_t(rows)*C);
    for (int r=0;r<rows;++r) std::copy(row.begin(),row.end(),input.begin()+std::size_t(r)*C);
    Buffer<half> x(input),residual(std::vector<half>(input.size(),__float2half(0))),y(input.size());
    for (int fused=0;fused<2;++fused) {
      if (fused) rwkv7_v3a_add_layer_norm_f16_launch(nullptr,rows,C,x.p,residual.p,w.p,b.p,x.p,y.p,1e-5f);
      else rwkv7_v3a_layer_norm_f16_launch(nullptr,rows,C,x.p,w.p,b.p,y.p,1e-5f);
      const auto got=y.get();
      for (std::size_t i=0;i<got.size();++i)
        require(std::fabs(__half2float(got[i])-__half2float(expected[i%C]))<0.002f,"4096-channel norm tile");
    }
  }
}
void quantized_batch() {
  constexpr int K=320;
  for (int N : {128,192}) for (int M : {1,3,8,24,32,33,64,65,129,257,1024}) {
    Buffer<half> x(std::vector<half>(M*K,__float2half(0.125f))), y(M*N),
        scales(std::vector<half>(N,__float2half(0.125f)));
    Buffer<std::int8_t> q(std::vector<std::int8_t>(N*K,1)), packed(N*K);
    Buffer<float> workspace(M*N);
    rwkv7_v4_i8_pack_launch(nullptr,q.p,packed.p,N,K);
    for (auto layout : {W8BLayout::NK,W8BLayout::KN,W8BLayout::PackedNK}) {
      for (int split : {0,1,3,5}) {
        rwkv7_w8a16_linear_launch(nullptr,M,K,N,x.p,
            layout == W8BLayout::PackedNK ? packed.p : q.p,scales.p,layout,
            y.p,workspace.p,M*N*sizeof(float),split);
        for (half v:y.get()) require(__half2float(v)==5.0f,"W8 tail batch/split K");
      }
    }
  }
}
void concurrent_linear() {
  std::vector<std::future<void>> jobs;
  for (int job=0;job<4;++job) jobs.emplace_back(std::async(std::launch::async,[job] {
    check(cudaSetDevice(0));
    cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    constexpr int M=33,K=256,N=128;
    Buffer<half> x(std::vector<half>(M*K,__float2half(0.125f*(job+1)))),
        w(std::vector<half>(N*K,__float2half(0.125f))),y(M*N),s(std::vector<half>(N,__float2half(0.125f)));
    Buffer<std::int8_t> q(std::vector<std::int8_t>(N*K,1));
    Buffer<float> workspace(M*N);
    for (int i=0;i<20;++i) {
      rwkv7_w8a16_linear_launch(stream,M,K,N,x.p,q.p,s.p,W8BLayout::KN,y.p,workspace.p,M*N*sizeof(float),3);
      rwkv7_v3a_linear_f16_orig_launch(stream,M,K,N,x.p,w.p,y.p);
    }
    check(cudaStreamSynchronize(stream));
    for (half v:y.get()) require(__half2float(v)==4.0f*(job+1),"concurrent BLAS stream routing");
    check(cudaStreamDestroy(stream));
  }));
  for (auto& job:jobs) job.get();
}
}
int main() {
  try {
    int count=0; if (cudaGetDeviceCount(&count)!=cudaSuccess || !count) return 77;
    sampling();
    for (auto bt : {std::pair<int,int>{1,8},{4,4},{3,3},{65,1},{129,1}}) wkv(bt.first,bt.second);
    sparse(256,512,2); sparse(768,384,2); sparse(512,1024,2);
    odd_linear(); odd_elementwise(); normalization_tiles(); quantized_batch(); concurrent_linear();
    std::cout << "kernel safety regressions passed\n";
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
