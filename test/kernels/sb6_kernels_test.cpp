#include "rwkv_asym_a16.cuh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
template<class T>struct Buffer{T* p=nullptr;explicit Buffer(std::size_t n){check(cudaMalloc(&p,n*sizeof(T)));}~Buffer(){cudaFree(p);}Buffer(const Buffer&)=delete;};
void run(int bits,int M,int K,int N){
  const int groups=(K+31)/32,stride=16+4*(bits-4);
  std::mt19937 gen(1234+K+M);std::uniform_real_distribution<float>d(-1,1);
  std::vector<unsigned char> q(std::size_t(N)*groups*stride,0);
  std::vector<half2> sm(std::size_t(N)*groups);
  std::vector<half>x(std::size_t(M)*K);
  std::vector<float>w(std::size_t(N)*K);
  for(auto& v:x)v=__float2half(d(gen));
  for(int n=0;n<N;++n)for(int g=0;g<groups;++g){
    const float s=__half2float(__float2half(.003*(1+(g+n)%5)));
    const float m=__half2float(__float2half((g+n)%2?.13:-.21));
    sm[std::size_t(n)*groups+g]=__floats2half2_rn(s,m);
    auto* p=q.data()+(std::size_t(n)*groups+g)*stride;
    for(int j=0;j<32;++j){
      const unsigned v=(j+13*g+7*n)&((1<<bits)-1);
      p[j/2]|=(v&15)<<((j&1)*4);
      for(int b=4;b<bits;++b)p[16+(b-4)*4+j/8]|=((v>>b)&1)<<(j%8);
      if(g*32+j<K)w[std::size_t(n)*K+g*32+j]=M<=4?std::fma(float(v),s,m):__half2float(__float2half(std::fma(float(v),s,m)));
    }
  }
  std::vector<double> ref(std::size_t(M)*N),mag(ref.size());
  for(int m=0;m<M;++m)for(int n=0;n<N;++n)for(int k=0;k<K;++k){
    const double v=double(__half2float(x[std::size_t(m)*K+k]))*w[std::size_t(n)*K+k];
    ref[std::size_t(m)*N+n]+=v;mag[std::size_t(m)*N+n]+=std::fabs(v);
  }
  Buffer<unsigned char>dq(q.size());Buffer<half2>ds(sm.size());Buffer<half>dx(x.size()),y0(ref.size()),y1(ref.size());
  const auto bytes=rwkv7_asym_workspace_bytes(M,N,32);Buffer<float>p0(bytes/4),p1(bytes/4);
  check(cudaMemcpy(dq.p,q.data(),q.size(),cudaMemcpyHostToDevice));check(cudaMemcpy(ds.p,sm.data(),sm.size()*sizeof(half2),cudaMemcpyHostToDevice));check(cudaMemcpy(dx.p,x.data(),x.size()*sizeof(half),cudaMemcpyHostToDevice));
  // Pageable H2D copies may still be in flight; nonblocking streams do not
  // inherit the default stream dependency. Publish inputs before both readers.
  check(cudaDeviceSynchronize());
  cudaStream_t streams[2];for(auto& s:streams)check(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
  int current_split=-1;
  auto verify=[&](half* p){std::vector<half>v(ref.size());check(cudaMemcpy(v.data(),p,v.size()*2,cudaMemcpyDeviceToHost));
    for(std::size_t i=0;i<v.size();++i)if(std::fabs(__half2float(v[i])-ref[i])>.0006*std::fabs(ref[i])+2e-6*mag[i]+.002){
      std::cerr<<"bits/M/K/N/index="<<bits<<'/'<<M<<'/'<<K<<'/'<<N<<'/'<<i<<" split="<<current_split<<" got="<<__half2float(v[i])<<" ref="<<ref[i]<<'\n';throw std::runtime_error("SB6 reference mismatch");}};
  const int tiles=(K+(M<=4?255:63))/(M<=4?256:64),splits=std::min(3,tiles);
  for(int split:{1,splits,0}){
    current_split=split;
    rwkv7_asym_linear_launch(streams[0],M,K,N,dx.p,dq.p,ds.p,y0.p,bits,p0.p,bytes,split);
    rwkv7_asym_linear_launch(streams[1],M,K,N,dx.p,dq.p,ds.p,y1.p,bits,p1.p,bytes,split);
    for(auto s:streams)check(cudaStreamSynchronize(s));verify(y0.p);verify(y1.p);
  }
  // Exercise both the no-workspace fallback and the exact-size split workspace
  // (no room for activation sums), in addition to the cached path above.
  rwkv7_asym_linear_launch(streams[0],M,K,N,dx.p,dq.p,ds.p,y0.p,bits);
  rwkv7_asym_linear_launch(streams[1],M,K,N,dx.p,dq.p,ds.p,y1.p,bits,p1.p,
                          rwkv7_asym_workspace_bytes(M,N,splits),splits);
  for(auto s:streams)check(cudaStreamSynchronize(s));verify(y0.p);verify(y1.p);
  cudaGraph_t graph;cudaGraphExec_t exec;
  check(cudaStreamBeginCapture(streams[0],cudaStreamCaptureModeThreadLocal));
  rwkv7_asym_linear_launch(streams[0],M,K,N,dx.p,dq.p,ds.p,y0.p,bits,p0.p,bytes,splits);
  check(cudaStreamEndCapture(streams[0],&graph));check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
  check(cudaGraphLaunch(exec,streams[0]));check(cudaGraphLaunch(exec,streams[0]));check(cudaStreamSynchronize(streams[0]));verify(y0.p);
  check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));
  if(splits>1){bool rejected=false;try{rwkv7_asym_linear_launch(nullptr,M,K,N,dx.p,dq.p,ds.p,y0.p,bits,p0.p,rwkv7_asym_workspace_bytes(M,N,splits)-1,splits);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"undersized workspace accepted");}
  for(auto s:streams)check(cudaStreamDestroy(s));
}
}
int main(int argc,char** argv){int n=0;if(cudaGetDeviceCount(&n)!=cudaSuccess||!n)return 77;
try{if(argc>1){for(int j=0;j<20;++j)for(int b:{4,6})run(b,129,4096,64);return 0;}for(int b:{4,6}){
  for(int m:{1,2,3,4,5,8,16,17,32,33,64,65,129})run(b,m,257,67);
  for(int m:{1,2,4,5,16,32,64,129})run(b,m,4096,64);
  for(int k:{1,2,31,32,33,63,64,128,255,256,300}){run(b,1,k,3);run(b,17,k,65);}
  run(b,4,16384,64);run(b,16,16384,64);
}
std::cout<<"SB6 4/6-bit codes, GEMV, register MMA, tails, split-K, streams and graphs passed\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
