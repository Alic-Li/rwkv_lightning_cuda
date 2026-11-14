#include "rwkv_asym_a16.cuh"
#include <mma.h>
#include <algorithm>
#include <limits>

namespace {
template<int Bits>
__device__ __forceinline__ unsigned code(const unsigned char* group, int k) {
  unsigned v = (group[k / 2] >> ((k & 1) * 4)) & 15;
  if constexpr (Bits == 6) v |= ((*reinterpret_cast<const unsigned*>(group + 16) >> k) & 1) << 4;
  if constexpr (Bits == 6) v |= ((*reinterpret_cast<const unsigned*>(group + 20) >> k) & 1) << 5;
  return v;
}

// Inject both high planes directly into half mantissas; no integer->float
// conversion and no extra floating add per high bit. k is even and < 32.
template<int Bits>
__device__ __forceinline__ half2 pair(const unsigned char* group, int k, half2 sm) {
  const unsigned byte = group[k / 2];
  unsigned bits = 0x64006400u | (byte & 15) | ((byte & 240) << 12);
  if constexpr (Bits == 6) {
    const unsigned h = *reinterpret_cast<const unsigned*>(group + 16) >> k;
    bits |= ((h & 1) << 4) | ((h & 2) << 19);
  }
  if constexpr (Bits == 6) {
    const unsigned h = *reinterpret_cast<const unsigned*>(group + 20) >> k;
    bits |= ((h & 1) << 5) | ((h & 2) << 20);
  }
  const unsigned magic = 0x64006400u;
  const half2 q = __hsub2(*reinterpret_cast<const half2*>(&bits), *reinterpret_cast<const half2*>(&magic));
  return __hfma2(q, __half2half2(__low2half(sm)), __half2half2(__high2half(sm)));
}

template<int Bits, int R>
__global__ void gemv(int M, int K, int N, const half* x, const unsigned char* q,
                     const half2* sm, half* y, float* partial, int splits) {
  __shared__ __align__(16) half sx[R][256];
  __shared__ float sums[R][8];
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  const int n = blockIdx.x * 4 + warp;
  const int groups = (K + 31) / 32, tiles = (groups + 7) / 8;
  const int start = tiles / splits * blockIdx.z + min(int(blockIdx.z), tiles % splits);
  const int end = start + tiles / splits + (int(blockIdx.z) < tiles % splits);
  float acc[R] = {};
  for (int tile = start; tile < end; ++tile) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      for (int i = threadIdx.x; i < 256; i += 128) {
        const int k = tile * 256 + i;
        sx[r][i] = k < K ? x[std::size_t(r) * K + k] : __float2half(0);
      }
    }
    __syncthreads();
    for (int g = warp; g < 8; g += 4) {
#pragma unroll
      for (int r = 0; r < R; ++r) {
        float sum = __half2float(sx[r][g * 32 + lane]);
        for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffff, sum, d);
        if (!lane) sums[r][g] = sum;
      }
    }
    __syncthreads();
    const int local_group = lane / 4, g = tile * 8 + local_group;
    if (n < N && g < groups) {
      const auto index = std::size_t(n) * groups + g;
      const auto* packed = q + index * (16 + 4 * (Bits - 4));
      const unsigned low = *reinterpret_cast<const unsigned*>(packed + (lane & 3) * 4);
      unsigned high = 0, high2 = 0;
      if constexpr (Bits == 6) high = *reinterpret_cast<const unsigned*>(packed + 16) >> ((lane & 3)*8);
      if constexpr (Bits == 6) high2 = *reinterpret_cast<const unsigned*>(packed + 20) >> ((lane & 3)*8);
      const half2 params = sm[index];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const int4 raw_x = *reinterpret_cast<const int4*>(&sx[r][local_group*32+(lane&3)*8]);
        const half* xv = reinterpret_cast<const half*>(&raw_x);
        float dot = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const unsigned v = ((low >> (4*j)) & 15) | (((high >> j)&1)<<4) | (((high2 >> j)&1)<<5);
          dot = fmaf(float(v), __half2float(xv[j]), dot);
        }
        // Four lanes own one 32-element group. The two-stage reduction runs
        // eight independent group dots per warp instead of eight full-warp dots.
        const unsigned active = __activemask();
        dot += __shfl_xor_sync(active, dot, 1, 4);
        dot += __shfl_xor_sync(active, dot, 2, 4);
        if (!(lane & 3)) acc[r] = fmaf(dot, __half2float(__low2half(params)),
                            fmaf(__half2float(__high2half(params)), sums[r][local_group], acc[r]));
      }
    }
    __syncthreads();
  }
  for (int r = 0; r < R; ++r) {
    for (int d = 16; d; d >>= 1) acc[r] += __shfl_down_sync(0xffffffff, acc[r], d);
    if (lane || n >= N) continue;
    const auto i = std::size_t(r) * N + n;
    if (splits == 1) y[i] = __float2half_rn(acc[r]);
    else partial[std::size_t(blockIdx.z) * M * N + i] = acc[r];
  }
}

// Large aligned models reuse one activation sum per group across every output
// channel. Workspace is optional; the shared-memory path above remains the
// allocation-free fallback when the caller supplies only split partials.
__global__ void activation_sums(const half* x, float* sums, int count) {
  const int lane=threadIdx.x&31, group=blockIdx.x*4+threadIdx.x/32;
  if(group>=count)return;
  float v=__half2float(x[std::size_t(group)*32+lane]);
  for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffff,v,d);
  if(!lane)sums[group]=v;
}
template<int Bits,int R>
__global__ void gemv_cached(int M,int K,int N,const half* x,const unsigned char* q,
    const half2* sm,const float* sums,half* y,float* partial,int splits) {
  const int lane=threadIdx.x&31,n=blockIdx.x*4+threadIdx.x/32;
  if(n>=N)return;
  const int tiles=K/256,groups=K/32;
  const int begin=tiles/splits*blockIdx.z+min(int(blockIdx.z),tiles%splits);
  const int end=begin+tiles/splits+(int(blockIdx.z)<tiles%splits);
  float acc[R]={};
  for(int t=begin;t<end;++t) {
    const int g=t*8+lane/4,k=t*256+lane*8;
    const auto index=std::size_t(n)*groups+g;
    const auto* packed=q+index*(16+4*(Bits-4));
    const unsigned low=*reinterpret_cast<const unsigned*>(packed+(lane&3)*4);
    unsigned high=0,high2=0;
    if constexpr(Bits==6)high=*reinterpret_cast<const unsigned*>(packed+16)>>((lane&3)*8);
    if constexpr(Bits==6)high2=*reinterpret_cast<const unsigned*>(packed+20)>>((lane&3)*8);
    const half2 params=sm[index];
#pragma unroll
    for(int r=0;r<R;++r) {
      const int4 raw=*reinterpret_cast<const int4*>(x+std::size_t(r)*K+k);
      const half* xv=reinterpret_cast<const half*>(&raw);
      float dot=0;
#pragma unroll
      for(int j=0;j<8;++j) {
        const unsigned v=((low>>(4*j))&15)|(((high>>j)&1)<<4)|(((high2>>j)&1)<<5);
        dot=fmaf(float(v),__half2float(xv[j]),dot);
      }
      dot+=__shfl_xor_sync(0xffffffff,dot,1,4);
      dot+=__shfl_xor_sync(0xffffffff,dot,2,4);
      if(!(lane&3))acc[r]=fmaf(dot,__half2float(__low2half(params)),
          fmaf(__half2float(__high2half(params)),sums[r*groups+g],acc[r]));
    }
  }
#pragma unroll
  for(int r=0;r<R;++r) {
    for(int d=16;d;d>>=1)acc[r]+=__shfl_down_sync(0xffffffff,acc[r],d);
    if(!lane) {
      const auto i=std::size_t(r)*N+n;
      if(splits==1)y[i]=__float2half_rn(acc[r]);else partial[std::size_t(blockIdx.z)*M*N+i]=acc[r];
    }
  }
}

// Direct register dequantization into MMA B operands. Tile mapping matches the
// existing symmetric m16n8 path; no fully dequantized weight allocation.
template<int Bits, int BM>
__global__ void mma_direct(int M, int K, int N, const half* x, const unsigned char* q,
                           const half2* sm, half* y, float* partial, int splits) {
#if __CUDA_ARCH__ >= 750
  __shared__ __align__(16) half a[BM * 64];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * 64;
  const int tiles = K / 64, groups = K / 32;
  const int begin = tiles / splits * blockIdx.z + min(int(blockIdx.z), tiles % splits);
  const int end = begin + tiles / splits + (int(blockIdx.z) < tiles % splits);
  float acc[BM / 16][2][4] = {};
  for (int tile = begin; tile < end; ++tile) {
    for (int i = tid; i < BM * 8; i += 128) {
      const int row = i / 8, chunk = i % 8;
      int4 xv = make_int4(0, 0, 0, 0);
      if (m0 + row < M) xv = *reinterpret_cast<const int4*>(x + std::size_t(m0 + row) * K + tile * 64 + chunk * 8);
      *reinterpret_cast<int4*>(a + row * 64 + (chunk ^ (row & 7)) * 8) = xv;
    }
    __syncthreads();
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const int n = n0 + warp * 16 + jj * 8 + lane / 4;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        const int k = tile * 64 + kk * 16 + (lane & 3) * 2;
        const auto group = std::size_t(n) * groups + k / 32;
        const auto* packed = q + group * (16 + 4 * (Bits - 4));
        const half2 params = sm[group];
        const half2 b0 = pair<Bits>(packed, k % 32, params);
        const half2 b1 = pair<Bits>(packed, k % 32 + 8, params);
        const unsigned bb0 = *reinterpret_cast<const unsigned*>(&b0), bb1 = *reinterpret_cast<const unsigned*>(&b1);
#pragma unroll
        for (int mm = 0; mm < BM / 16; ++mm) {
          unsigned av[4];
          const int row = mm * 16 + lane % 16, chunk = kk * 2 + lane / 16;
          const unsigned addr = __cvta_generic_to_shared(a + row * 64 + (chunk ^ (row & 7)) * 8);
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
              : "=r"(av[0]), "=r"(av[1]), "=r"(av[2]), "=r"(av[3]) : "r"(addr) : "memory");
#if __CUDA_ARCH__ >= 800
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(acc[mm][jj][0]), "+f"(acc[mm][jj][1]), "+f"(acc[mm][jj][2]), "+f"(acc[mm][jj][3])
              : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]), "r"(bb0), "r"(bb1));
#else
          asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
              "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
              : "+f"(acc[mm][jj][0]), "+f"(acc[mm][jj][1]), "+f"(acc[mm][jj][2]), "+f"(acc[mm][jj][3])
              : "r"(av[0]), "r"(av[1]), "r"(bb0));
          asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
              "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
              : "+f"(acc[mm][jj][0]), "+f"(acc[mm][jj][1]), "+f"(acc[mm][jj][2]), "+f"(acc[mm][jj][3])
              : "r"(av[2]), "r"(av[3]), "r"(bb1));
#endif
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int mm = 0; mm < BM / 16; ++mm) for (int jj = 0; jj < 2; ++jj) for (int j = 0; j < 2; ++j) {
    const int m = m0 + mm * 16 + lane / 4 + j * 8, n = n0 + warp * 16 + jj * 8 + (lane & 3) * 2;
    if (m < M) {
      const auto i = std::size_t(m) * N + n;
      if (splits == 1) *reinterpret_cast<half2*>(y + i) = __floats2half2_rn(acc[mm][jj][j*2], acc[mm][jj][j*2+1]);
      else *reinterpret_cast<float2*>(partial + std::size_t(blockIdx.z)*M*N + i) = make_float2(acc[mm][jj][j*2], acc[mm][jj][j*2+1]);
    }
  }
#endif
}

// Bounds-safe WMMA path for ragged K/N, still dequantizing pairs in registers.
template<int Bits>
__global__ void gemm_tail(int M, int K, int N, const half* x, const unsigned char* q,
                          const half2* sm, half* y, float* partial, int splits) {
  using namespace nvcuda;
  __shared__ __align__(32) half a[16*72], b[64*72];
  __shared__ __align__(32) float c[16*64];
  const int tid=threadIdx.x, warp=tid/32, m0=blockIdx.y*16, n0=blockIdx.x*64;
  const int groups=(K+31)/32, tiles=(K+63)/64;
  const int begin=tiles/splits*blockIdx.z+min(int(blockIdx.z),tiles%splits);
  const int end=begin+tiles/splits+(int(blockIdx.z)<tiles%splits);
  wmma::fragment<wmma::accumulator,16,16,16,float> acc;
  wmma::fill_fragment(acc,0.f);
  for(int t=begin;t<end;++t) {
    for(int i=tid;i<16*64;i+=128) {
      const int m=m0+i/64,k=t*64+i%64;
      a[i/64*72+i%64]=m<M&&k<K?x[std::size_t(m)*K+k]:__float2half(0);
    }
    for(int i=tid;i<64*32;i+=128) {
      const int n=n0+i/32,k=t*64+(i%32)*2;
      half2 v=__float2half2_rn(0.f);
      if(n<N&&k<K) {
        const auto g=std::size_t(n)*groups+k/32;
        v=pair<Bits>(q+g*(16+4*(Bits-4)),k%32,sm[g]);
        if(k+1>=K) v=__halves2half2(__low2half(v),__float2half(0));
      }
      *reinterpret_cast<half2*>(b+i/32*72+(i%32)*2)=v;
    }
    __syncthreads();
    for(int k=0;k<64;k+=16) {
      wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> af;
      wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> bf;
      wmma::load_matrix_sync(af,a+k,72); wmma::load_matrix_sync(bf,b+warp*16*72+k,72);
      wmma::mma_sync(acc,af,bf,acc);
    }
    __syncthreads();
  }
  wmma::store_matrix_sync(c+warp*16,acc,64,wmma::mem_row_major); __syncthreads();
  for(int i=tid;i<16*64;i+=128) {
    const int m=m0+i/64,n=n0+i%64;
    if(m<M&&n<N) {
      const auto out=std::size_t(m)*N+n;
      if(splits==1)y[out]=__float2half_rn(c[i]);else partial[std::size_t(blockIdx.z)*M*N+out]=c[i];
    }
  }
}
__global__ void reduce(const float* p, half* y, std::size_t count, int splits) {
  const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<count) { float s=0;for(int j=0;j<splits;++j)s+=p[std::size_t(j)*count+i];y[i]=__float2half_rn(s); }
}
template<int Bits>
void launch(cudaStream_t stream,int M,int K,int N,const half* x,const unsigned char* q,
            const half2* sm,half* y,float* p,int splits,const float* sums) {
  if(M<=4) {
    const dim3 grid((N+3)/4,1,splits);
    switch(M) {
#define RUN(R) case R: if(sums)gemv_cached<Bits,R><<<grid,128,0,stream>>>(M,K,N,x,q,sm,sums,y,p,splits);else gemv<Bits,R><<<grid,128,0,stream>>>(M,K,N,x,q,sm,y,p,splits); break;
      RUN(1) RUN(2) RUN(3) RUN(4)
#undef RUN
    }
  } else if(K%64==0&&N%64==0) {
#define MMA(BM) mma_direct<Bits,BM><<<dim3(N/64,(M+BM-1)/BM,splits),128,0,stream>>>(M,K,N,x,q,sm,y,p,splits)
    if(M<=16){MMA(16);}else if(M<=32){MMA(32);}else{MMA(64);}
#undef MMA
  } else gemm_tail<Bits><<<dim3((N+63)/64,(M+15)/16,splits),128,0,stream>>>(M,K,N,x,q,sm,y,p,splits);
  if(splits>1)reduce<<<(std::size_t(M)*N+255)/256,256,0,stream>>>(p,y,std::size_t(M)*N,splits);
}
}
std::size_t rwkv7_asym_workspace_bytes(int M,int N,int splits) {
  if(M<=0||N<=0||splits<=1)return 0;
  if(std::size_t(M)>SIZE_MAX/std::size_t(N)/sizeof(float)/std::size_t(splits))throw std::invalid_argument("SB6 workspace overflow");
  return std::size_t(M)*N*splits*sizeof(float);
}
void rwkv7_asym_linear_launch(cudaStream_t stream,int M,int K,int N,const half* x,
    const std::uint8_t* q,const half2* sm,half* y,int bits,void* workspace,std::size_t bytes,int force) {
  if((bits!=4&&bits!=6)||force<0)throw std::invalid_argument("invalid SB6 bit width/split");
  if(M<=0||K<=0||N<=0)return;
  if(!x||!q||!sm||!y)throw std::invalid_argument("null SB6 buffer");
  const int tile=M<=4?256:64;
  const int tiles=K/tile+(K%tile!=0);
  int splits=force?force:std::min(tiles,K>=4096&&N<=8192?8:1);
  if(splits>tiles)throw std::invalid_argument("SB6 split exceeds K tiles");
  if(!force&&(!workspace||bytes<rwkv7_asym_workspace_bytes(M,N,splits)))splits=1;
  if(splits>1&&(!workspace||reinterpret_cast<std::uintptr_t>(workspace)%alignof(float)||bytes<rwkv7_asym_workspace_bytes(M,N,splits)))
    throw std::invalid_argument("insufficient/misaligned SB6 workspace");
  auto* p=static_cast<float*>(workspace);
  float* sums=nullptr;
  const auto partial_bytes=rwkv7_asym_workspace_bytes(M,N,splits);
  if(M<=4 && K%256==0 && workspace && reinterpret_cast<std::uintptr_t>(workspace)%alignof(float)==0 &&
      bytes>=partial_bytes+std::size_t(M)*(K/32)*sizeof(float)) {
    sums=p+partial_bytes/sizeof(float);
    activation_sums<<<(M*(K/32)+3)/4,128,0,stream>>>(x,sums,M*(K/32));
  }
  switch(bits) {
    case 4:launch<4>(stream,M,K,N,x,q,sm,y,p,splits,sums);break;
    case 6:launch<6>(stream,M,K,N,x,q,sm,y,p,splits,sums);break;
  }
}
