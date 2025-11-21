#include "spmv.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cassert>

union common128 {
    int4 I;
    struct {int x,y,z,w;} J;
    struct {float x,y,z,w;} F;
    struct {double x,y;} D;
    struct {half2 x,y,z,w;} G;
    struct {half a,b,c,d,e,f,g,h;} H;
    half h[8];
    int i[4];
    float f[4];
};

template <int N>
__device__ __forceinline__ void cp_async_gs_conditional(void const *const smem_addr,
                                       void const *const global_ptr, bool cond) {
    static_assert(N == 16 || N == 8 || N == 4);
    int bytes = cond ? N : 0;
    unsigned int addr = __cvta_generic_to_shared(smem_addr);
    if constexpr (N == 16) {
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], %2, %3;"
            ::"r"(addr),
            "l"(global_ptr), "n"(N), "r"(bytes));
    } else {
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], %2, %3;"
            ::"r"(addr),
            "l"(global_ptr), "n"(N), "r"(bytes));
    }
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    if constexpr (N == 0) {
        asm volatile("cp.async.wait_all;\n" ::);
    } else {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
    }
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__global__ void __launch_bounds__(SPMV_BLOCKDIM, 1) spvecmatmul_noindices(
    const int C,
    const half* __restrict__ vec,
    const half* __restrict__ mat,
    half* __restrict__ out
){
    __shared__ __align__(256) half mat_row_smem[2][2*SPMV_BLOCKDIM];
    __shared__ __align__(256) half vec_slice[SPMV_MAXNPERBLOCK];
    __shared__ __align__(256) int nnz_ids[SPMV_MAXNPERBLOCK];
    __shared__ int nnz_count;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int t = threadIdx.x;
    const int start_pos = bx * SPMV_MAXNPERBLOCK;

    if (t < 32){
        *(half2*)(vec_slice + t*2) = *(const half2*)(vec + start_pos + t*2);
    }
    __syncthreads();
    if (t == 0){
        int cnt = 0;
        #pragma unroll
        for (int i=0; i<8; ++i) {
            common128 z;
            z.I = ((const int4*)vec_slice)[i];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                unsigned short bits = __half_as_ushort(z.h[j]);
                if (bits != 0x0000 && bits != 0x8000) {
                    int idx = i * 8 + j;
                    nnz_ids[cnt] = idx;
                    cnt++;
                }
            }
        }
        nnz_count = cnt;
    }
    __syncthreads();

    half2 out_frag;
    *(int*)(&out_frag) = 0;
    // init
    #pragma unroll
    for(int i = 0; i < 2; i++){
        if (i < nnz_count){
            int actual_pos = start_pos + nnz_ids[i];
            cp_async_gs_conditional<4>(mat_row_smem[i%2] + t*2, mat + actual_pos * C + by * (2*SPMV_BLOCKDIM) + t*2, true);
            cp_async_commit();
        }
    }
    // main for
    for(int i = 0; i < nnz_count-2; i++){
        // take data
        cp_async_wait<1>();
        __syncthreads();

        half2 mat_row_frag = *(half2*) (mat_row_smem[i%2] + t*2);
        half vec_value = vec_slice[nnz_ids[i]];

        // store
        int actual_pos = start_pos + nnz_ids[i+2];
        cp_async_gs_conditional<4>(mat_row_smem[i%2] + t*2, mat + actual_pos * C + by * (2*SPMV_BLOCKDIM) + t*2, true);
        cp_async_commit();

        // compute
        out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
    }

    // end
    if (nnz_count >= 2){
        cp_async_wait<1>();
        __syncthreads();

        half2 mat_row_frag = *(half2*) (mat_row_smem[nnz_count%2] + t*2);
        half vec_value = vec_slice[nnz_ids[nnz_count - 2]];

        out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
    }
    if (nnz_count >= 1){
        cp_async_wait<0>();
        __syncthreads();

        half2 mat_row_frag = *(half2*) (mat_row_smem[(nnz_count+1)%2] + t*2);
        half vec_value = vec_slice[nnz_ids[nnz_count - 1]];

        out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
    }
    atomicAdd((half2*)(out + by*(2*SPMV_BLOCKDIM) + t*2), out_frag);
}

void spmv_forward_fp16(
    int D, int C,
    const half* vec,
    const half* mat,
    half* out,
    cudaStream_t stream
) {
    assert(C % (2*SPMV_BLOCKDIM) == 0);
    assert(D % SPMV_MAXNPERBLOCK == 0);
    spvecmatmul_noindices<<<dim3(D/SPMV_MAXNPERBLOCK, C/(2*SPMV_BLOCKDIM), 1), dim3(SPMV_BLOCKDIM, 1, 1), 0, stream>>>
    (C, vec, mat, out);
}

