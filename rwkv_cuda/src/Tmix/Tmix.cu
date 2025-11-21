#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include "Tmix.cuh"
#include "module/linear_cublas.cuh"
#include "module/norm.cuh"
#include "module/tensor_utils.cuh"
#include "wkv/wkv.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <vector>

// 调试辅助函数：打印 GPU 数组的前几个值
void debug_print_array(const half* d_data, int size, const char* name, cudaStream_t stream = nullptr) {
    if (stream) cudaStreamSynchronize(stream);
    else cudaDeviceSynchronize();
    
    std::vector<half> h_data(size);
    cudaMemcpy(h_data.data(), d_data, size * sizeof(half), cudaMemcpyDeviceToHost);
    
    printf("[DEBUG] %s: ", name);
    int print_size = size < 10 ? size : 10;
    for (int i = 0; i < print_size; i++) {
        float val = __half2float(h_data[i]);
        if (isnan(val) || isinf(val)) {
            printf("%s ", isnan(val) ? "nan" : (val > 0 ? "inf" : "-inf"));
        } else {
            printf("%.6f ", val);
        }
    }
    if (size > 10) printf("...");
    printf("\n");
}

// 辅助函数：计算 x + xx * scale
__global__ void compute_x_plus_xx_scale_kernel(
    const half* x, const half* xx, const half* scale,
    half* output, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __hadd(x[idx], __hmul(xx[idx], scale[idx]));
    }
}

void compute_x_plus_xx_scale(
    const half* x, const half* xx, const half* scale,
    half* output, int size, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    compute_x_plus_xx_scale_kernel<<<blocks, threads, 0, stream>>>(
        x, xx, scale, output, size);
}

// 辅助函数：计算 k * (1 + (a-1) * k_a)
__global__ void compute_k_scaled_kernel(
    const half* k, const half* a, const half* k_a,
    half* output, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        half a_minus_1 = __hsub(a[idx], __float2half(1.0f));
        half scaled = __hmul(a_minus_1, k_a[idx]);
        half one_plus_scaled = __hadd(__float2half(1.0f), scaled);
        output[idx] = __hmul(k[idx], one_plus_scaled);
    }
}

void compute_k_scaled(
    const half* k, const half* a, const half* k_a,
    half* output, int size, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    compute_k_scaled_kernel<<<blocks, threads, 0, stream>>>(
        k, a, k_a, output, size);
}

// 辅助函数：计算 v = v + (v_first - v) * sigmoid(...)
__global__ void compute_v_with_sigmoid_kernel(
    const half* v, const half* v_first, const half* sigmoid_val,
    half* output, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        half diff = __hsub(v_first[idx], v[idx]);
        half scaled = __hmul(diff, sigmoid_val[idx]);
        output[idx] = __hadd(v[idx], scaled);
    }
}

void compute_v_with_sigmoid(
    const half* v, const half* v_first, const half* sigmoid_val,
    half* output, int size, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    compute_v_with_sigmoid_kernel<<<blocks, threads, 0, stream>>>(
        v, v_first, sigmoid_val, output, size);
}

// 辅助函数：计算 (r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)
__global__ void compute_rkrk_sum_times_v_kernel(
    const half* r, const half* k, const half* r_k, const half* v,
    half* output, int H, int N
) {
    int h = blockIdx.x;
    if (h >= H) return;
    
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ half sum_val;
    
    int tid = threadIdx.x;
    float thread_sum = 0.0f;
    
    // 使用 float 精度累加，提高数值稳定性
    for (int i = tid; i < N; i += blockDim.x) {
        int idx = h * N + i;
        float rk = __half2float(__hmul(r[idx], k[idx]));
        float rkrk = rk * __half2float(r_k[i]);
        thread_sum += rkrk;
    }
    
    float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    
    // 将 sum 写入共享内存（转换为 half），让所有线程可以访问
    if (tid == 0) {
        sum_val = __float2half(block_sum);
    }
    __syncthreads();
    
    // 所有线程并行写入 output
    for (int i = tid; i < N; i += blockDim.x) {
        int idx = h * N + i;
        output[idx] = __hmul(sum_val, v[idx]);
    }
}

void compute_rkrk_sum_times_v(
    const half* r, const half* k, const half* r_k, const half* v,
    half* output, int H, int N, cudaStream_t stream
) {
    dim3 grid(H);
    dim3 block(256);
    compute_rkrk_sum_times_v_kernel<<<grid, block, 0, stream>>>(
        r, k, r_k, v, output, H, N);
}

// 序列版本的 rkrk 计算
__global__ void compute_rkrk_sum_times_v_seq_kernel(
    const half* r, const half* k, const half* r_k, const half* v,
    half* output, int T, int H, int N
) {
    int th = blockIdx.x;
    if (th >= T * H) return;
    
    int t = th / H;
    int h = th % H;
    
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ half sum_val;
    
    int tid = threadIdx.x;
    float thread_sum = 0.0f;
    
    // 使用 float 精度累加，提高数值稳定性
    for (int i = tid; i < N; i += blockDim.x) {
        int idx = (t * H + h) * N + i;
        float rk = __half2float(__hmul(r[idx], k[idx]));
        float rkrk = rk * __half2float(r_k[i]);
        thread_sum += rkrk;
    }
    
    float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    
    // 将 sum 写入共享内存（转换为 half），让所有线程可以访问
    if (tid == 0) {
        sum_val = __float2half(block_sum);
    }
    __syncthreads();
    
    // 所有线程并行写入 output
    for (int i = tid; i < N; i += blockDim.x) {
        int idx = (t * H + h) * N + i;
        output[idx] = __hmul(sum_val, v[idx]);
    }
}

void compute_rkrk_sum_times_v_seq(
    const half* r, const half* k, const half* r_k, const half* v,
    half* output, int T, int H, int N, cudaStream_t stream
) {
    dim3 grid(T * H);
    dim3 block(256);
    compute_rkrk_sum_times_v_seq_kernel<<<grid, block, 0, stream>>>(
        r, k, r_k, v, output, T, H, N);
}

// 批量版本的 rkrk 计算
__global__ void compute_rkrk_sum_times_v_batch_kernel(
    const half* r, const half* k, const half* r_k, const half* v,
    half* output, int B, int T, int H, int N
) {
    int bth = blockIdx.x;
    if (bth >= B * T * H) return;
    
    int b = bth / (T * H);
    int th = bth % (T * H);
    int t = th / H;
    int h = th % H;
    
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ half sum_val;
    
    int tid = threadIdx.x;
    float thread_sum = 0.0f;
    
    // 使用 float 精度累加，提高数值稳定性
    for (int i = tid; i < N; i += blockDim.x) {
        int idx = ((b * T + t) * H + h) * N + i;
        float rk = __half2float(__hmul(r[idx], k[idx]));
        float rkrk = rk * __half2float(r_k[i]);
        thread_sum += rkrk;
    }
    
    float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    
    // 将 sum 写入共享内存（转换为 half），让所有线程可以访问
    if (tid == 0) {
        sum_val = __float2half(block_sum);
    }
    __syncthreads();
    
    // 所有线程并行写入 output
    for (int i = tid; i < N; i += blockDim.x) {
        int idx = ((b * T + t) * H + h) * N + i;
        output[idx] = __hmul(sum_val, v[idx]);
    }
}

void compute_rkrk_sum_times_v_batch(
    const half* r, const half* k, const half* r_k, const half* v,
    half* output, int B, int T, int H, int N, cudaStream_t stream
) {
    dim3 grid(B * T * H);
    dim3 block(256);
    compute_rkrk_sum_times_v_batch_kernel<<<grid, block, 0, stream>>>(
        r, k, r_k, v, output, B, T, H, N);
}

// tmix_one_fp16 实现
void tmix_one_fp16(
    int layer_id, int H, int N,
    const half* x,
    half* x_prev,
    half* v_first,
    half* state,
    const half* x_r, const half* x_w, const half* x_k, const half* x_v,
    const half* x_a, const half* x_g,
    const half* w0, const half* w1, const half* w2,
    const half* a0, const half* a1, const half* a2,
    const half* v0, const half* v1, const half* v2,
    const half* g1, const half* g2,
    const half* k_k, const half* k_a, const half* r_k,
    const half* R_, const half* K_, const half* V_, const half* O_,
    const half* ln_w, const half* ln_b,
    const int* elapsed_t,
    TmixOneTempBuffers* temp_buf,
    half* output,
    cudaStream_t stream
) {
    int C = H * N;
    
    // ===== 参数验证：必须在任何 Kernel Launch 之前完成 =====
    // 检查基础权重指针（所有层都需要）
    if (R_ == nullptr || w1 == nullptr || w2 == nullptr || w0 == nullptr || 
        K_ == nullptr || V_ == nullptr || a1 == nullptr || a2 == nullptr || a0 == nullptr ||
        g1 == nullptr || g2 == nullptr || O_ == nullptr ||
        k_k == nullptr || k_a == nullptr || r_k == nullptr ||
        ln_w == nullptr || ln_b == nullptr) {
        fprintf(stderr, "Error: Required weights missing for layer %d.\n", layer_id);
        fprintf(stderr, "  R_=%p, w1=%p, w2=%p, w0=%p\n", R_, w1, w2, w0);
        fprintf(stderr, "  K_=%p, V_=%p, a1=%p, a2=%p, a0=%p\n", K_, V_, a1, a2, a0);
        fprintf(stderr, "  g1=%p, g2=%p, O_=%p\n", g1, g2, O_);
        fprintf(stderr, "  k_k=%p, k_a=%p, r_k=%p\n", k_k, k_a, r_k);
        fprintf(stderr, "  ln_w=%p, ln_b=%p\n", ln_w, ln_b);
        exit(1);
    }
    
    // 检查 v 相关权重（layer_id > 0 时必需）
    if (layer_id > 0 && (v0 == nullptr || v1 == nullptr || v2 == nullptr)) {
        fprintf(stderr, "Error: v weights missing for layer %d (layer_id > 0 requires v0, v1, v2).\n", layer_id);
        fprintf(stderr, "  v0=%p, v1=%p, v2=%p\n", v0, v1, v2);
        exit(1);
    }
    
    // 检查输入输出指针
    if (x == nullptr || x_prev == nullptr || v_first == nullptr || state == nullptr || 
        output == nullptr || temp_buf == nullptr) {
        fprintf(stderr, "Error: Invalid input/output pointers for layer %d.\n", layer_id);
        fprintf(stderr, "  x=%p, x_prev=%p, v_first=%p, state=%p, output=%p, temp_buf=%p\n",
                x, x_prev, v_first, state, output, temp_buf);
        exit(1);
    }
    // ===== 参数验证结束 =====
    
    // 使用传入的临时缓冲区
    half *xx = temp_buf->xx;
    half *xr = temp_buf->xr;
    half *xw = temp_buf->xw;
    half *xk = temp_buf->xk;
    half *xv = temp_buf->xv;
    half *xa = temp_buf->xa;
    half *xg = temp_buf->xg;
    half *r = temp_buf->r;
    half *w_intermediate = temp_buf->w_intermediate;
    half *w = temp_buf->w;
    half *k = temp_buf->k;
    half *v = temp_buf->v;
    half *a_intermediate = temp_buf->a_intermediate;
    half *a = temp_buf->a;
    half *g_intermediate = temp_buf->g_intermediate;
    half *g = temp_buf->g;
    half *kk = temp_buf->kk;
    half *k_scaled = temp_buf->k_scaled;
    half *kka = temp_buf->kka;
    half *v_intermediate = temp_buf->v_intermediate;
    half *v_sigmoid = temp_buf->v_sigmoid;
    half *xx_wkv = temp_buf->xx_wkv;
    half *xx_gn = temp_buf->xx_gn;
    half *xx_final = temp_buf->xx_final;
    half *g_scaled = temp_buf->g_scaled;
    
    printf("\n=== DEBUG tmix_one_fp16 (layer_id=%d, H=%d, N=%d, C=%d) ===\n", layer_id, H, N, C);
    
    // 1. xx = x_prev[0] - x
    element_wise_sub_fp16(x_prev, x, xx, C, stream);
    debug_print_array(xx, C, "1. xx = x_prev - x");
    
    // 2. x_prev[0] = x
    copy_fp16(x, x_prev, C, stream);
    
    // 3. 计算 xr, xw, xk, xv, xa, xg = x + xx * x_r, ...
    compute_x_plus_xx_scale(x, xx, x_r, xr, C, stream);
    debug_print_array(xr, C, "3. xr = x + xx * x_r");
    compute_x_plus_xx_scale(x, xx, x_w, xw, C, stream);
    debug_print_array(xw, C, "3. xw = x + xx * x_w");
    compute_x_plus_xx_scale(x, xx, x_k, xk, C, stream);
    debug_print_array(xk, C, "3. xk = x + xx * x_k");
    compute_x_plus_xx_scale(x, xx, x_v, xv, C, stream);
    debug_print_array(xv, C, "3. xv = x + xx * x_v");
    compute_x_plus_xx_scale(x, xx, x_a, xa, C, stream);
    debug_print_array(xa, C, "3. xa = x + xx * x_a");
    compute_x_plus_xx_scale(x, xx, x_g, xg, C, stream);
    debug_print_array(xg, C, "3. xg = x + xx * x_g");
    
    // 4. 线性层操作（此时权重指针已确认非空）
    
    // r = F.linear(xr, R_)
    linear_vec_fp16(xr, R_, nullptr, r, C, C, stream);
    debug_print_array(r, C, "4. r = linear(xr, R_)");
    
    // w = F.linear(torch.tanh(F.linear(xw, w1)), w2, bias=w0)
    linear_vec_fp16(xw, w1, nullptr, w_intermediate, C, C, stream);
    debug_print_array(w_intermediate, C, "4. w_intermediate = linear(xw, w1)");
    tanh_fp16(w_intermediate, w_intermediate, C, stream);
    debug_print_array(w_intermediate, C, "4. w_intermediate = tanh(w_intermediate)");
    linear_vec_fp16(w_intermediate, w2, w0, w, C, C, stream);
    debug_print_array(w, C, "4. w = linear(tanh(...), w2, bias=w0)");
    
    // k = F.linear(xk, K_)
    linear_vec_fp16(xk, K_, nullptr, k, C, C, stream);
    debug_print_array(k, C, "4. k = linear(xk, K_)");
    
    // v = F.linear(xv, V_)
    linear_vec_fp16(xv, V_, nullptr, v, C, C, stream);
    debug_print_array(v, C, "4. v = linear(xv, V_)");
    
    // a = torch.sigmoid(F.linear(F.linear(xa, a1), a2, bias=a0))
    linear_vec_fp16(xa, a1, nullptr, a_intermediate, C, C, stream);
    debug_print_array(a_intermediate, C, "4. a_intermediate = linear(xa, a1)");
    linear_vec_fp16(a_intermediate, a2, a0, a_intermediate, C, C, stream);
    debug_print_array(a_intermediate, C, "4. a_intermediate = linear(..., a2, bias=a0)");
    sigmoid_fp16(a_intermediate, a, C, stream);
    debug_print_array(a, C, "4. a = sigmoid(a_intermediate)");
    
    // g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
    linear_vec_fp16(xg, g1, nullptr, g_intermediate, C, C, stream);
    debug_print_array(g_intermediate, C, "4. g_intermediate = linear(xg, g1)");
    sigmoid_fp16(g_intermediate, g_intermediate, C, stream);
    debug_print_array(g_intermediate, C, "4. g_intermediate = sigmoid(g_intermediate)");
    linear_vec_fp16(g_intermediate, g2, nullptr, g, C, C, stream);
    debug_print_array(g, C, "4. g = linear(sigmoid(...), g2)");
    
    // 5. kk = F.normalize((k * k_k).view(H,N), dim=-1, p=2.0).view(H*N)
    element_wise_mul_fp16(k, k_k, kk, C, stream);
    debug_print_array(kk, C, "5. kk = k * k_k (before normalize)");
    l2_normalize_fp16(kk, kk, H, N, 1e-5f, stream);
    debug_print_array(kk, C, "5. kk = normalize(kk)");
    
    // 6. k = k * (1 + (a-1) * k_a)
    compute_k_scaled(k, a, k_a, k_scaled, C, stream);
    copy_fp16(k_scaled, k, C, stream);
    debug_print_array(k, C, "6. k = k * (1 + (a-1) * k_a)");
    
    // 7. kka = kk * a
    element_wise_mul_fp16(kk, a, kka, C, stream);
    debug_print_array(kka, C, "7. kka = kk * a");
    
    // 8. 处理 v_first（v0, v1, v2 已在函数开始处验证）
    if (layer_id == 0) {
        copy_fp16(v, v_first, C, stream);
        debug_print_array(v_first, C, "8. v_first = v (layer_id==0)");
    } else {
        // v = v + (v_first - v) * torch.sigmoid(F.linear(F.linear(xv, v1), v2, bias=v0))
        linear_vec_fp16(xv, v1, nullptr, v_intermediate, C, C, stream);
        debug_print_array(v_intermediate, C, "8. v_intermediate = linear(xv, v1)");
        linear_vec_fp16(v_intermediate, v2, v0, v_intermediate, C, C, stream);
        debug_print_array(v_intermediate, C, "8. v_intermediate = linear(..., v2, bias=v0)");
        sigmoid_fp16(v_intermediate, v_sigmoid, C, stream);
        debug_print_array(v_sigmoid, C, "8. v_sigmoid = sigmoid(v_intermediate)");
        compute_v_with_sigmoid(v, v_first, v_sigmoid, v, C, stream);
        debug_print_array(v, C, "8. v = v + (v_first - v) * v_sigmoid");
    }
    
    // 9. xx = RWKV7_ONE_OP(state, r, w, k, v, -kk, kka, elapsed_t)
    half *neg_kk = temp_buf->neg_kk;
    negate_fp16(kk, neg_kk, C, stream);
    debug_print_array(neg_kk, C, "9. neg_kk = -kk");
    wkv_forward_one(1, C, H, state, r, w, k, v, neg_kk, kka, xx_wkv, elapsed_t, stream);
    debug_print_array(xx_wkv, C, "9. xx_wkv = wkv_forward_one(...)");
    
    // 10. xx = F.group_norm(xx.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N)
    group_norm_half8_fp16(xx_wkv, ln_w, ln_b, xx_gn, 1, H, N, 64e-5f, stream);
    debug_print_array(xx_gn, C, "10. xx_gn = group_norm(xx_wkv)");
    
    // 11. xx = xx + ((r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)
    compute_rkrk_sum_times_v(r, k, r_k, v, xx_final, H, N, stream);
    debug_print_array(xx_final, C, "11. xx_final = (r*k*r_k).sum() * v");
    element_wise_add_fp16(xx_gn, xx_final, xx_gn, C, stream);
    debug_print_array(xx_gn, C, "11. xx_gn = xx_gn + xx_final");
    
    // 12. return F.linear((xx * g), O_), v_first
    element_wise_mul_fp16(xx_gn, g, g_scaled, C, stream);
    debug_print_array(g_scaled, C, "12. g_scaled = xx_gn * g");
    linear_vec_fp16(g_scaled, O_, nullptr, output, C, C, stream);
    debug_print_array(output, C, "12. output = linear(g_scaled, O_)");
    
    printf("=== END DEBUG ===\n\n");
}

// tmix_seq_fp16 实现
void tmix_seq_fp16(
    int layer_id, int H, int N,
    const half* x, int T,
    half* x_prev,
    half* v_first,
    half* state,
    const half* x_r, const half* x_w, const half* x_k, const half* x_v,
    const half* x_a, const half* x_g,
    const half* w0, const half* w1, const half* w2,
    const half* a0, const half* a1, const half* a2,
    const half* v0, const half* v1, const half* v2,
    const half* g1, const half* g2,
    const half* k_k, const half* k_a, const half* r_k,
    const half* R_, const half* K_, const half* V_, const half* O_,
    const half* ln_w, const half* ln_b,
    const int* elapsed_t,
    TmixSeqTempBuffers* temp_buf,
    half* output,
    cudaStream_t stream
) {
    int C = H * N;
    int total_size = T * C;
    
    // ===== 参数验证：必须在任何 Kernel Launch 之前完成 =====
    // 检查基础权重指针（所有层都需要）
    if (R_ == nullptr || w1 == nullptr || w2 == nullptr || w0 == nullptr || 
        K_ == nullptr || V_ == nullptr || a1 == nullptr || a2 == nullptr || a0 == nullptr ||
        g1 == nullptr || g2 == nullptr || O_ == nullptr ||
        k_k == nullptr || k_a == nullptr || r_k == nullptr ||
        ln_w == nullptr || ln_b == nullptr) {
        fprintf(stderr, "Error: Required weights missing for layer %d (seq).\n", layer_id);
        fprintf(stderr, "  R_=%p, w1=%p, w2=%p, w0=%p\n", R_, w1, w2, w0);
        fprintf(stderr, "  K_=%p, V_=%p, a1=%p, a2=%p, a0=%p\n", K_, V_, a1, a2, a0);
        fprintf(stderr, "  g1=%p, g2=%p, O_=%p\n", g1, g2, O_);
        fprintf(stderr, "  k_k=%p, k_a=%p, r_k=%p\n", k_k, k_a, r_k);
        fprintf(stderr, "  ln_w=%p, ln_b=%p\n", ln_w, ln_b);
        exit(1);
    }
    
    // 检查 v 相关权重（layer_id > 0 时必需）
    if (layer_id > 0 && (v0 == nullptr || v1 == nullptr || v2 == nullptr)) {
        fprintf(stderr, "Error: v weights missing for layer %d (seq, layer_id > 0 requires v0, v1, v2).\n", layer_id);
        fprintf(stderr, "  v0=%p, v1=%p, v2=%p\n", v0, v1, v2);
        exit(1);
    }
    
    // 检查输入输出指针
    if (x == nullptr || x_prev == nullptr || v_first == nullptr || state == nullptr || 
        output == nullptr || temp_buf == nullptr) {
        fprintf(stderr, "Error: Invalid input/output pointers for layer %d (seq).\n", layer_id);
        fprintf(stderr, "  x=%p, x_prev=%p, v_first=%p, state=%p, output=%p, temp_buf=%p\n",
                x, x_prev, v_first, state, output, temp_buf);
        exit(1);
    }
    
    // 检查序列长度
    if (T <= 0) {
        fprintf(stderr, "Error: Invalid sequence length T=%d for layer %d (seq).\n", T, layer_id);
        exit(1);
    }
    // ===== 参数验证结束 =====
    
    // 使用传入的临时缓冲区
    half *xx = temp_buf->xx;
    half *xr = temp_buf->xr;
    half *xw = temp_buf->xw;
    half *xk = temp_buf->xk;
    half *xv = temp_buf->xv;
    half *xa = temp_buf->xa;
    half *xg = temp_buf->xg;
    half *r = temp_buf->r;
    half *w_intermediate = temp_buf->w_intermediate;
    half *w = temp_buf->w;
    half *k = temp_buf->k;
    half *v = temp_buf->v;
    half *a_intermediate = temp_buf->a_intermediate;
    half *a = temp_buf->a;
    half *g_intermediate = temp_buf->g_intermediate;
    half *g = temp_buf->g;
    half *kk = temp_buf->kk;
    half *k_scaled = temp_buf->k_scaled;
    half *kka = temp_buf->kka;
    half *v_intermediate = temp_buf->v_intermediate;
    half *v_sigmoid = temp_buf->v_sigmoid;
    half *xx_wkv = temp_buf->xx_wkv;
    half *xx_gn = temp_buf->xx_gn;
    half *xx_final = temp_buf->xx_final;
    half *g_scaled = temp_buf->g_scaled;
    half *x_concat = temp_buf->x_concat;
    half *x_r_expanded = temp_buf->x_r_expanded;
    half *x_w_expanded = temp_buf->x_w_expanded;
    half *x_k_expanded = temp_buf->x_k_expanded;
    half *x_v_expanded = temp_buf->x_v_expanded;
    half *x_a_expanded = temp_buf->x_a_expanded;
    half *x_g_expanded = temp_buf->x_g_expanded;
    
    // 1. xx = torch.cat((x_prev[0].unsqueeze(0), x[:-1,:])) - x
    // 先创建拼接后的张量
    cat_fp16(x_prev, x, x_concat, 0, 1, C, 1, T-1, stream);
    // 然后减去 x
    element_wise_sub_fp16(x_concat, x, xx, total_size, stream);
    
    // 2. x_prev[0] = x[-1,:]
    copy_fp16(x + (T-1) * C, x_prev, C, stream);
    
    // 3. 计算 xr, xw, xk, xv, xa, xg
    // 需要广播 x_r 等参数到 [T, C]
    broadcast_v_first_fp16(x_r, x_r_expanded, T, C, stream);
    broadcast_v_first_fp16(x_w, x_w_expanded, T, C, stream);
    broadcast_v_first_fp16(x_k, x_k_expanded, T, C, stream);
    broadcast_v_first_fp16(x_v, x_v_expanded, T, C, stream);
    broadcast_v_first_fp16(x_a, x_a_expanded, T, C, stream);
    broadcast_v_first_fp16(x_g, x_g_expanded, T, C, stream);
    
    compute_x_plus_xx_scale(x, xx, x_r_expanded, xr, total_size, stream);
    compute_x_plus_xx_scale(x, xx, x_w_expanded, xw, total_size, stream);
    compute_x_plus_xx_scale(x, xx, x_k_expanded, xk, total_size, stream);
    compute_x_plus_xx_scale(x, xx, x_v_expanded, xv, total_size, stream);
    compute_x_plus_xx_scale(x, xx, x_a_expanded, xa, total_size, stream);
    compute_x_plus_xx_scale(x, xx, x_g_expanded, xg, total_size, stream);
    
    // 4. 线性层操作（对每个时间步）
    linear_fp16(xr, R_, nullptr, r, T, C, C, stream);
    linear_fp16(xw, w1, nullptr, w_intermediate, T, C, C, stream);
    tanh_fp16(w_intermediate, w_intermediate, total_size, stream);
    linear_fp16(w_intermediate, w2, w0, w, T, C, C, stream);
    linear_fp16(xk, K_, nullptr, k, T, C, C, stream);
    linear_fp16(xv, V_, nullptr, v, T, C, C, stream);
    linear_fp16(xa, a1, nullptr, a_intermediate, T, C, C, stream);
    linear_fp16(a_intermediate, a2, a0, a_intermediate, T, C, C, stream);
    sigmoid_fp16(a_intermediate, a, total_size, stream);
    linear_fp16(xg, g1, nullptr, g_intermediate, T, C, C, stream);
    sigmoid_fp16(g_intermediate, g_intermediate, total_size, stream);
    linear_fp16(g_intermediate, g2, nullptr, g, T, C, C, stream);
    
    // 5. kk = F.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
    half *k_k_expanded = temp_buf->k_k_expanded;
    broadcast_v_first_fp16(k_k, k_k_expanded, T, C, stream);
    element_wise_mul_fp16(k, k_k_expanded, kk, total_size, stream);
    l2_normalize_fp16(kk, kk, T * H, N, 1e-5f, stream);
    
    // 6. k = k * (1 + (a-1) * k_a)
    half *k_a_expanded = temp_buf->k_a_expanded;
    broadcast_v_first_fp16(k_a, k_a_expanded, T, C, stream);
    compute_k_scaled(k, a, k_a_expanded, k_scaled, total_size, stream);
    copy_fp16(k_scaled, k, total_size, stream);
    
    // 7. kka = kk * a
    element_wise_mul_fp16(kk, a, kka, total_size, stream);
    
    // 8. 处理 v_first
    if (layer_id == 0) {
        copy_fp16(v, v_first, C, stream);
    } else {
        // 广播 v_first 到 [T, C]
        half *v_first_expanded = temp_buf->v_first_expanded;
        broadcast_v_first_fp16(v_first, v_first_expanded, T, C, stream);
        
        linear_fp16(xv, v1, nullptr, v_intermediate, T, C, C, stream);
        linear_fp16(v_intermediate, v2, v0, v_intermediate, T, C, C, stream);
        sigmoid_fp16(v_intermediate, v_sigmoid, total_size, stream);
        compute_v_with_sigmoid(v, v_first_expanded, v_sigmoid, v, total_size, stream);
    }
    
    // 9. xx = RWKV7_SEQ_OP(state, r, w, k, v, -kk, kka, elapsed_t)
    half *neg_kk = temp_buf->neg_kk;
    negate_fp16(kk, neg_kk, total_size, stream);
    wkv_forward_seq(1, T, C, H, state, r, w, k, v, neg_kk, kka, xx_wkv, elapsed_t, stream);
    
    // 10. xx = F.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
    group_norm_half8_fp16(xx_wkv, ln_w, ln_b, xx_gn, T, H, N, 64e-5f, stream);
    
    // 11. xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
    compute_rkrk_sum_times_v_seq(r, k, r_k, v, xx_final, T, H, N, stream);
    element_wise_add_fp16(xx_gn, xx_final, xx_gn, total_size, stream);
    
    // 12. return F.linear((xx * g), O_), v_first
    element_wise_mul_fp16(xx_gn, g, g_scaled, total_size, stream);
    linear_fp16(g_scaled, O_, nullptr, output, T, C, C, stream);
}

// tmix_seq_batch_fp16 实现
void tmix_seq_batch_fp16(
    int layer_id, int H, int N,
    const half* x, int B, int T,
    half* x_prev,
    half* v_first,
    half* state,
    const half* x_r, const half* x_w, const half* x_k, const half* x_v,
    const half* x_a, const half* x_g,
    const half* w0, const half* w1, const half* w2,
    const half* a0, const half* a1, const half* a2,
    const half* v0, const half* v1, const half* v2,
    const half* g1, const half* g2,
    const half* k_k, const half* k_a, const half* r_k,
    const half* R_, const half* K_, const half* V_, const half* O_,
    const half* ln_w, const half* ln_b,
    const int* elapsed_t,
    TmixSeqTempBuffers* temp_buf,
    half* output,
    cudaStream_t stream
) {
    int C = H * N;
    int total_size = B * T * C;
    
    // ===== 参数验证：必须在任何 Kernel Launch 之前完成 =====
    // 检查基础权重指针（所有层都需要）
    if (R_ == nullptr || w1 == nullptr || w2 == nullptr || w0 == nullptr || 
        K_ == nullptr || V_ == nullptr || a1 == nullptr || a2 == nullptr || a0 == nullptr ||
        g1 == nullptr || g2 == nullptr || O_ == nullptr ||
        k_k == nullptr || k_a == nullptr || r_k == nullptr ||
        ln_w == nullptr || ln_b == nullptr) {
        fprintf(stderr, "Error: Required weights missing for layer %d (batch).\n", layer_id);
        fprintf(stderr, "  R_=%p, w1=%p, w2=%p, w0=%p\n", R_, w1, w2, w0);
        fprintf(stderr, "  K_=%p, V_=%p, a1=%p, a2=%p, a0=%p\n", K_, V_, a1, a2, a0);
        fprintf(stderr, "  g1=%p, g2=%p, O_=%p\n", g1, g2, O_);
        fprintf(stderr, "  k_k=%p, k_a=%p, r_k=%p\n", k_k, k_a, r_k);
        fprintf(stderr, "  ln_w=%p, ln_b=%p\n", ln_w, ln_b);
        exit(1);
    }
    
    // 检查 v 相关权重（layer_id > 0 时必需）
    if (layer_id > 0 && (v0 == nullptr || v1 == nullptr || v2 == nullptr)) {
        fprintf(stderr, "Error: v weights missing for layer %d (batch, layer_id > 0 requires v0, v1, v2).\n", layer_id);
        fprintf(stderr, "  v0=%p, v1=%p, v2=%p\n", v0, v1, v2);
        exit(1);
    }
    
    // 检查输入输出指针
    if (x == nullptr || x_prev == nullptr || v_first == nullptr || state == nullptr || 
        output == nullptr || temp_buf == nullptr) {
        fprintf(stderr, "Error: Invalid input/output pointers for layer %d (batch).\n", layer_id);
        fprintf(stderr, "  x=%p, x_prev=%p, v_first=%p, state=%p, output=%p, temp_buf=%p\n",
                x, x_prev, v_first, state, output, temp_buf);
        exit(1);
    }
    
    // 检查批次和序列长度
    if (B <= 0 || T <= 0) {
        fprintf(stderr, "Error: Invalid batch size B=%d or sequence length T=%d for layer %d (batch).\n", B, T, layer_id);
        exit(1);
    }
    // ===== 参数验证结束 =====
    
    // 使用传入的临时缓冲区
    half *xx = temp_buf->xx;
    half *xr = temp_buf->xr;
    half *xw = temp_buf->xw;
    half *xk = temp_buf->xk;
    half *xv = temp_buf->xv;
    half *xa = temp_buf->xa;
    half *xg = temp_buf->xg;
    half *r = temp_buf->r;
    half *w_intermediate = temp_buf->w_intermediate;
    half *w = temp_buf->w;
    half *k = temp_buf->k;
    half *v = temp_buf->v;
    half *a_intermediate = temp_buf->a_intermediate;
    half *a = temp_buf->a;
    half *g_intermediate = temp_buf->g_intermediate;
    half *g = temp_buf->g;
    half *kk = temp_buf->kk;
    half *k_scaled = temp_buf->k_scaled;
    half *kka = temp_buf->kka;
    half *v_intermediate = temp_buf->v_intermediate;
    half *v_sigmoid = temp_buf->v_sigmoid;
    half *xx_wkv = temp_buf->xx_wkv;
    half *xx_gn = temp_buf->xx_gn;
    half *xx_final = temp_buf->xx_final;
    half *g_scaled = temp_buf->g_scaled;
    half *x_concat = temp_buf->x_concat;
    half *x_r_expanded = temp_buf->x_r_expanded;
    half *x_w_expanded = temp_buf->x_w_expanded;
    half *x_k_expanded = temp_buf->x_k_expanded;
    half *x_v_expanded = temp_buf->x_v_expanded;
    half *x_a_expanded = temp_buf->x_a_expanded;
    half *x_g_expanded = temp_buf->x_g_expanded;
    
    // 1. xx = torch.cat((x_prev[0].unsqueeze(1), x[:,:-1,:]), dim=1) - x
    create_concat_batch_fp16(x_prev, x, x_concat, B, T, C, stream);
    element_wise_sub_fp16(x_concat, x, xx, total_size, stream);
    
    // 2. x_prev[0] = x[:,-1,:]
    update_x_prev_batch_fp16(x, x_prev, B, T, C, stream);
    
    // 3. 计算 xr, xw, xk, xv, xa, xg
    // 需要广播 x_r 等参数到 [B, T, C]
    broadcast_v_first_batch_fp16(x_r, x_r_expanded, B, T, C, stream);
    broadcast_v_first_batch_fp16(x_w, x_w_expanded, B, T, C, stream);
    broadcast_v_first_batch_fp16(x_k, x_k_expanded, B, T, C, stream);
    broadcast_v_first_batch_fp16(x_v, x_v_expanded, B, T, C, stream);
    broadcast_v_first_batch_fp16(x_a, x_a_expanded, B, T, C, stream);
    broadcast_v_first_batch_fp16(x_g, x_g_expanded, B, T, C, stream);
    
    compute_x_plus_xx_scale(x, xx, x_r_expanded, xr, total_size, stream);
    compute_x_plus_xx_scale(x, xx, x_w_expanded, xw, total_size, stream);
    compute_x_plus_xx_scale(x, xx, x_k_expanded, xk, total_size, stream);
    compute_x_plus_xx_scale(x, xx, x_v_expanded, xv, total_size, stream);
    compute_x_plus_xx_scale(x, xx, x_a_expanded, xa, total_size, stream);
    compute_x_plus_xx_scale(x, xx, x_g_expanded, xg, total_size, stream);
    
    // 4. 线性层操作（对每个批次和时间步）
    linear_fp16(xr, R_, nullptr, r, B * T, C, C, stream);
    linear_fp16(xw, w1, nullptr, w_intermediate, B * T, C, C, stream);
    tanh_fp16(w_intermediate, w_intermediate, total_size, stream);
    linear_fp16(w_intermediate, w2, w0, w, B * T, C, C, stream);
    linear_fp16(xk, K_, nullptr, k, B * T, C, C, stream);
    linear_fp16(xv, V_, nullptr, v, B * T, C, C, stream);
    linear_fp16(xa, a1, nullptr, a_intermediate, B * T, C, C, stream);
    linear_fp16(a_intermediate, a2, a0, a_intermediate, B * T, C, C, stream);
    sigmoid_fp16(a_intermediate, a, total_size, stream);
    linear_fp16(xg, g1, nullptr, g_intermediate, B * T, C, C, stream);
    sigmoid_fp16(g_intermediate, g_intermediate, total_size, stream);
    linear_fp16(g_intermediate, g2, nullptr, g, B * T, C, C, stream);
    
    // 5. kk = F.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
    half *k_k_expanded = temp_buf->k_k_expanded;
    broadcast_v_first_batch_fp16(k_k, k_k_expanded, B, T, C, stream);
    element_wise_mul_fp16(k, k_k_expanded, kk, total_size, stream);
    l2_normalize_fp16(kk, kk, B * T * H, N, 1e-5f, stream);
    
    // 6. k = k * (1 + (a-1) * k_a)
    half *k_a_expanded = temp_buf->k_a_expanded;
    broadcast_v_first_batch_fp16(k_a, k_a_expanded, B, T, C, stream);
    compute_k_scaled(k, a, k_a_expanded, k_scaled, total_size, stream);
    copy_fp16(k_scaled, k, total_size, stream);
    
    // 7. kka = kk * a
    element_wise_mul_fp16(kk, a, kka, total_size, stream);
    
    // 8. 处理 v_first
    if (layer_id == 0) {
        copy_fp16(v, v_first, B * C, stream);
    } else {
        // 广播 v_first 到 [B, T, C]
        half *v_first_expanded = temp_buf->v_first_expanded;
        broadcast_v_first_batch_fp16(v_first, v_first_expanded, B, T, C, stream);
        
        linear_fp16(xv, v1, nullptr, v_intermediate, B * T, C, C, stream);
        linear_fp16(v_intermediate, v2, v0, v_intermediate, B * T, C, C, stream);
        sigmoid_fp16(v_intermediate, v_sigmoid, total_size, stream);
        compute_v_with_sigmoid(v, v_first_expanded, v_sigmoid, v, total_size, stream);
    }
    
    // 9. xx = RWKV7_BATCH_OP(state, r, w, k, v, -kk, kka, elapsed_t).view(B*T,H*N)
    half *neg_kk = temp_buf->neg_kk;
    negate_fp16(kk, neg_kk, total_size, stream);
    wkv_forward_seq(B, T, C, H, state, r, w, k, v, neg_kk, kka, xx_wkv, elapsed_t, stream);
    
    // 10. xx = F.group_norm(xx.view(B*T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(B,T,H*N)
    group_norm_half8_fp16(xx_wkv, ln_w, ln_b, xx_gn, B * T, H, N, 64e-5f, stream);
    
    // 11. xx = xx + ((r * k * r_k).view(B,T,H,N).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)
    compute_rkrk_sum_times_v_batch(r, k, r_k, v, xx_final, B, T, H, N, stream);
    element_wise_add_fp16(xx_gn, xx_final, xx_gn, total_size, stream);
    
    // 12. return F.linear((xx * g), O_), v_first
    element_wise_mul_fp16(xx_gn, g, g_scaled, total_size, stream);
    linear_fp16(g_scaled, O_, nullptr, output, B * T, C, C, stream);
}

