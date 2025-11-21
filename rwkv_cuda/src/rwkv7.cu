#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include "rwkv7.cuh"
#include "Tmix/Tmix.cuh"
#include "Cmix/Cmix.cuh"
#include "module/linear_cublas.cuh"
#include "module/norm.cuh"
#include "module/tensor_utils.cuh"
#include "spmv/spmv.cuh"
#include "utils/load_model.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <cstdlib>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            assert(false); \
        } \
    } while(0)

// CUDA kernel: 更新 elapsed time（批量情况）
__global__ void increment_elapsed_kernel(int* elapsed_t, int bsz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < bsz) {
        elapsed_t[idx]++;
    }
}

// RWKVModel 构造函数
RWKVModel::RWKVModel() 
    : n_layer(0), n_embd(0), n_head(0), head_size(64), vocab_size(65536), ffn_dim(0) {
}

// RWKVModel 析构函数
RWKVModel::~RWKVModel() {
    clear();
}

// 清理资源
void RWKVModel::clear() {
    for (auto& pair : weights) {
        if (pair.second != nullptr) {
            cudaFree(pair.second);
            pair.second = nullptr;
        }
    }
    weights.clear();
}

// 获取权重指针
half* RWKVModel::get_weight(const std::string& name) const {
    auto it = weights.find(name);
    if (it != weights.end()) {
        return it->second;
    }
    return nullptr;
}

// 检查权重是否存在
bool RWKVModel::has_weight(const std::string& name) const {
    return weights.find(name) != weights.end();
}

// 获取权重形状
std::vector<int64_t> RWKVModel::get_weight_shape(const std::string& name) const {
    auto it = weight_shapes.find(name);
    if (it != weight_shapes.end()) {
        return it->second;
    }
    return std::vector<int64_t>();
}

// 从 safetensors 文件加载模型
bool RWKVModel::load_from_safetensors(const std::string& model_path) {
    SafeTensorsLoader loader;
    if (!loader.load(model_path)) {
        fprintf(stderr, "Failed to load model from %s\n", model_path.c_str());
        return false;
    }
    
    // 获取所有张量名称
    auto tensor_names = loader.get_tensor_names();
    
    // 确定层数
    int max_layer = -1;
    for (const auto& name : tensor_names) {
        if (name.find("blocks.") == 0) {
            size_t dot_pos = name.find('.', 7);
            if (dot_pos != std::string::npos) {
                int layer = std::stoi(name.substr(7, dot_pos - 7));
                max_layer = std::max(max_layer, layer);
            }
        }
    }
    n_layer = max_layer + 1;
    
    // 从 blocks.0.att.r_k 获取 n_head 和 head_size
    TensorInfo r_k_info;
    if (loader.get_tensor_info("blocks.0.att.r_k", r_k_info)) {
        if (r_k_info.shape.size() == 2) {
            n_head = r_k_info.shape[0];
            head_size = r_k_info.shape[1];
        } else if (r_k_info.shape.size() == 1) {
            // 如果已经被 flatten
            n_head = r_k_info.shape[0] / head_size;
        }
        n_embd = n_head * head_size;
    } else {
        fprintf(stderr, "Warning: Cannot find blocks.0.att.r_k, using default values\n");
        n_embd = 1024;  // 默认值
        n_head = n_embd / head_size;
    }
    
    // 获取 vocab_size（从 emb.weight）
    TensorInfo emb_info;
    if (loader.get_tensor_info("emb.weight", emb_info)) {
        if (emb_info.shape.size() >= 1) {
            vocab_size = emb_info.shape[0];  // [vocab_size, n_embd]
        }
    }
    
    // 获取 FFN 维度（从 blocks.0.ffn.key.weight）
    TensorInfo ffn_key_info;
    if (loader.get_tensor_info("blocks.0.ffn.key.weight", ffn_key_info)) {
        if (ffn_key_info.shape.size() == 2) {
            ffn_dim = ffn_key_info.shape[1];  // [C, D] -> D
        } else {
            ffn_dim = n_embd * 4;  // 默认值：通常是 n_embd 的 4 倍
        }
    } else {
        ffn_dim = n_embd * 4;  // 默认值
    }
    
    printf("Model config: n_layer=%d, n_embd=%d, n_head=%d, head_size=%d\n", 
           n_layer, n_embd, n_head, head_size);
    
    // 加载所有权重
    for (const auto& name : tensor_names) {
        TensorInfo info;
        if (!loader.get_tensor_info(name, info)) {
            continue;
        }
        
        size_t numel = 1;
        for (int64_t dim : info.shape) {
            numel *= dim;
        }
        
        // 分配 GPU 内存
        half* d_weight = nullptr;
        CUDA_CHECK(cudaMalloc(&d_weight, numel * sizeof(half)));
        
        // 加载到 GPU
        if (!loader.load_tensor_to_gpu(name, d_weight)) {
            fprintf(stderr, "Failed to load tensor %s\n", name.c_str());
            cudaFree(d_weight);
            continue;
        }
        
        weights[name] = d_weight;
        weight_shapes[name] = info.shape;  // 保存形状信息
    }
    
    // 处理 emb.weight 的 layer norm（使用 blocks.0.ln0）
    if (has_weight("emb.weight") && has_weight("blocks.0.ln0.weight") && has_weight("blocks.0.ln0.bias")) {
        half* emb = get_weight("emb.weight");
        const half* ln_w = get_weight("blocks.0.ln0.weight");
        const half* ln_b = get_weight("blocks.0.ln0.bias");
        
        // 使用 CUDA kernel 进行 layer norm
        // emb.weight 形状: [vocab_size, n_embd]
        // 对每一行进行 layer norm
        layer_norm_half8_fp16(emb, ln_w, ln_b, emb, vocab_size, n_embd, 1e-5f, nullptr);
        cudaDeviceSynchronize();  // 确保完成后再继续
    }
    
    // 处理 blocks.0.att.v0, v1, v2（使用 a0, a1, a2）
    if (has_weight("blocks.0.att.a0") && !has_weight("blocks.0.att.v0")) {
        weights["blocks.0.att.v0"] = get_weight("blocks.0.att.a0");
    }
    if (has_weight("blocks.0.att.a1") && !has_weight("blocks.0.att.v1")) {
        weights["blocks.0.att.v1"] = get_weight("blocks.0.att.a1");
    }
    if (has_weight("blocks.0.att.a2") && !has_weight("blocks.0.att.v2")) {
        weights["blocks.0.att.v2"] = get_weight("blocks.0.att.a2");
    }
    
    printf("Loaded %zu tensors\n", weights.size());
    return true;
}

// 生成零状态
void generate_zero_state(
    const RWKVModel* model,
    int bsz,
    half** state0,
    half** state1,
    int** state2,
    cudaStream_t stream
) {
    int n_layer = model->n_layer;
    int n_embd = model->n_embd;
    int H = model->n_head;
    int N = model->head_size;
    
    if (bsz >= 1) {
        // state0: [n_layer, 2, bsz, n_embd]
        size_t state0_size = n_layer * 2 * bsz * n_embd * sizeof(half);
        CUDA_CHECK(cudaMalloc(state0, state0_size));
        CUDA_CHECK(cudaMemsetAsync(*state0, 0, state0_size, stream));
        
        // state1: [n_layer, bsz, H, N, N]
        size_t state1_size = n_layer * bsz * H * N * N * sizeof(half);
        CUDA_CHECK(cudaMalloc(state1, state1_size));
        CUDA_CHECK(cudaMemsetAsync(*state1, 0, state1_size, stream));
        
        // state2: [bsz]
        size_t state2_size = bsz * sizeof(int);
        CUDA_CHECK(cudaMalloc((void**)state2, state2_size));
        CUDA_CHECK(cudaMemsetAsync(*state2, 0, state2_size, stream));
    } else {
        // state0: [n_layer, 2, n_embd]
        size_t state0_size = n_layer * 2 * n_embd * sizeof(half);
        CUDA_CHECK(cudaMalloc(state0, state0_size));
        CUDA_CHECK(cudaMemsetAsync(*state0, 0, state0_size, stream));
        
        // state1: [n_layer, H, N, N]
        size_t state1_size = n_layer * H * N * N * sizeof(half);
        CUDA_CHECK(cudaMalloc(state1, state1_size));
        CUDA_CHECK(cudaMemsetAsync(*state1, 0, state1_size, stream));
        
        // state2: scalar
        CUDA_CHECK(cudaMalloc((void**)state2, sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(*state2, 0, sizeof(int), stream));
    }
}

// 辅助函数：获取权重或返回 nullptr
static inline half* get_weight_safe(const RWKVModel* model, const std::string& name) {
    return model->has_weight(name) ? model->get_weight(name) : nullptr;
}

// 分配 TmixOneTempBuffers
static void allocate_tmix_one_buf(int C, TmixOneTempBuffers* buf) {
    CUDA_CHECK(cudaMalloc(&buf->xx, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xr, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xw, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xk, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xv, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xa, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xg, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->r, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->w_intermediate, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->w, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->k, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->v, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->a_intermediate, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->a, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->g_intermediate, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->g, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->kk, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->k_scaled, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->kka, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->v_intermediate, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->v_sigmoid, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xx_wkv, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xx_gn, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xx_final, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->g_scaled, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->neg_kk, C * sizeof(half)));
}

// 释放 TmixOneTempBuffers
static void free_tmix_one_buf(TmixOneTempBuffers* buf) {
    if (buf->xx) cudaFree(buf->xx);
    if (buf->xr) cudaFree(buf->xr);
    if (buf->xw) cudaFree(buf->xw);
    if (buf->xk) cudaFree(buf->xk);
    if (buf->xv) cudaFree(buf->xv);
    if (buf->xa) cudaFree(buf->xa);
    if (buf->xg) cudaFree(buf->xg);
    if (buf->r) cudaFree(buf->r);
    if (buf->w_intermediate) cudaFree(buf->w_intermediate);
    if (buf->w) cudaFree(buf->w);
    if (buf->k) cudaFree(buf->k);
    if (buf->v) cudaFree(buf->v);
    if (buf->a_intermediate) cudaFree(buf->a_intermediate);
    if (buf->a) cudaFree(buf->a);
    if (buf->g_intermediate) cudaFree(buf->g_intermediate);
    if (buf->g) cudaFree(buf->g);
    if (buf->kk) cudaFree(buf->kk);
    if (buf->k_scaled) cudaFree(buf->k_scaled);
    if (buf->kka) cudaFree(buf->kka);
    if (buf->v_intermediate) cudaFree(buf->v_intermediate);
    if (buf->v_sigmoid) cudaFree(buf->v_sigmoid);
    if (buf->xx_wkv) cudaFree(buf->xx_wkv);
    if (buf->xx_gn) cudaFree(buf->xx_gn);
    if (buf->xx_final) cudaFree(buf->xx_final);
    if (buf->g_scaled) cudaFree(buf->g_scaled);
    if (buf->neg_kk) cudaFree(buf->neg_kk);
}

// 分配 TmixSeqTempBuffers
static void allocate_tmix_seq_buf(int total_size, int C, TmixSeqTempBuffers* buf) {
    CUDA_CHECK(cudaMalloc(&buf->xx, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xr, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xw, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xk, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xv, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xa, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xg, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->r, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->w_intermediate, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->w, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->k, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->v, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->a_intermediate, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->a, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->g_intermediate, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->g, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->kk, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->k_scaled, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->kka, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->v_intermediate, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->v_sigmoid, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xx_wkv, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xx_gn, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->xx_final, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->g_scaled, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->neg_kk, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->x_concat, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->x_r_expanded, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->x_w_expanded, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->x_k_expanded, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->x_v_expanded, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->x_a_expanded, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->x_g_expanded, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->k_k_expanded, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->k_a_expanded, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->v_first_expanded, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->x_prev_expanded, C * sizeof(half)));
}

// 释放 TmixSeqTempBuffers
static void free_tmix_seq_buf(TmixSeqTempBuffers* buf) {
    if (buf->xx) cudaFree(buf->xx);
    if (buf->xr) cudaFree(buf->xr);
    if (buf->xw) cudaFree(buf->xw);
    if (buf->xk) cudaFree(buf->xk);
    if (buf->xv) cudaFree(buf->xv);
    if (buf->xa) cudaFree(buf->xa);
    if (buf->xg) cudaFree(buf->xg);
    if (buf->r) cudaFree(buf->r);
    if (buf->w_intermediate) cudaFree(buf->w_intermediate);
    if (buf->w) cudaFree(buf->w);
    if (buf->k) cudaFree(buf->k);
    if (buf->v) cudaFree(buf->v);
    if (buf->a_intermediate) cudaFree(buf->a_intermediate);
    if (buf->a) cudaFree(buf->a);
    if (buf->g_intermediate) cudaFree(buf->g_intermediate);
    if (buf->g) cudaFree(buf->g);
    if (buf->kk) cudaFree(buf->kk);
    if (buf->k_scaled) cudaFree(buf->k_scaled);
    if (buf->kka) cudaFree(buf->kka);
    if (buf->v_intermediate) cudaFree(buf->v_intermediate);
    if (buf->v_sigmoid) cudaFree(buf->v_sigmoid);
    if (buf->xx_wkv) cudaFree(buf->xx_wkv);
    if (buf->xx_gn) cudaFree(buf->xx_gn);
    if (buf->xx_final) cudaFree(buf->xx_final);
    if (buf->g_scaled) cudaFree(buf->g_scaled);
    if (buf->neg_kk) cudaFree(buf->neg_kk);
    if (buf->x_concat) cudaFree(buf->x_concat);
    if (buf->x_r_expanded) cudaFree(buf->x_r_expanded);
    if (buf->x_w_expanded) cudaFree(buf->x_w_expanded);
    if (buf->x_k_expanded) cudaFree(buf->x_k_expanded);
    if (buf->x_v_expanded) cudaFree(buf->x_v_expanded);
    if (buf->x_a_expanded) cudaFree(buf->x_a_expanded);
    if (buf->x_g_expanded) cudaFree(buf->x_g_expanded);
    if (buf->k_k_expanded) cudaFree(buf->k_k_expanded);
    if (buf->k_a_expanded) cudaFree(buf->k_a_expanded);
    if (buf->v_first_expanded) cudaFree(buf->v_first_expanded);
    if (buf->x_prev_expanded) cudaFree(buf->x_prev_expanded);
}

// 分配 CmixOneTempBuffers
static void allocate_cmix_one_buf(int C, int D, CmixOneTempBuffers* buf) {
    CUDA_CHECK(cudaMalloc(&buf->xx, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->k, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->k_linear, D * sizeof(half)));
}

// 释放 CmixOneTempBuffers
static void free_cmix_one_buf(CmixOneTempBuffers* buf) {
    if (buf->xx) cudaFree(buf->xx);
    if (buf->k) cudaFree(buf->k);
    if (buf->k_linear) cudaFree(buf->k_linear);
}

// 分配 CmixSeqTempBuffers
static void allocate_cmix_seq_buf(int total_size, int total_k_size, CmixSeqTempBuffers* buf) {
    CUDA_CHECK(cudaMalloc(&buf->xx, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->k, total_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf->k_linear, total_k_size * sizeof(half)));
}

// 释放 CmixSeqTempBuffers
static void free_cmix_seq_buf(CmixSeqTempBuffers* buf) {
    if (buf->xx) cudaFree(buf->xx);
    if (buf->k) cudaFree(buf->k);
    if (buf->k_linear) cudaFree(buf->k_linear);
}

// 分配前向传播临时缓冲区
RWKVForwardTempBuffers* allocate_forward_temp_buffers(
    const RWKVModel* model,
    ForwardMode mode,
    int max_bsz,
    int max_seq_len
) {
    RWKVForwardTempBuffers* bufs = new RWKVForwardTempBuffers();
    bufs->n_layer = model->n_layer;
    bufs->max_bsz = max_bsz;
    bufs->max_seq_len = max_seq_len;
    bufs->initialized = false;
    
    int C = model->n_embd;
    int D = model->ffn_dim;
    
    // 根据模式分配不同的缓冲区
    if (mode == ForwardMode::ONE) {
        // forward_one: 每层需要 TmixOneTempBuffers 和 CmixOneTempBuffers
        bufs->tmix_one_bufs = new TmixOneTempBuffers[model->n_layer];
        bufs->cmix_one_bufs = new CmixOneTempBuffers[model->n_layer];
        
        for (int i = 0; i < model->n_layer; i++) {
            allocate_tmix_one_buf(C, &bufs->tmix_one_bufs[i]);
            allocate_cmix_one_buf(C, D, &bufs->cmix_one_bufs[i]);
        }
        
        // 通用缓冲区
        CUDA_CHECK(cudaMalloc(&bufs->x_current, C * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&bufs->x_ln, C * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&bufs->xx_tmix, C * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&bufs->xx_cmix, C * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&bufs->v_first, C * sizeof(half)));
        
        bufs->x_seq = nullptr;
        bufs->x_last_batch = nullptr;
        bufs->tmix_seq_bufs = nullptr;
        bufs->cmix_seq_bufs = nullptr;
        
    } else if (mode == ForwardMode::SEQ) {
        // forward_seq: 每层需要 TmixSeqTempBuffers 和 CmixSeqTempBuffers
        // 确保 max_seq_len 至少为 1
        if (max_seq_len < 1) max_seq_len = 1;
        int total_size = max_seq_len * C;
        int total_k_size = max_seq_len * D;
        
        bufs->tmix_seq_bufs = new TmixSeqTempBuffers[model->n_layer];
        bufs->cmix_seq_bufs = new CmixSeqTempBuffers[model->n_layer];
        
        for (int i = 0; i < model->n_layer; i++) {
            allocate_tmix_seq_buf(total_size, C, &bufs->tmix_seq_bufs[i]);
            allocate_cmix_seq_buf(total_size, total_k_size, &bufs->cmix_seq_bufs[i]);
        }
        
        // 通用缓冲区
        CUDA_CHECK(cudaMalloc(&bufs->x_current, total_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&bufs->x_ln, total_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&bufs->xx_tmix, total_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&bufs->xx_cmix, total_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&bufs->v_first, C * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&bufs->x_seq, total_size * sizeof(half)));
        
        bufs->x_last_batch = nullptr;
        bufs->tmix_one_bufs = nullptr;
        bufs->cmix_one_bufs = nullptr;
        
    } else { // SEQ_BATCH
        // forward_seq_batch: 每层需要 TmixSeqTempBuffers 和 CmixSeqTempBuffers
        int total_size = max_bsz * max_seq_len * C;
        int total_k_size = max_bsz * max_seq_len * D;
        
        bufs->tmix_seq_bufs = new TmixSeqTempBuffers[model->n_layer];
        bufs->cmix_seq_bufs = new CmixSeqTempBuffers[model->n_layer];
        
        for (int i = 0; i < model->n_layer; i++) {
            allocate_tmix_seq_buf(total_size, C, &bufs->tmix_seq_bufs[i]);
            allocate_cmix_seq_buf(total_size, total_k_size, &bufs->cmix_seq_bufs[i]);
        }
        
        // 通用缓冲区
        CUDA_CHECK(cudaMalloc(&bufs->x_current, total_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&bufs->x_ln, total_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&bufs->xx_tmix, total_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&bufs->xx_cmix, total_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&bufs->v_first, max_bsz * C * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&bufs->x_seq, total_size * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&bufs->x_last_batch, max_bsz * C * sizeof(half)));
        
        bufs->tmix_one_bufs = nullptr;
        bufs->cmix_one_bufs = nullptr;
    }
    
    bufs->initialized = true;
    return bufs;
}

// 释放前向传播临时缓冲区
void free_forward_temp_buffers(RWKVForwardTempBuffers* bufs) {
    if (!bufs) return;
    
    // 释放每层的缓冲区
    if (bufs->tmix_one_bufs) {
        for (int i = 0; i < bufs->n_layer; i++) {
            free_tmix_one_buf(&bufs->tmix_one_bufs[i]);
        }
        delete[] bufs->tmix_one_bufs;
    }
    
    if (bufs->tmix_seq_bufs) {
        for (int i = 0; i < bufs->n_layer; i++) {
            free_tmix_seq_buf(&bufs->tmix_seq_bufs[i]);
        }
        delete[] bufs->tmix_seq_bufs;
    }
    
    if (bufs->cmix_one_bufs) {
        for (int i = 0; i < bufs->n_layer; i++) {
            free_cmix_one_buf(&bufs->cmix_one_bufs[i]);
        }
        delete[] bufs->cmix_one_bufs;
    }
    
    if (bufs->cmix_seq_bufs) {
        for (int i = 0; i < bufs->n_layer; i++) {
            free_cmix_seq_buf(&bufs->cmix_seq_bufs[i]);
        }
        delete[] bufs->cmix_seq_bufs;
    }
    
    // 释放通用缓冲区
    if (bufs->x_current) cudaFree(bufs->x_current);
    if (bufs->x_ln) cudaFree(bufs->x_ln);
    if (bufs->xx_tmix) cudaFree(bufs->xx_tmix);
    if (bufs->xx_cmix) cudaFree(bufs->xx_cmix);
    if (bufs->v_first) cudaFree(bufs->v_first);
    if (bufs->x_seq) cudaFree(bufs->x_seq);
    if (bufs->x_last_batch) cudaFree(bufs->x_last_batch);
    
    delete bufs;
}

// 单 token 前向传播
void forward_one(
    const RWKVModel* model,
    const half* x,
    half* state0,
    half* state1,
    int* state2,
    half* output,
    int bsz,
    cudaStream_t stream
) {
    int n_layer = model->n_layer;
    int n_embd = model->n_embd;
    int H = model->n_head;
    int N = model->head_size;
    int C = n_embd;
    int vocab_size = model->vocab_size;
    
    // ===== 参数验证 =====
    if (x == nullptr || state0 == nullptr || state1 == nullptr || 
        state2 == nullptr || output == nullptr) {
        fprintf(stderr, "Error: Invalid pointers in forward_one\n");
        exit(1);
    }
    // ===== 参数验证结束 =====
    
    // 如果 bufs 为 nullptr，分配临时缓冲区（兼容旧接口）
    // 实际使用中应该预先分配并传入
    static RWKVForwardTempBuffers* static_bufs = nullptr;
    static ForwardMode last_mode = ForwardMode::ONE;
    
    bool need_allocate = (static_bufs == nullptr || 
                         last_mode != ForwardMode::ONE || 
                         !static_bufs->initialized);
    
    if (need_allocate) {
        if (static_bufs) {
            free_forward_temp_buffers(static_bufs);
        }
        static_bufs = allocate_forward_temp_buffers(model, ForwardMode::ONE, 1, 1);
        last_mode = ForwardMode::ONE;
    }
    
    RWKVForwardTempBuffers* bufs = static_bufs;
    
    // 验证缓冲区已正确分配
    if (bufs == nullptr || !bufs->initialized) {
        fprintf(stderr, "Error: Temporary buffers not initialized in forward_one\n");
        exit(1);
    }
    
    // 使用预分配的临时缓冲区
    half *x_current = bufs->x_current;
    half *x_ln = bufs->x_ln;
    half *xx_tmix = bufs->xx_tmix;
    half *xx_cmix = bufs->xx_cmix;
    half *v_first = bufs->v_first;
    
    // 复制输入
    CUDA_CHECK(cudaMemcpyAsync(x_current, x, C * sizeof(half), cudaMemcpyDeviceToDevice, stream));
    
    // 遍历每一层
    for (int i = 0; i < n_layer; i++) {
        char bbb[64], att[64], ffn[64];
        snprintf(bbb, sizeof(bbb), "blocks.%d.", i);
        snprintf(att, sizeof(att), "blocks.%d.att.", i);
        snprintf(ffn, sizeof(ffn), "blocks.%d.ffn.", i);
        
        // Layer norm 1
        std::string ln1_w = std::string(bbb) + "ln1.weight";
        std::string ln1_b = std::string(bbb) + "ln1.bias";
        layer_norm_half8_fp16(x_current, get_weight_safe(model, ln1_w), 
                             get_weight_safe(model, ln1_b), x_ln, 1, C, 1e-5f, stream);
        
        // TMix
        half* x_prev_layer = state0 + (i * 2 * (bsz >= 1 ? bsz * C : C));
        half* state1_layer = state1 + (i * (bsz >= 1 ? bsz * H * N * N : H * N * N));
        
        // ✅ 使用 bufs 中分配的临时缓冲区（不是 model->temp_bufs）
        tmix_one_fp16(
            i, H, N,
            x_ln,
            x_prev_layer,
            v_first,
            state1_layer,
            get_weight_safe(model, std::string(att) + "x_r"),
            get_weight_safe(model, std::string(att) + "x_w"),
            get_weight_safe(model, std::string(att) + "x_k"),
            get_weight_safe(model, std::string(att) + "x_v"),
            get_weight_safe(model, std::string(att) + "x_a"),
            get_weight_safe(model, std::string(att) + "x_g"),
            get_weight_safe(model, std::string(att) + "w0"),
            get_weight_safe(model, std::string(att) + "w1"),
            get_weight_safe(model, std::string(att) + "w2"),
            get_weight_safe(model, std::string(att) + "a0"),
            get_weight_safe(model, std::string(att) + "a1"),
            get_weight_safe(model, std::string(att) + "a2"),
            get_weight_safe(model, std::string(att) + "v0"),
            get_weight_safe(model, std::string(att) + "v1"),
            get_weight_safe(model, std::string(att) + "v2"),
            get_weight_safe(model, std::string(att) + "g1"),
            get_weight_safe(model, std::string(att) + "g2"),
            get_weight_safe(model, std::string(att) + "k_k"),
            get_weight_safe(model, std::string(att) + "k_a"),
            get_weight_safe(model, std::string(att) + "r_k"),
            get_weight_safe(model, std::string(att) + "receptance.weight"),
            get_weight_safe(model, std::string(att) + "key.weight"),
            get_weight_safe(model, std::string(att) + "value.weight"),
            get_weight_safe(model, std::string(att) + "output.weight"),
            get_weight_safe(model, std::string(att) + "ln_x.weight"),
            get_weight_safe(model, std::string(att) + "ln_x.bias"),
            state2,
            &bufs->tmix_one_bufs[i],
            xx_tmix,
            stream
        );
        
        // 残差连接
        element_wise_add_fp16(x_current, xx_tmix, x_current, C, stream);
        
        // Layer norm 2
        std::string ln2_w = std::string(bbb) + "ln2.weight";
        std::string ln2_b = std::string(bbb) + "ln2.bias";
        layer_norm_half8_fp16(x_current, get_weight_safe(model, ln2_w), 
                             get_weight_safe(model, ln2_b), x_ln, 1, C, 1e-5f, stream);
        
        // CMix
        half* x_prev_cmix = x_prev_layer;
        int D = model->ffn_dim;  // 使用保存的 ffn_dim
        
        // ✅ 使用 bufs 中分配的临时缓冲区（不是 model->temp_bufs）
        cmix_one_fp16(
            x_ln,
            x_prev_cmix,
            get_weight_safe(model, std::string(ffn) + "x_k"),
            get_weight_safe(model, std::string(ffn) + "key.weight"),
            get_weight_safe(model, std::string(ffn) + "value.weight"),
            &bufs->cmix_one_bufs[i],
            xx_cmix,
            C, D,
            stream
        );
        
        // 残差连接
        element_wise_add_fp16(x_current, xx_cmix, x_current, C, stream);
    }
    
    // 最终 layer norm
    layer_norm_half8_fp16(x_current, get_weight_safe(model, "ln_out.weight"), 
                         get_weight_safe(model, "ln_out.bias"), x_ln, 1, C, 1e-5f, stream);
    
    // 输出层
    linear_fp16(x_ln, get_weight_safe(model, "head.weight"), nullptr, output, 1, vocab_size, C, stream);
    
    // 更新 elapsed time
    if (bsz >= 1) {
        // 批量情况：每个样本的 elapsed_t 都加 1
        dim3 grid((bsz + 255) / 256);
        dim3 block(256);
        increment_elapsed_kernel<<<grid, block, 0, stream>>>(state2, bsz);
    } else {
        // 单样本情况：scalar 加 1
        int elapsed;
        CUDA_CHECK(cudaMemcpyAsync(&elapsed, state2, sizeof(int), cudaMemcpyDeviceToHost, stream));
        elapsed++;
        CUDA_CHECK(cudaMemcpyAsync(state2, &elapsed, sizeof(int), cudaMemcpyHostToDevice, stream));
    }
}

// 序列前向传播
void forward_seq(
    const RWKVModel* model,
    const int* tokens,
    int T,
    half* state0,
    half* state1,
    int* state2,
    half* output,
    bool full_output,
    cudaStream_t stream
) {
    int n_layer = model->n_layer;
    int n_embd = model->n_embd;
    int vocab_size = model->vocab_size;
    int C = n_embd;
    
    // ===== 参数验证 =====
    if (tokens == nullptr || state0 == nullptr || state1 == nullptr || 
        state2 == nullptr || output == nullptr) {
        fprintf(stderr, "Error: Invalid pointers in forward_seq\n");
        exit(1);
    }
    if (T <= 0) {
        fprintf(stderr, "Error: Invalid sequence length T=%d in forward_seq\n", T);
        exit(1);
    }
    // ===== 参数验证结束 =====
    
    // 获取 embedding 权重
    half* emb_weight = get_weight_safe(model, "emb.weight");
    if (!emb_weight) {
        fprintf(stderr, "Error: emb.weight not found\n");
        exit(1);
    }
    
    // 如果 bufs 为 nullptr，分配临时缓冲区（兼容旧接口）
    static RWKVForwardTempBuffers* static_bufs = nullptr;
    static ForwardMode last_mode = ForwardMode::SEQ;
    
    bool need_allocate = (static_bufs == nullptr || 
                         last_mode != ForwardMode::SEQ || 
                         static_bufs->max_seq_len < T ||
                         !static_bufs->initialized);
    
    if (need_allocate) {
        // 如果需要重新分配，使用更大的序列长度（至少是 T，如果之前分配过则使用之前的大小和 T 的最大值）
        int new_seq_len = (static_bufs && static_bufs->max_seq_len > 0) 
                         ? std::max(T, static_bufs->max_seq_len) 
                         : T;
        if (static_bufs) {
            free_forward_temp_buffers(static_bufs);
        }
        static_bufs = allocate_forward_temp_buffers(model, ForwardMode::SEQ, 1, new_seq_len);
        last_mode = ForwardMode::SEQ;
    }
    
    RWKVForwardTempBuffers* bufs = static_bufs;
    
    // 使用预分配的临时缓冲区
    half *x_seq = bufs->x_seq;
    half *x_current = bufs->x_current;
    half *x_ln = bufs->x_ln;
    half *xx_tmix = bufs->xx_tmix;
    half *xx_cmix = bufs->xx_cmix;
    half *v_first = bufs->v_first;
    
    // 从 embedding 中获取 token embeddings
    // tokens[i] -> emb_weight[tokens[i]]
    for (int t = 0; t < T; t++) {
        CUDA_CHECK(cudaMemcpyAsync(
            x_seq + t * C,
            emb_weight + tokens[t] * C,
            C * sizeof(half),
            cudaMemcpyDeviceToDevice,
            stream
        ));
    }
    
    // 复制输入
    CUDA_CHECK(cudaMemcpyAsync(x_current, x_seq, T * C * sizeof(half), cudaMemcpyDeviceToDevice, stream));
    
    int H = model->n_head;
    int N = model->head_size;
    
    // 遍历每一层
    for (int i = 0; i < n_layer; i++) {
        char bbb[64], att[64], ffn[64];
        snprintf(bbb, sizeof(bbb), "blocks.%d.", i);
        snprintf(att, sizeof(att), "blocks.%d.att.", i);
        snprintf(ffn, sizeof(ffn), "blocks.%d.ffn.", i);
        
        // Layer norm 1
        std::string ln1_w = std::string(bbb) + "ln1.weight";
        std::string ln1_b = std::string(bbb) + "ln1.bias";
        layer_norm_half8_fp16(x_current, get_weight_safe(model, ln1_w), 
                             get_weight_safe(model, ln1_b), x_ln, T, C, 1e-5f, stream);
        
        // TMix
        half* x_prev_layer = state0 + (i * 2 * C);
        half* state1_layer = state1 + (i * H * N * N);
        
        // ✅ 使用 bufs 中分配的临时缓冲区（不是 model->temp_bufs）
        tmix_seq_fp16(
            i, H, N,
            x_ln, T,
            x_prev_layer,
            v_first,
            state1_layer,
            get_weight_safe(model, std::string(att) + "x_r"),
            get_weight_safe(model, std::string(att) + "x_w"),
            get_weight_safe(model, std::string(att) + "x_k"),
            get_weight_safe(model, std::string(att) + "x_v"),
            get_weight_safe(model, std::string(att) + "x_a"),
            get_weight_safe(model, std::string(att) + "x_g"),
            get_weight_safe(model, std::string(att) + "w0"),
            get_weight_safe(model, std::string(att) + "w1"),
            get_weight_safe(model, std::string(att) + "w2"),
            get_weight_safe(model, std::string(att) + "a0"),
            get_weight_safe(model, std::string(att) + "a1"),
            get_weight_safe(model, std::string(att) + "a2"),
            get_weight_safe(model, std::string(att) + "v0"),
            get_weight_safe(model, std::string(att) + "v1"),
            get_weight_safe(model, std::string(att) + "v2"),
            get_weight_safe(model, std::string(att) + "g1"),
            get_weight_safe(model, std::string(att) + "g2"),
            get_weight_safe(model, std::string(att) + "k_k"),
            get_weight_safe(model, std::string(att) + "k_a"),
            get_weight_safe(model, std::string(att) + "r_k"),
            get_weight_safe(model, std::string(att) + "receptance.weight"),
            get_weight_safe(model, std::string(att) + "key.weight"),
            get_weight_safe(model, std::string(att) + "value.weight"),
            get_weight_safe(model, std::string(att) + "output.weight"),
            get_weight_safe(model, std::string(att) + "ln_x.weight"),
            get_weight_safe(model, std::string(att) + "ln_x.bias"),
            state2,
            &bufs->tmix_seq_bufs[i],
            xx_tmix,
            stream
        );
        
        // 残差连接
        element_wise_add_fp16(x_current, xx_tmix, x_current, T * C, stream);
        
        // Layer norm 2
        std::string ln2_w = std::string(bbb) + "ln2.weight";
        std::string ln2_b = std::string(bbb) + "ln2.bias";
        layer_norm_half8_fp16(x_current, get_weight_safe(model, ln2_w), 
                             get_weight_safe(model, ln2_b), x_ln, T, C, 1e-5f, stream);
        
        // CMix
        // x_prev_layer 指向 [2, C] 的起始位置
        // cmix_seq_fp16 期望 x_prev 指向 [2, C] 的起始位置，它会访问 x_prev[1] = x_prev + C
        half* x_prev_cmix = x_prev_layer;
        int D = model->ffn_dim;  // 使用保存的 ffn_dim
        
        // ✅ 使用 bufs 中分配的临时缓冲区（不是 model->temp_bufs）
        cmix_seq_fp16(
            x_ln,
            x_prev_cmix,
            get_weight_safe(model, std::string(ffn) + "x_k"),
            get_weight_safe(model, std::string(ffn) + "key.weight"),
            get_weight_safe(model, std::string(ffn) + "value.weight"),
            &bufs->cmix_seq_bufs[i],
            xx_cmix,
            T, C, D,
            stream
        );
        
        // 残差连接
        element_wise_add_fp16(x_current, xx_cmix, x_current, T * C, stream);
    }
    
    // 最终 layer norm
    layer_norm_half8_fp16(x_current, get_weight_safe(model, "ln_out.weight"), 
                         get_weight_safe(model, "ln_out.bias"), x_ln, T, C, 1e-5f, stream);
    
    // 输出层
    if (full_output) {
        // 输出所有时间步: [T, vocab_size]
        linear_fp16(x_ln, get_weight_safe(model, "head.weight"), nullptr, output, T, vocab_size, C, stream);
    } else {
        // 只输出最后一个时间步: [vocab_size]
        half* x_last = x_ln + (T - 1) * C;
        linear_fp16(x_last, get_weight_safe(model, "head.weight"), nullptr, output, 1, vocab_size, C, stream);
    }
    
    // 更新 elapsed time
    int elapsed;
    CUDA_CHECK(cudaMemcpyAsync(&elapsed, state2, sizeof(int), cudaMemcpyDeviceToHost, stream));
    elapsed += T;
    CUDA_CHECK(cudaMemcpyAsync(state2, &elapsed, sizeof(int), cudaMemcpyHostToDevice, stream));
}

// 批量序列前向传播
void forward_seq_batch(
    const RWKVModel* model,
    const int* tokens,
    int B,
    int T,
    const int* lengths,
    half* state0,
    half* state1,
    int* state2,
    half* output,
    bool full_output,
    cudaStream_t stream
) {
    int n_layer = model->n_layer;
    int n_embd = model->n_embd;
    int vocab_size = model->vocab_size;
    int C = n_embd;
    int H = model->n_head;
    int N = model->head_size;
    
    // ===== 参数验证 =====
    if (tokens == nullptr || state0 == nullptr || state1 == nullptr || 
        state2 == nullptr || output == nullptr) {
        fprintf(stderr, "Error: Invalid pointers in forward_seq_batch\n");
        exit(1);
    }
    if (B <= 0 || T <= 0) {
        fprintf(stderr, "Error: Invalid dimensions B=%d, T=%d in forward_seq_batch\n", B, T);
        exit(1);
    }
    // ===== 参数验证结束 =====
    
    // 获取 embedding 权重
    half* emb_weight = get_weight_safe(model, "emb.weight");
    if (!emb_weight) {
        fprintf(stderr, "Error: emb.weight not found\n");
        exit(1);
    }
    
    // 如果 bufs 为 nullptr，分配临时缓冲区（兼容旧接口）
    static RWKVForwardTempBuffers* static_bufs = nullptr;
    static ForwardMode last_mode = ForwardMode::SEQ_BATCH;
    static int last_bsz = 0;
    static int last_seq_len = 0;
    
    bool need_allocate = (static_bufs == nullptr || 
                         last_mode != ForwardMode::SEQ_BATCH || 
                         last_bsz < B ||
                         last_seq_len < T ||
                         !static_bufs->initialized);
    
    if (need_allocate) {
        if (static_bufs) {
            free_forward_temp_buffers(static_bufs);
        }
        static_bufs = allocate_forward_temp_buffers(model, ForwardMode::SEQ_BATCH, 
                                                    std::max(B, last_bsz), 
                                                    std::max(T, last_seq_len));
        last_mode = ForwardMode::SEQ_BATCH;
        last_bsz = std::max(B, last_bsz);
        last_seq_len = std::max(T, last_seq_len);
    }
    
    RWKVForwardTempBuffers* bufs = static_bufs;
    
    // 验证缓冲区已正确分配
    if (bufs == nullptr || !bufs->initialized) {
        fprintf(stderr, "Error: Temporary buffers not initialized in forward_seq_batch\n");
        exit(1);
    }
    
    // 使用预分配的临时缓冲区
    half *x_seq = bufs->x_seq;
    half *x_current = bufs->x_current;
    half *x_ln = bufs->x_ln;
    half *xx_tmix = bufs->xx_tmix;
    half *xx_cmix = bufs->xx_cmix;
    half *v_first = bufs->v_first;
    half *x_last_batch = bufs->x_last_batch;
    
    // 从 embedding 中获取 token embeddings
    for (int b = 0; b < B; b++) {
        int actual_T = (lengths != nullptr) ? lengths[b] : T;
        for (int t = 0; t < actual_T && t < T; t++) {
            int token_idx = tokens[b * T + t];
            CUDA_CHECK(cudaMemcpyAsync(
                x_seq + (b * T + t) * C,
                emb_weight + token_idx * C,
                C * sizeof(half),
                cudaMemcpyDeviceToDevice,
                stream
            ));
        }
        // 如果序列长度小于 T，用最后一个 token 填充
        for (int t = actual_T; t < T; t++) {
            CUDA_CHECK(cudaMemcpyAsync(
                x_seq + (b * T + t) * C,
                x_seq + (b * T + (actual_T - 1)) * C,
                C * sizeof(half),
                cudaMemcpyDeviceToDevice,
                stream
            ));
        }
    }
    
    // 复制输入
    CUDA_CHECK(cudaMemcpyAsync(x_current, x_seq, B * T * C * sizeof(half), cudaMemcpyDeviceToDevice, stream));
    
    // 遍历每一层
    for (int i = 0; i < n_layer; i++) {
        char bbb[64], att[64], ffn[64];
        snprintf(bbb, sizeof(bbb), "blocks.%d.", i);
        snprintf(att, sizeof(att), "blocks.%d.att.", i);
        snprintf(ffn, sizeof(ffn), "blocks.%d.ffn.", i);
        
        // Layer norm 1
        std::string ln1_w = std::string(bbb) + "ln1.weight";
        std::string ln1_b = std::string(bbb) + "ln1.bias";
        layer_norm_half8_fp16(x_current, get_weight_safe(model, ln1_w), 
                             get_weight_safe(model, ln1_b), x_ln, B * T, C, 1e-5f, stream);
        
        // TMix
        half* x_prev_layer = state0 + (i * 2 * B * C);
        half* state1_layer = state1 + (i * B * H * N * N);
        
        // ✅ 使用 bufs 中分配的临时缓冲区（不是 model->temp_bufs）
        tmix_seq_batch_fp16(
            i, H, N,
            x_ln, B, T,
            x_prev_layer,
            v_first,
            state1_layer,
            get_weight_safe(model, std::string(att) + "x_r"),
            get_weight_safe(model, std::string(att) + "x_w"),
            get_weight_safe(model, std::string(att) + "x_k"),
            get_weight_safe(model, std::string(att) + "x_v"),
            get_weight_safe(model, std::string(att) + "x_a"),
            get_weight_safe(model, std::string(att) + "x_g"),
            get_weight_safe(model, std::string(att) + "w0"),
            get_weight_safe(model, std::string(att) + "w1"),
            get_weight_safe(model, std::string(att) + "w2"),
            get_weight_safe(model, std::string(att) + "a0"),
            get_weight_safe(model, std::string(att) + "a1"),
            get_weight_safe(model, std::string(att) + "a2"),
            get_weight_safe(model, std::string(att) + "v0"),
            get_weight_safe(model, std::string(att) + "v1"),
            get_weight_safe(model, std::string(att) + "v2"),
            get_weight_safe(model, std::string(att) + "g1"),
            get_weight_safe(model, std::string(att) + "g2"),
            get_weight_safe(model, std::string(att) + "k_k"),
            get_weight_safe(model, std::string(att) + "k_a"),
            get_weight_safe(model, std::string(att) + "r_k"),
            get_weight_safe(model, std::string(att) + "receptance.weight"),
            get_weight_safe(model, std::string(att) + "key.weight"),
            get_weight_safe(model, std::string(att) + "value.weight"),
            get_weight_safe(model, std::string(att) + "output.weight"),
            get_weight_safe(model, std::string(att) + "ln_x.weight"),
            get_weight_safe(model, std::string(att) + "ln_x.bias"),
            state2,
            &bufs->tmix_seq_bufs[i],
            xx_tmix,
            stream
        );
        
        // 残差连接
        element_wise_add_fp16(x_current, xx_tmix, x_current, B * T * C, stream);
        
        // Layer norm 2
        std::string ln2_w = std::string(bbb) + "ln2.weight";
        std::string ln2_b = std::string(bbb) + "ln2.bias";
        layer_norm_half8_fp16(x_current, get_weight_safe(model, ln2_w), 
                             get_weight_safe(model, ln2_b), x_ln, B * T, C, 1e-5f, stream);
        
        // CMix
        half* x_prev_cmix = x_prev_layer;
        int D = model->ffn_dim;  // 使用保存的 ffn_dim
        
        // ✅ 使用 bufs 中分配的临时缓冲区（不是 model->temp_bufs）
        cmix_seq_batch_fp16(
            x_ln,
            x_prev_cmix,
            get_weight_safe(model, std::string(ffn) + "x_k"),
            get_weight_safe(model, std::string(ffn) + "key.weight"),
            get_weight_safe(model, std::string(ffn) + "value.weight"),
            &bufs->cmix_seq_bufs[i],
            xx_cmix,
            B, T, C, D,
            stream
        );
        
        // 残差连接
        element_wise_add_fp16(x_current, xx_cmix, x_current, B * T * C, stream);
    }
    
    // 最终 layer norm
    layer_norm_half8_fp16(x_current, get_weight_safe(model, "ln_out.weight"), 
                         get_weight_safe(model, "ln_out.bias"), x_ln, B * T, C, 1e-5f, stream);
    
    // 输出层
    if (full_output) {
        // 输出所有时间步: [B, T, vocab_size]
        linear_fp16(x_ln, get_weight_safe(model, "head.weight"), nullptr, output, B * T, vocab_size, C, stream);
    } else {
        // 只输出最后一个时间步: [B, vocab_size]
        // 对每个样本的最后一个时间步进行线性变换
        for (int b = 0; b < B; b++) {
            int actual_T = (lengths != nullptr) ? lengths[b] : T;
            CUDA_CHECK(cudaMemcpyAsync(
                x_last_batch + b * C,
                x_ln + (b * T + (actual_T - 1)) * C,
                C * sizeof(half),
                cudaMemcpyDeviceToDevice,
                stream
            ));
        }
        linear_fp16(x_last_batch, get_weight_safe(model, "head.weight"), nullptr, output, B, vocab_size, C, stream);
    }
    
    // 更新 elapsed time
    for (int b = 0; b < B; b++) {
        int actual_T = (lengths != nullptr) ? lengths[b] : T;
        int elapsed;
        CUDA_CHECK(cudaMemcpyAsync(&elapsed, state2 + b, sizeof(int), cudaMemcpyDeviceToHost, stream));
        elapsed += actual_T;
        CUDA_CHECK(cudaMemcpyAsync(state2 + b, &elapsed, sizeof(int), cudaMemcpyHostToDevice, stream));
    }
}

// 初始化 CUDA Graph（可选，用于性能优化）
// 注意：CUDA Graph 需要固定的输入形状和指针地址
bool init_cuda_graph(
    RWKVCudaGraph* graph,
    const RWKVModel* model,
    ForwardMode mode,
    int bsz,
    int seq_len,
    half* static_x,
    half* static_state0,
    half* static_state1,
    int* static_state2,
    half* static_output,
    RWKVForwardTempBuffers* temp_bufs,
    cudaStream_t stream
) {
    if (!graph || !model || !static_x || !static_state0 || !static_state1 || 
        !static_state2 || !static_output || !temp_bufs) {
        fprintf(stderr, "Error: Invalid pointers in init_cuda_graph\n");
        return false;
    }
    
    // 如果已经初始化，先销毁
    if (graph->initialized) {
        graph->destroy();
    }
    
    // 保存参数
    graph->static_x = static_x;
    graph->static_state0 = static_state0;
    graph->static_state1 = static_state1;
    graph->static_state2 = static_state2;
    graph->static_output = static_output;
    graph->temp_bufs = temp_bufs;
    graph->mode = mode;
    graph->bsz = bsz;
    graph->seq_len = seq_len;
    
    // 创建 stream 如果没有提供
    cudaStream_t capture_stream = stream;
    if (capture_stream == nullptr) {
        CUDA_CHECK(cudaStreamCreate(&capture_stream));
    }
    
    // 开始捕获
    CUDA_CHECK(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal));
    
    // 根据模式执行前向传播
    if (mode == ForwardMode::ONE) {
        forward_one(model, static_x, static_state0, static_state1, static_state2,
                   static_output, bsz, capture_stream);
    } else if (mode == ForwardMode::SEQ) {
        // 注意：forward_seq 需要 tokens，但 Graph 捕获时不支持 CPU 指针
        // 这里我们只捕获计算部分，token embedding 需要单独处理
        fprintf(stderr, "Warning: CUDA Graph for SEQ mode is not fully supported yet\n");
        CUDA_CHECK(cudaStreamEndCapture(capture_stream, &graph->graph));
        if (capture_stream != stream && stream == nullptr) {
            cudaStreamDestroy(capture_stream);
        }
        return false;
    } else { // SEQ_BATCH
        fprintf(stderr, "Warning: CUDA Graph for SEQ_BATCH mode is not fully supported yet\n");
        CUDA_CHECK(cudaStreamEndCapture(capture_stream, &graph->graph));
        if (capture_stream != stream && stream == nullptr) {
            cudaStreamDestroy(capture_stream);
        }
        return false;
    }
    
    // 结束捕获
    CUDA_CHECK(cudaStreamEndCapture(capture_stream, &graph->graph));
    
    // 实例化 Graph
    CUDA_CHECK(cudaGraphInstantiate(&graph->graph_exec, graph->graph, nullptr, nullptr, 0));
    
    if (capture_stream != stream && stream == nullptr) {
        cudaStreamDestroy(capture_stream);
    }
    
    graph->initialized = true;
    return true;
}


