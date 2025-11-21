#ifndef RWKV7_CUH
#define RWKV7_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

// RWKV-7 模型 CUDA 实现 Only support fp16 precision

// 前向传播模式
enum class ForwardMode {
    ONE,      // 单 token 前向传播
    SEQ,      // 序列前向传播
    SEQ_BATCH // 批量序列前向传播
};

// 前向传播临时缓冲区管理器
struct RWKVForwardTempBuffers {
    // Tmix 临时缓冲区（每层一个）
    struct TmixOneTempBuffers* tmix_one_bufs;     // [n_layer] - 用于 forward_one
    struct TmixSeqTempBuffers* tmix_seq_bufs; // [n_layer] - 用于 forward_seq/forward_seq_batch
    
    // Cmix 临时缓冲区（每层一个）
    struct CmixOneTempBuffers* cmix_one_bufs;     // [n_layer] - 用于 forward_one
    struct CmixSeqTempBuffers* cmix_seq_bufs;     // [n_layer] - 用于 forward_seq/forward_seq_batch
    
    // 通用临时缓冲区
    half* x_current;   // [C] 或 [T*C] 或 [B*T*C] - 当前层输入
    half* x_ln;        // [C] 或 [T*C] 或 [B*T*C] - Layer norm 输出
    half* xx_tmix;     // [C] 或 [T*C] 或 [B*T*C] - TMix 输出
    half* xx_cmix;     // [C] 或 [T*C] 或 [B*T*C] - CMix 输出
    half* v_first;     // [C] 或 [B*C] - v_first（每层共享，大小取决于 bsz）
    
    // 序列特有缓冲区
    half* x_seq;       // [T*C] 或 [B*T*C] - 输入序列（仅用于 forward_seq/forward_seq_batch）
    half* x_last_batch;// [B*C] - 批量最后时间步（仅用于 forward_seq_batch）
    
    int n_layer;
    int max_bsz;       // 最大批次大小
    int max_seq_len;   // 最大序列长度
    bool initialized;  // 是否已初始化
};

// CUDA Graph 管理器
struct RWKVCudaGraph {
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    bool initialized;
    
    // 用于 Graph 捕获的静态输入输出
    half* static_x;           // [C] 或 [T*C] 或 [B*T*C]
    half* static_state0;      // 状态0
    half* static_state1;      // 状态1
    int* static_state2;       // 状态2
    half* static_output;      // 输出
    
    // 临时缓冲区（在 Graph 中使用）
    RWKVForwardTempBuffers* temp_bufs;
    
    ForwardMode mode;
    int bsz;
    int seq_len;
    
    RWKVCudaGraph() : graph(nullptr), graph_exec(nullptr), initialized(false),
                     static_x(nullptr), static_state0(nullptr), static_state1(nullptr),
                     static_state2(nullptr), static_output(nullptr), temp_bufs(nullptr),
                     mode(ForwardMode::ONE), bsz(0), seq_len(0) {}
    
    ~RWKVCudaGraph() {
        destroy();
    }
    
    void destroy() {
        if (graph_exec != nullptr) {
            cudaGraphExecDestroy(graph_exec);
            graph_exec = nullptr;
        }
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
            graph = nullptr;
        }
        initialized = false;
    }
};

// RWKV 模型结构
struct RWKVModel {
    int n_layer;      // 层数
    int n_embd;       // 嵌入维度
    int n_head;       // 头数
    int head_size;    // 头大小（固定为 64）没试过大的或小的哈哈 我也不敢试
    int vocab_size;   // 词汇表大小
    int ffn_dim;      // FFN 维度（CMix 的 D）
    
    // 权重存储（GPU Memory）
    std::unordered_map<std::string, half*> weights;
    
    // 权重形状信息
    std::unordered_map<std::string, std::vector<int64_t>> weight_shapes;
    
    // 构造函数
    RWKVModel();
    ~RWKVModel();
    
    // 从 safetensors 文件加载模型
    bool load_from_safetensors(const std::string& model_path);
    
    // 清理资源
    void clear();
    
    // 获取权重指针
    half* get_weight(const std::string& name) const;
    
    // 检查权重是否存在
    bool has_weight(const std::string& name) const;
    
    // 获取权重形状
    std::vector<int64_t> get_weight_shape(const std::string& name) const;
};

// 生成零状态
// state[0]: [n_layer, 2, bsz, n_embd] 或 [n_layer, 2, n_embd] (bsz=0)
// state[1]: [n_layer, bsz, H, N, N] 或 [n_layer, H, N, N] (bsz=0)
// state[2]: [bsz] 或 scalar (bsz=0)
// 返回: state[0], state[1], state[2] 的 GPU 指针
void generate_zero_state(
    const RWKVModel* model,
    int bsz,
    half** state0,  // [n_layer, 2, bsz, n_embd] or [n_layer, 2, n_embd]
    half** state1,  // [n_layer, bsz, H, N, N] or [n_layer, H, N, N]
    int** state2,   // [bsz] or scalar
    cudaStream_t stream = nullptr
);

// 分配前向传播临时缓冲区
// 根据模式分配所需的缓冲区
RWKVForwardTempBuffers* allocate_forward_temp_buffers(
    const RWKVModel* model,
    ForwardMode mode,
    int max_bsz = 1,      // 最大批次大小（用于 SEQ_BATCH）
    int max_seq_len = 1   // 最大序列长度（用于 SEQ/SEQ_BATCH）
);

// 释放前向传播临时缓冲区
void free_forward_temp_buffers(RWKVForwardTempBuffers* bufs);

// 初始化 CUDA Graph（可选，用于性能优化）
// 需要在 warmup 后调用
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
    cudaStream_t stream = nullptr
);

// 单 token 前向传播
// x: [C] - 输入嵌入向量（GPU 内存）
// state0: [n_layer, 2, bsz, n_embd] 或 [n_layer, 2, n_embd] - 状态0（会被修改）
// state1: [n_layer, bsz, H, N, N] 或 [n_layer, H, N, N] - 状态1（会被修改）
// state2: [bsz] 或 scalar - 经过的时间（会被修改）
// output: [vocab_size] - 输出 logits（GPU 内存，已分配）
// bsz: batch size (0 表示单样本)
void forward_one(
    const RWKVModel* model,
    const half* x,
    half* state0,
    half* state1,
    int* state2,
    half* output,
    int bsz = 0,
    cudaStream_t stream = nullptr
);

// 序列前向传播
// tokens: [T] - token IDs（CPU 内存）
// state0: [n_layer, 2, n_embd] - 状态0（会被修改）
// state1: [n_layer, H, N, N] - 状态1（会被修改）
// state2: scalar - 经过的时间（会被修改）
// output: [vocab_size] 或 [T, vocab_size] - 输出 logits（GPU 内存，已分配）
// full_output: 是否输出所有时间步
void forward_seq(
    const RWKVModel* model,
    const int* tokens,
    int T,
    half* state0,
    half* state1,
    int* state2,
    half* output,
    bool full_output = false,
    cudaStream_t stream = nullptr
);

// 批量序列前向传播
// tokens: [B, T] - token IDs（CPU 内存，每行是一个序列）
// lengths: [B] - 每个序列的实际长度（CPU 内存，可选，如果为 nullptr 则假设所有序列长度相同）
// state0: [n_layer, 2, B, n_embd] - 状态0（会被修改）
// state1: [n_layer, B, H, N, N] - 状态1（会被修改）
// state2: [B] - 经过的时间（会被修改）
// output: [B, vocab_size] 或 [B, T, vocab_size] - 输出 logits（GPU 内存，已分配）
// full_output: 是否输出所有时间步
void forward_seq_batch(
    const RWKVModel* model,
    const int* tokens,
    int B,
    int T,
    const int* lengths,  // 可选，如果为 nullptr 则假设所有序列长度相同
    half* state0,
    half* state1,
    int* state2,
    half* output,
    bool full_output = false,
    cudaStream_t stream = nullptr
);

#endif // RWKV7_CUH

