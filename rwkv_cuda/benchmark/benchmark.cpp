#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../src/rwkv7.cuh"
#include "../src/utils/load_model.cuh"
#include "../src/utils/sampler.cuh"
#include "../src/utils/tokenizer.cuh"

// 可选的 CUDA Graph 测试（仅在 forward_one 模式下有效）
#define USE_CUDA_GRAPH 1  // 设置为 1 以启用 CUDA Graph（实验性）

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// 计算百分位数
double percentile(std::vector<double>& times, double p) {
    if (times.empty()) return 0.0;
    std::sort(times.begin(), times.end());
    int idx = static_cast<int>(times.size() * p / 100.0);
    if (idx >= static_cast<int>(times.size())) idx = times.size() - 1;
    return times[idx];
}

// 打印分隔线
void print_separator(const std::string& title) {
    int len = 80;
    int title_len = title.length();
    int left = 3;
    int right = len - title_len - left;
    std::cout << "\n" << std::string(left, '#') << " " << title 
              << " " << std::string(right, '#') << "\n" << std::endl;
}

// 单样本解码性能测试
void benchmark_decode_one(const RWKVModel* model, trie_tokenizer& tokenizer, const std::string& prompt, int length_per_trial = 256) {
    print_separator("Decode (Single Sample)");
    
    int vocab_size = model->vocab_size;
    int C = model->n_embd;
    half* emb_weight = model->get_weight("emb.weight");
    
    // 生成零状态
    half* state0 = nullptr;
    half* state1 = nullptr;
    int* state2 = nullptr;
    generate_zero_state(model, 0, &state0, &state1, &state2, nullptr);
    
    // 分配输出缓冲区和输入缓冲区
    half* output;
    CUDA_CHECK(cudaMalloc(&output, vocab_size * sizeof(half)));
    
    half* x_input;  // 用于 forward_one 的输入
    CUDA_CHECK(cudaMalloc(&x_input, C * sizeof(half)));
    
    // 使用 tokenizer 编码 prompt
    std::vector<int> prompt_tokens = tokenizer.encode(prompt);
    if (prompt_tokens.empty()) {
        prompt_tokens.push_back(0);
    }
    
    // 先用 prompt 初始化状态（使用 forward_seq）
    forward_seq(model, prompt_tokens.data(), static_cast<int>(prompt_tokens.size()), state0, state1, state2, output, false, nullptr);
    cudaDeviceSynchronize();
    
    // 可选的 CUDA Graph 初始化（实验性）
    RWKVCudaGraph* graph = nullptr;
    RWKVForwardTempBuffers* temp_bufs = nullptr;
    half* static_x = nullptr;
    half* static_output = nullptr;
    
#if USE_CUDA_GRAPH
    // 分配静态输入输出缓冲区（用于 CUDA Graph）
    CUDA_CHECK(cudaMalloc(&static_x, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&static_output, vocab_size * sizeof(half)));
    
    // 分配临时缓冲区
    temp_bufs = allocate_forward_temp_buffers(model, ForwardMode::ONE, 1, 1);
    
    // 初始化 CUDA Graph
    graph = new RWKVCudaGraph();
    if (init_cuda_graph(graph, model, ForwardMode::ONE, 0, 1, 
                        static_x, state0, state1, state2, static_output,
                        temp_bufs, nullptr)) {
        std::cout << "CUDA Graph initialized successfully" << std::endl;
    } else {
        std::cout << "CUDA Graph initialization failed, using normal mode" << std::endl;
        delete graph;
        graph = nullptr;
        free_forward_temp_buffers(temp_bufs);
        temp_bufs = nullptr;
    }
#endif
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        int dummy_token = 0;
        CUDA_CHECK(cudaMemcpy(x_input, emb_weight + dummy_token * C, C * sizeof(half), cudaMemcpyDeviceToDevice));
        forward_one(model, x_input, state0, state1, state2, output, 0, nullptr);
        cudaDeviceSynchronize();
    }
    
    // 测试
    std::vector<double> times;
    std::vector<double> all_times;
    
    auto t000 = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < length_per_trial; i++) {
        auto t00 = std::chrono::high_resolution_clock::now();
        
        // 采样：使用 sampler_simple
        int sampled_token = sampler_simple(output, vocab_size, 0.0f, 1.0f);
        
        // 准备输入（从 embedding 中获取）
        CUDA_CHECK(cudaMemcpy(x_input, emb_weight + sampled_token * C, C * sizeof(half), cudaMemcpyDeviceToDevice));
        
        // 前向传播
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        
#if USE_CUDA_GRAPH
        if (graph && graph->initialized) {
            // 使用 CUDA Graph
            CUDA_CHECK(cudaMemcpy(graph->static_x, x_input, C * sizeof(half), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaGraphLaunch(graph->graph_exec, nullptr));
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(output, graph->static_output, vocab_size * sizeof(half), cudaMemcpyDeviceToDevice));
        } else {
            // 正常模式
            forward_one(model, x_input, state0, state1, state2, output, 0, nullptr);
        }
#else
        forward_one(model, x_input, state0, state1, state2, output, 0, nullptr);
#endif
        
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        
        double forward_time = std::chrono::duration<double>(t1 - t0).count();
        double total_time = std::chrono::duration<double>(t1 - t00).count();
        
        times.push_back(forward_time);
        all_times.push_back(total_time);
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(t_end - t000).count();
    
    // 计算统计信息
    double p50_forward = percentile(times, 50.0);
    double p50_total = percentile(all_times, 50.0);
    
    double tokens_per_sec_forward = 1.0 / p50_forward;
    double tokens_per_sec_total = 1.0 / p50_total;
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Token/s = " << tokens_per_sec_forward 
              << " (forward), " << tokens_per_sec_total << " (full)" << std::endl;
    std::cout << "Total time = " << total_elapsed << "s" << std::endl;
    
#if USE_CUDA_GRAPH
    if (graph) {
        delete graph;
    }
    if (temp_bufs) {
        free_forward_temp_buffers(temp_bufs);
    }
    if (static_x) {
        cudaFree(static_x);
    }
    if (static_output) {
        cudaFree(static_output);
    }
#endif
    
    // 清理
    cudaFree(state0);
    cudaFree(state1);
    cudaFree(state2);
    cudaFree(output);
    cudaFree(x_input);
}

// 批量解码性能测试
void benchmark_decode_batch(const RWKVModel* model, 
                            const std::vector<int>& batch_sizes,
                            trie_tokenizer& tokenizer,
                            int length_per_trial = 32) {
    print_separator("Decode (Batch)");
    
    int vocab_size = model->vocab_size;
    
    for (int bsz : batch_sizes) {
        // 生成零状态
        half* state0 = nullptr;
        half* state1 = nullptr;
        int* state2 = nullptr;
        generate_zero_state(model, bsz, &state0, &state1, &state2, nullptr);
        
        // 分配输出缓冲区
        half* output;
        CUDA_CHECK(cudaMalloc(&output, bsz * vocab_size * sizeof(half)));
        
        // 准备初始 prompt
        std::vector<std::string> prompts(bsz, "The apple can be");
        if (bsz == 2) {
            prompts[0] = "The apple can be";
            prompts[1] = "The cat can't be";
        }
        std::vector<std::vector<int>> prompt_tokens(bsz);
        std::vector<int> lengths(bsz, 0);
        int max_len = 0;
        for (int b = 0; b < bsz; b++) {
            prompt_tokens[b] = tokenizer.encode(prompts[b]);
            if (prompt_tokens[b].empty()) prompt_tokens[b].push_back(0);
            lengths[b] = static_cast<int>(prompt_tokens[b].size());
            max_len = std::max(max_len, lengths[b]);
        }
        std::vector<int> tokens(bsz * max_len, 0);
        for (int b = 0; b < bsz; b++) {
            for (int t = 0; t < max_len; t++) {
                int idx = b * max_len + t;
                tokens[idx] = (t < lengths[b]) ? prompt_tokens[b][t] : prompt_tokens[b].back();
            }
        }
        
        // Warmup
        for (int i = 0; i < 5; i++) {
            // 使用 forward_seq_batch 进行批量前向传播（使用完整 prompt）
            forward_seq_batch(model, tokens.data(), bsz, max_len, lengths.data(), 
                            state0, state1, state2, output, false, nullptr);
            cudaDeviceSynchronize();
        }
        
        // 测试
        std::vector<double> times;
        std::vector<double> all_times;
        
        // 用于存储采样的 token IDs（CPU 内存）
        std::vector<int> sampled_tokens(bsz);
        
        auto t000 = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < length_per_trial; i++) {
            auto t00 = std::chrono::high_resolution_clock::now();
            
            // 批量采样：使用 sampler_simple_batch
            sampler_simple_batch(output, sampled_tokens.data(), bsz, vocab_size, 0.0f, 1.0f);
            
            cudaDeviceSynchronize();
            auto t0 = std::chrono::high_resolution_clock::now();
            
            // 批量前向传播：使用 forward_seq_batch（T=1）
            // tokens 参数需要在 CPU 内存
            forward_seq_batch(model, sampled_tokens.data(), bsz, 1, nullptr, 
                            state0, state1, state2, output, false, nullptr);
            
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            
            double forward_time = std::chrono::duration<double>(t1 - t0).count();
            double total_time = std::chrono::duration<double>(t1 - t00).count();
            
            times.push_back(forward_time);
            all_times.push_back(total_time);
        }
        
        auto t_end = std::chrono::high_resolution_clock::now();
        double total_elapsed = std::chrono::duration<double>(t_end - t000).count();
        
        // 计算统计信息
        double p50_forward = percentile(times, 50.0);
        double p50_total = percentile(all_times, 50.0);
        
        double tokens_per_sec_forward = bsz / p50_forward;
        double tokens_per_sec_total = bsz / p50_total;
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Bsz " << bsz << " || Token/s = " << tokens_per_sec_forward 
                  << " (forward), " << tokens_per_sec_total << " (full) || "
                  << total_elapsed << "s" << std::endl;
        
        // 清理
        cudaFree(state0);
        cudaFree(state1);
        cudaFree(state2);
        cudaFree(output);
        
        // 清理 GPU 缓存
        cudaDeviceSynchronize();
    }
}

// 序列前向传播性能测试
void benchmark_forward_seq(const RWKVModel* model, trie_tokenizer& tokenizer, const std::string& prompt, int seq_length = 512) {
    print_separator("Forward Sequence");
    
    int vocab_size = model->vocab_size;
    
    // 生成零状态
    half* state0 = nullptr;
    half* state1 = nullptr;
    int* state2 = nullptr;
    generate_zero_state(model, 0, &state0, &state1, &state2, nullptr);
    
    // 分配输出缓冲区
    half* output;
    CUDA_CHECK(cudaMalloc(&output, vocab_size * sizeof(half)));
    
    // 创建测试 tokens
    std::vector<int> tokens = tokenizer.encode(prompt);
    if (tokens.empty()) tokens.push_back(0);
    if (static_cast<int>(tokens.size()) < seq_length) {
        tokens.resize(seq_length, tokens.back());
    } else if (static_cast<int>(tokens.size()) > seq_length) {
        tokens.resize(seq_length);
    }
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        forward_seq(model, tokens.data(), seq_length, state0, state1, state2, 
                   output, false, nullptr);
        cudaDeviceSynchronize();
    }
    
    // 测试
    std::vector<double> times;
    
    for (int i = 0; i < 10; i++) {
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        
        forward_seq(model, tokens.data(), seq_length, state0, state1, state2, 
                   output, false, nullptr);
        
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        
        double time = std::chrono::duration<double>(t1 - t0).count();
        times.push_back(time);
    }
    
    // 计算统计信息
    double p50 = percentile(times, 50.0);
    double tokens_per_sec = seq_length / p50;
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Sequence length: " << seq_length << std::endl;
    std::cout << "Time: " << p50 << "s" << std::endl;
    std::cout << "Tokens/s: " << tokens_per_sec << std::endl;
    
    // 清理
    cudaFree(state0);
    cudaFree(state1);
    cudaFree(state2);
    cudaFree(output);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path.safetensors>" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    
    std::cout << "\nUsing CUDA fp16. Loading " << model_path << " ...\n" << std::endl;
    
    // 创建模型
    RWKVModel model;
    if (!model.load_from_safetensors(model_path)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    
    // 打印模型信息
    std::cout << "\nModel config:" << std::endl;
    std::cout << "  n_layer: " << model.n_layer << std::endl;
    std::cout << "  n_embd: " << model.n_embd << std::endl;
    std::cout << "  n_head: " << model.n_head << std::endl;
    std::cout << "  head_size: " << model.head_size << std::endl;
    std::cout << "  vocab_size: " << model.vocab_size << std::endl;
    
    // 加载 tokenizer
    trie_tokenizer tokenizer;
    std::string vocab_path = "./src/utils/rwkv_vocab_v20230424.txt";
    if (tokenizer.load(vocab_path) != RWKV_SUCCESS) {
        std::cerr << "Failed to load tokenizer vocab from " << vocab_path << std::endl;
        return 1;
    }
    
    std::string prompt = "The Eiffel tower is in the city of";
    
    // 运行基准测试
    benchmark_decode_one(&model, tokenizer, prompt, 256);
    
    std::vector<int> batch_sizes = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    benchmark_decode_batch(&model, batch_sizes, tokenizer, 32);
    
    benchmark_forward_seq(&model, tokenizer, prompt, 512);
    
    std::cout << "\nBenchmark completed!" << std::endl;
    
    return 0;
}

