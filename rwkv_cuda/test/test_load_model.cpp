#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../src/utils/load_model.cuh"

// 辅助函数：检查 CUDA 错误
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// 辅助函数：打印数组（用于调试）
void print_array(const half* data, size_t size, const char* name, size_t max_print = 10) {
    if (name && strlen(name) > 0) {
        std::cout << name << ": ";
    }
    for (size_t i = 0; i < std::min(size, max_print); i++) {
        std::cout << __half2float(data[i]) << " ";
    }
    if (size > max_print) std::cout << "...";
    std::cout << std::endl;
}

void print_array_fp32(const float* data, size_t size, const char* name, size_t max_print = 10) {
    if (name && strlen(name) > 0) {
        std::cout << name << ": ";
    }
    for (size_t i = 0; i < std::min(size, max_print); i++) {
        std::cout << data[i] << " ";
    }
    if (size > max_print) std::cout << "...";
    std::cout << std::endl;
}

// 测试加载 safetensors 文件
void test_load_safetensors(const std::string& filepath) {
    std::cout << "\n=== Testing SafeTensors Loader ===" << std::endl;
    std::cout << "File: " << filepath << std::endl;
    
    SafeTensorsLoader loader;
    
    // 加载文件
    if (!loader.load(filepath)) {
        std::cerr << "Failed to load safetensors file!" << std::endl;
        return;
    }
    
    // 获取所有张量名称
    std::vector<std::string> tensor_names = loader.get_tensor_names();
    
    // 自定义排序：按 block 顺序排列
    // 1. 非 blocks 的张量（emb, ln_out 等）放在最前面
    // 2. blocks.X 按照 X 的数字大小排序
    std::sort(tensor_names.begin(), tensor_names.end(), [](const std::string& a, const std::string& b) {
        bool a_is_block = (a.find("blocks.") == 0);
        bool b_is_block = (b.find("blocks.") == 0);
        
        // 非 block 张量排在前面
        if (!a_is_block && b_is_block) return true;
        if (a_is_block && !b_is_block) return false;
        
        // 如果都不是 block，按字典序
        if (!a_is_block && !b_is_block) return a < b;
        
        // 都是 block，提取 block 编号
        size_t a_dot = a.find('.', 7);  // 找到 "blocks.X." 中的第二个点
        size_t b_dot = b.find('.', 7);
        
        if (a_dot == std::string::npos || b_dot == std::string::npos) {
            return a < b;  // 如果格式不对，回退到字典序
        }
        
        int a_block = std::stoi(a.substr(7, a_dot - 7));
        int b_block = std::stoi(b.substr(7, b_dot - 7));
        
        // 先按 block 编号排序
        if (a_block != b_block) {
            return a_block < b_block;
        }
        
        // 同一 block 内，按字典序
        return a < b;
    });
    
    std::cout << "\nFound " << tensor_names.size() << " tensors (sorted by block order):" << std::endl;
    
    // 显示所有张量信息
    for (size_t i = 0; i < tensor_names.size(); i++) {
        const std::string& name = tensor_names[i];
        TensorInfo info;
        
        if (loader.get_tensor_info(name, info)) {
            std::cout << "  [" << i << "] " << name << ": ";
            std::cout << "shape=[";
            for (size_t j = 0; j < info.shape.size(); j++) {
                std::cout << info.shape[j];
                if (j < info.shape.size() - 1) std::cout << ", ";
            }
            std::cout << "], dtype=";
            switch (info.dtype) {
                case 0: std::cout << "F32"; break;
                case 1: std::cout << "F16"; break;
                case 2: std::cout << "I32"; break;
                case 3: std::cout << "I64"; break;
                default: std::cout << "UNKNOWN"; break;
            }
            std::cout << ", size=" << info.data_size << " bytes";
            std::cout << ", numel=" << loader.get_tensor_numel(name);
            std::cout << std::endl;
        }
    }
    
    // 测试加载几个张量到 GPU
    std::cout << "\n=== Testing GPU Loading ===" << std::endl;
    
    // 尝试加载前几个 F16 张量
    int loaded_count = 0;
    for (const std::string& name : tensor_names) {
        if (loaded_count >= 3) break;  // 只测试前 3 个
        
        TensorInfo info;
        if (!loader.get_tensor_info(name, info)) {
            continue;
        }
        
        // 只测试 F16 张量
        if (info.dtype != 1) {  // 1 = F16
            continue;
        }
        
        size_t numel = loader.get_tensor_numel(name);
        if (numel == 0) {
            continue;
        }
        
        std::cout << "\nLoading tensor: " << name << std::endl;
        std::cout << "  Shape: [";
        for (size_t j = 0; j < info.shape.size(); j++) {
            std::cout << info.shape[j];
            if (j < info.shape.size() - 1) std::cout << ", ";
        }
        std::cout << "], Numel: " << numel << std::endl;
        
        // 分配 GPU 内存
        half* d_data;
        CUDA_CHECK(cudaMalloc(&d_data, numel * sizeof(half)));
        
        // 加载到 GPU
        if (loader.load_tensor_to_gpu(name, d_data)) {
            std::cout << "  ✓ Successfully loaded to GPU" << std::endl;
            
            // 复制回主机验证
            std::vector<half> h_data(numel);
            CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, numel * sizeof(half), cudaMemcpyDeviceToHost));
            
            std::cout << "  Sample values: ";
            print_array(h_data.data(), numel, "", 5);
            
            loaded_count++;
        } else {
            std::cerr << "  ✗ Failed to load to GPU" << std::endl;
        }
        
        CUDA_CHECK(cudaFree(d_data));
    }
    
    if (loaded_count == 0) {
        std::cout << "No F16 tensors found to test GPU loading" << std::endl;
    }
    
    std::cout << "\n✓ All tests completed!" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "RWKV SafeTensors Loader Test" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // 检查 CUDA 设备
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    
    // 获取文件路径
    std::string filepath;
    if (argc >= 2) {
        filepath = argv[1];
    } else {
        std::cout << "\nUsage: " << argv[0] << " <safetensors_file.st>" << std::endl;
        std::cout << "Example: " << argv[0] << " model.st" << std::endl;
        std::cout << "\nPlease provide a safetensors file path." << std::endl;
        return 1;
    }
    
    // 运行测试
    test_load_safetensors(filepath);
    
    return 0;
}

