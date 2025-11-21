#include "load_model.cuh"
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

SafeTensorsLoader::SafeTensorsLoader() 
    : file_data_(nullptr), file_size_(0), is_mapped_(false) {
}

SafeTensorsLoader::~SafeTensorsLoader() {
    clear();
}

bool SafeTensorsLoader::read_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return false;
    }
    
    file_size_ = file.tellg();
    file.seekg(0, std::ios::beg);
    
    file_data_ = malloc(file_size_);
    if (!file_data_) {
        std::cerr << "Failed to allocate memory for file" << std::endl;
        return false;
    }
    
    file.read(reinterpret_cast<char*>(file_data_), file_size_);
    file.close();
    
    is_mapped_ = false;
    return true;
}

bool SafeTensorsLoader::parse_header(const uint8_t* header_data, size_t header_size) {
    // Safetensors 格式：
    // - 前 8 字节：头部长度（little-endian uint64）
    // - 接下来 header_len 字节：JSON 格式的头部
    // - 剩余：张量数据
    
    if (header_size < 8) {
        std::cerr << "Invalid safetensors file: header too short" << std::endl;
        return false;
    }
    
    // 读取头部长度
    uint64_t header_len = 0;
    memcpy(&header_len, header_data, 8);
    
    if (header_len + 8 > header_size) {
        std::cerr << "Invalid header length" << std::endl;
        return false;
    }
    
    // 解析 JSON 头部
    std::string header_json(reinterpret_cast<const char*>(header_data + 8), header_len);
    
    try {
        nlohmann::json j = nlohmann::json::parse(header_json);
        
        // 遍历所有张量
        for (auto& [key, value] : j.items()) {
            TensorInfo info;
            
            // 解析 dtype
            std::string dtype_str = value["dtype"].get<std::string>();
            if (dtype_str == "F32" || dtype_str == "FLOAT") {
                info.dtype = 0;  // float32
            } else if (dtype_str == "F16" || dtype_str == "HALF") {
                info.dtype = 1;  // float16
            } else if (dtype_str == "I32" || dtype_str == "INT") {
                info.dtype = 2;  // int32
            } else if (dtype_str == "I64" || dtype_str == "LONG") {
                info.dtype = 3;  // int64
            } else {
                std::cerr << "Unsupported dtype: " << dtype_str << " for tensor: " << key << std::endl;
                continue;
            }
            
            // 解析 shape
            info.shape = value["shape"].get<std::vector<int64_t>>();
            
            // 解析 data_offsets [start, end]
            std::vector<int64_t> offsets = value["data_offsets"].get<std::vector<int64_t>>();
            if (offsets.size() != 2) {
                std::cerr << "Invalid data_offsets for tensor: " << key << std::endl;
                continue;
            }
            
            info.data_offset = offsets[0];
            info.data_size = offsets[1] - offsets[0];
            
            // 验证数据大小
            size_t expected_size = info.data_size;
            size_t numel = 1;
            for (int64_t dim : info.shape) {
                numel *= dim;
            }
            
            size_t dtype_size = (info.dtype == 0) ? 4 : (info.dtype == 1) ? 2 : (info.dtype == 2) ? 4 : 8;
            size_t calculated_size = numel * dtype_size;
            
            if (expected_size != calculated_size) {
                std::cerr << "Warning: Size mismatch for tensor " << key 
                          << ": expected " << expected_size 
                          << ", calculated " << calculated_size << std::endl;
            }
            
            tensors_[key] = info;
        }
        
        return true;
    } catch (const nlohmann::json::exception& e) {
        std::cerr << "JSON parsing error: " << e.what() << std::endl;
        return false;
    }
}

bool SafeTensorsLoader::load(const std::string& filepath) {
    clear();
    
    if (!read_file(filepath)) {
        return false;
    }
    
    const uint8_t* data = reinterpret_cast<const uint8_t*>(file_data_);
    
    // 读取头部长度
    if (file_size_ < 8) {
        std::cerr << "File too small" << std::endl;
        return false;
    }
    
    uint64_t header_len = 0;
    memcpy(&header_len, data, 8);
    
    if (header_len + 8 > file_size_) {
        std::cerr << "Invalid header length" << std::endl;
        return false;
    }
    
    // 解析头部（使用 nlohmann/json）
    if (!parse_header(data, header_len + 8)) {
        return false;
    }
    
    std::cout << "Loaded " << tensors_.size() << " tensors from safetensors file" << std::endl;
    
    return true;
}

std::vector<std::string> SafeTensorsLoader::get_tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& pair : tensors_) {
        names.push_back(pair.first);
    }
    return names;
}

bool SafeTensorsLoader::get_tensor_info(const std::string& name, TensorInfo& info) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return false;
    }
    info = it->second;
    return true;
}

size_t SafeTensorsLoader::get_tensor_numel(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return 0;
    }
    
    size_t numel = 1;
    for (int64_t dim : it->second.shape) {
        numel *= dim;
    }
    return numel;
}

bool SafeTensorsLoader::load_tensor_to_gpu(const std::string& name, half* d_data) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        std::cerr << "Tensor not found: " << name << std::endl;
        return false;
    }
    
    const TensorInfo& info = it->second;
    
    // 检查数据类型
    if (info.dtype != 1) {  // 1 = float16
        std::cerr << "Tensor " << name << " is not float16" << std::endl;
        return false;
    }
    
    // 计算数据位置
    const uint8_t* file_data = reinterpret_cast<const uint8_t*>(file_data_);
    uint64_t header_len = 0;
    memcpy(&header_len, file_data, 8);
    
    const half* src_data = reinterpret_cast<const half*>(
        file_data + 8 + header_len + info.data_offset
    );
    
    // 复制到 GPU
    size_t numel = get_tensor_numel(name);
    cudaError_t err = cudaMemcpy(d_data, src_data, numel * sizeof(half), cudaMemcpyHostToDevice);
    
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    return true;
}

bool SafeTensorsLoader::load_tensor_to_gpu_fp32(const std::string& name, float* d_data) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        std::cerr << "Tensor not found: " << name << std::endl;
        return false;
    }
    
    const TensorInfo& info = it->second;
    
    // 计算数据位置
    const uint8_t* file_data = reinterpret_cast<const uint8_t*>(file_data_);
    uint64_t header_len = 0;
    memcpy(&header_len, file_data, 8);
    
    const void* src_data = file_data + 8 + header_len + info.data_offset;
    
    size_t numel = get_tensor_numel(name);
    
    // 根据数据类型转换
    if (info.dtype == 0) {  // float32
        cudaMemcpy(d_data, src_data, numel * sizeof(float), cudaMemcpyHostToDevice);
    } else if (info.dtype == 1) {  // float16
        // 需要转换
        std::vector<float> temp(numel);
        const half* src_half = reinterpret_cast<const half*>(src_data);
        for (size_t i = 0; i < numel; i++) {
            temp[i] = __half2float(src_half[i]);
        }
        cudaMemcpy(d_data, temp.data(), numel * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        std::cerr << "Unsupported dtype for fp32 conversion" << std::endl;
        return false;
    }
    
    return true;
}

void SafeTensorsLoader::clear() {
    if (file_data_) {
        if (is_mapped_) {
#ifdef _WIN32
            UnmapViewOfFile(file_data_);
#else
            munmap(file_data_, file_size_);
#endif
        } else {
            free(file_data_);
        }
        file_data_ = nullptr;
    }
    file_size_ = 0;
    tensors_.clear();
}

