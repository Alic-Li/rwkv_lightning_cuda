#ifndef LOAD_MODEL_CUH
#define LOAD_MODEL_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

// Safetensors 模型加载器

struct TensorInfo {
    std::vector<int64_t> shape;
    size_t data_offset;  // 在文件中的偏移量
    size_t data_size;   // 数据大小（字节）
    int dtype;           // 数据类型：0=float32, 1=float16, 2=int32, 3=int64
};

class SafeTensorsLoader {
public:
    SafeTensorsLoader();
    ~SafeTensorsLoader();
    
    // 加载 safetensors 文件
    // 返回是否成功
    bool load(const std::string& filepath);
    
    // 获取所有张量名称
    std::vector<std::string> get_tensor_names() const;
    
    // 获取张量信息
    bool get_tensor_info(const std::string& name, TensorInfo& info) const;
    
    // 加载张量到 GPU（fp16）
    // name: 张量名称
    // d_data: GPU 内存指针（已分配）
    // 返回是否成功
    bool load_tensor_to_gpu(const std::string& name, half* d_data) const;
    
    // 加载张量到 GPU（fp32）
    bool load_tensor_to_gpu_fp32(const std::string& name, float* d_data) const;
    
    // 获取张量元素数量
    size_t get_tensor_numel(const std::string& name) const;
    
    // 清理资源
    void clear();

private:
    std::unordered_map<std::string, TensorInfo> tensors_;
    void* file_data_;      // 文件数据（mmap 或读取到内存）
    size_t file_size_;
    bool is_mapped_;       // 是否使用 mmap
    
    // 解析 safetensors 头部
    bool parse_header(const uint8_t* header_data, size_t header_size);
    
    // 读取文件
    bool read_file(const std::string& filepath);
};

#endif // LOAD_MODEL_CUH

