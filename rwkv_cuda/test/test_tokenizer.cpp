#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include "../src/utils/tokenizer.cuh"

#ifndef RWKV_SUCCESS
#define RWKV_SUCCESS 0
#endif

#ifndef RWKV_ERROR_TOKENIZER
#define RWKV_ERROR_TOKENIZER -1
#endif

// 测试加载词汇表文件
void test_load() {
    std::cout << "\n=== Testing tokenizer load ===" << std::endl;
    
    trie_tokenizer tokenizer;
    std::string vocab_path = "rwkv_cuda/src/utils/rwkv_vocab_v20230424.txt";
    tokenizer.load(vocab_path);
}

// 测试编码功能
void test_encode(const std::string& vocab_path) {
    std::cout << "\n=== Testing tokenizer encode ===" << std::endl;
    
    trie_tokenizer tokenizer;
    int result = tokenizer.load(vocab_path);
    if (result != RWKV_SUCCESS) {
        std::cout << "✗ Failed to load vocab file for encode test" << std::endl;
        return;
    }
    
    // 测试编码简单字符串
    std::string test_str = "Hello";
    std::vector<int> tokens = tokenizer.encode(test_str);
    std::cout << "Encode \"" << test_str << "\": ";
    for (size_t i = 0; i < tokens.size(); i++) {
        std::cout << tokens[i];
        if (i < tokens.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    
    if (!tokens.empty()) {
        std::cout << "✓ Encode returned " << tokens.size() << " token(s)" << std::endl;
    } else {
        std::cout << "✗ Encode returned empty result" << std::endl;
    }
    
    // 测试编码更复杂的字符串
    test_str = "The quick brown fox";
    tokens = tokenizer.encode(test_str);
    std::cout << "Encode \"" << test_str << "\": " << tokens.size() << " token(s)" << std::endl;
    
    // 测试编码空字符串
    test_str = "";
    tokens = tokenizer.encode(test_str);
    std::cout << "Encode empty string: " << tokens.size() << " token(s)" << std::endl;
    
    // 测试编码中文字符串
    test_str = "你好";
    tokens = tokenizer.encode(test_str);
    std::cout << "Encode \"" << test_str << "\": " << tokens.size() << " token(s)" << std::endl;
}

// 测试解码功能
void test_decode(const std::string& vocab_path) {
    std::cout << "\n=== Testing tokenizer decode ===" << std::endl;
    
    trie_tokenizer tokenizer;
    int result = tokenizer.load(vocab_path);
    if (result != RWKV_SUCCESS) {
        std::cout << "✗ Failed to load vocab file for decode test" << std::endl;
        return;
    }
    
    // 测试解码单个token ID
    int token_id = 0;
    std::string decoded = tokenizer.decode(token_id);
    std::cout << "Decode token " << token_id << ": \"" << decoded << "\"" << std::endl;
    
    // 测试解码多个token IDs
    std::vector<int> token_ids = {0, 1, 2};
    decoded = tokenizer.decode(token_ids);
    std::cout << "Decode tokens [0, 1, 2]: \"" << decoded << "\"" << std::endl;
    
    // 先编码一个字符串，再解码，测试往返一致性
    std::string original = "Hello world";
    std::vector<int> encoded = tokenizer.encode(original);
    std::string roundtrip = tokenizer.decode(encoded);
    std::cout << "Roundtrip test:" << std::endl;
    std::cout << "  Original: \"" << original << "\"" << std::endl;
    std::cout << "  Encoded: [";
    for (size_t i = 0; i < encoded.size(); i++) {
        std::cout << encoded[i];
        if (i < encoded.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  Decoded: \"" << roundtrip << "\"" << std::endl;
    
    if (original == roundtrip) {
        std::cout << "✓ Roundtrip test passed!" << std::endl;
    } else {
        std::cout << "✗ Roundtrip test failed (original != decoded)" << std::endl;
    }
    
    // 测试更多往返案例
    std::vector<std::string> test_strings = {
        "The",
        "test",
        "Hello, world!",
        "123",
        "A"
    };
    
    int passed = 0;
    for (const auto& str : test_strings) {
        encoded = tokenizer.encode(str);
        roundtrip = tokenizer.decode(encoded);
        if (str == roundtrip) {
            passed++;
        } else {
            std::cout << "  Roundtrip failed for \"" << str << "\" -> \"" << roundtrip << "\"" << std::endl;
        }
    }
    std::cout << "Roundtrip tests: " << passed << "/" << test_strings.size() << " passed" << std::endl;
}

// 测试边界情况
void test_edge_cases(const std::string& vocab_path) {
    std::cout << "\n=== Testing edge cases ===" << std::endl;
    
    trie_tokenizer tokenizer;
    int result = tokenizer.load(vocab_path);
    if (result != RWKV_SUCCESS) {
        std::cout << "✗ Failed to load vocab file for edge case test" << std::endl;
        return;
    }
    
    // 测试空向量解码
    std::vector<int> empty_ids;
    std::string decoded = tokenizer.decode(empty_ids);
    std::cout << "Decode empty vector: \"" << decoded << "\" (length: " << decoded.length() << ")" << std::endl;
    
    // 测试大token ID
    int large_id = 65535;
    decoded = tokenizer.decode(large_id);
    std::cout << "Decode large token ID " << large_id << ": \"" << decoded << "\"" << std::endl;
    
    // 测试包含特殊字符的字符串
    std::string special = "Hello\nWorld\tTest";
    std::vector<int> tokens = tokenizer.encode(special);
    std::cout << "Encode string with special chars: " << tokens.size() << " token(s)" << std::endl;
    
    std::cout << "✓ Edge case tests completed" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "RWKV Tokenizer Test" << std::endl;
    std::cout << "===================" << std::endl;
    
    // 如果提供了命令行参数，使用它作为词汇表文件路径
    std::string vocab_path;
    if (argc > 1) {
        vocab_path = argv[1];
        std::cout << "Using vocab file from command line: " << vocab_path << std::endl;
    }
    
    // 测试加载
    test_load();
    
    // 如果提供了词汇表路径，进行完整测试
    if (!vocab_path.empty()) {
        trie_tokenizer tokenizer;
        if (tokenizer.load(vocab_path) == RWKV_SUCCESS) {
            test_encode(vocab_path);
            test_decode(vocab_path);
            test_edge_cases(vocab_path);
        } else {
            std::cerr << "Failed to load vocab file: " << vocab_path << std::endl;
            return 1;
        }
    } else {
        // 尝试自动查找词汇表文件
        std::vector<std::string> vocab_paths = {
            "rwkv_batch/rwkv_vocab_v20230424.txt",
            "rwkv_cuda/src/utils/b_rwkv_vocab_v20230424.txt",
            "../rwkv_batch/rwkv_vocab_v20230424.txt",
            "../../rwkv_batch/rwkv_vocab_v20230424.txt"
        };
        
        bool found = false;
        for (const auto& path : vocab_paths) {
            trie_tokenizer tokenizer;
            if (tokenizer.load(path) == RWKV_SUCCESS) {
                vocab_path = path;
                found = true;
                std::cout << "\nFound vocab file: " << path << std::endl;
                test_encode(vocab_path);
                test_decode(vocab_path);
                test_edge_cases(vocab_path);
                break;
            }
        }
        
        if (!found) {
            std::cout << "\n⚠ Could not find vocab file automatically." << std::endl;
            std::cout << "  Please provide the vocab file path as a command line argument:" << std::endl;
            std::cout << "  ./test_tokenizer <path_to_vocab_file>" << std::endl;
        }
    }
    
    std::cout << "\n✓ All tests completed!" << std::endl;
    return 0;
}

