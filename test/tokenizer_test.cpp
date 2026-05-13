#include <iostream>
#include <vector>

#include "rwkv_tokenizer.hpp"
#include "test_common.hpp"

int main() {
  try {
    rwkv7_server::TrieTokenizer tokenizer;

    TEST_EQ(tokenizer.load("/definitely/missing_vocab.txt"), rwkv7_server::kTokenizerError);
    TEST_EQ(tokenizer.load(rwkv_test::vocab_path().string()), rwkv7_server::kTokenizerSuccess);
    TEST_EQ(tokenizer.load(rwkv_test::vocab_path().string()), rwkv7_server::kTokenizerSuccess);

    const std::vector<std::string> samples{
        "hello world",
        "System: test\n\nUser: hi\n\nAssistant:",
        "中文测试：你好，世界！",
        "line1\nline2\n\nline3"};

    for (const auto& sample : samples) {
      const auto ids = tokenizer.encode(sample);
      TEST_CHECK(!ids.empty());
      TEST_EQ(tokenizer.decode(ids), sample);
      TEST_EQ(tokenizer.decode(ids.front()), tokenizer.decode(std::vector<int>{ids.front()}));
    }

    const auto empty = tokenizer.encode("");
    TEST_CHECK(empty.empty());

    std::cout << "rwkv_tokenizer_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "rwkv_tokenizer_test failed: " << e.what() << "\n";
    return 1;
  }
}
