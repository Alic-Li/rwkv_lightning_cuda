#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>
#include "rwkv/io/pth_writer.hpp"
#include "rwkv/server/rwkv_uploaded_state_store.hpp"

#define CHECK(x) do { if (!(x)) throw std::runtime_error("check failed: " #x); } while (0)

int main() {
  auto& store = rwkv7_server::UploadedStateStore::instance();
  const auto fixture = std::filesystem::temp_directory_path() /
      ("rwkv_uploaded_state_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".pth");
  auto bytes = [&](std::vector<llm_infer::WriteTensor> tensors) {
    llm_infer::write_pth(fixture.string(), tensors);
    std::ifstream input(fixture, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input), {});
  };
  auto tensor = [](const std::string& name, bool bf16 = true) {
    llm_infer::WriteTensor result;
    result.name = name;
    result.shape = {2, 4, 4};
    result.bf16 = bf16;
    result.data.resize(32 * (bf16 ? 2 : 4));
    return result;
  };
  try {
    store.shutdown();
    const auto data = bytes({tensor("blocks.0.att.time_state"), tensor("blocks.1.att.time_state", false)});
    const auto before = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    auto a = store.upload("state01.pth", data.data(), data.size());
    auto b = store.upload("C:\\states\\state01.pth", data.data(), data.size());
    const std::regex pattern("state01-[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}");
    CHECK(std::regex_match(a.state_id, pattern));
    CHECK(a.state_id != b.state_id);
    CHECK(a.filename == a.state_id && a.original_filename == "state01.pth");
    CHECK(b.original_filename == "state01.pth");
    CHECK(a.size_bytes == data.size() && a.layers == 2 && a.tensor_count == 2);
    CHECK(a.heads == 2 && a.head_size == 4);
    CHECK(a.created_ms >= before && a.created == a.created_ms / 1000);
    CHECK(std::regex_match(a.uploaded_at, std::regex("[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\\.[0-9]{3}Z")));
    const std::string uuid = "017f22e2-79b0-7cc3-98c4-dc0c0c07398f";
    auto c = store.upload("../../state01.pth", data.data(), data.size(), uuid);
    CHECK(c.state_id == "state01-" + uuid);
    CHECK(store.list().size() == 3);
    auto reject = [&](const std::string& name, const std::string& payload, const std::string& id = "") {
      bool rejected = false;
      try { store.upload(name, payload.data(), payload.size(), id); }
      catch (const std::exception&) { rejected = true; }
      CHECK(rejected);
      CHECK(store.list().size() == 3);
    };
    const auto repeated = store.upload("state01.pth", data.data(), data.size(), uuid);
    CHECK(repeated.state_id == c.state_id && repeated.created_ms == c.created_ms);
    CHECK(store.list().size() == 3);
    auto changed_tensor = tensor("blocks.0.att.time_state"); changed_tensor.data[0] = 1;
    const auto changed = bytes({changed_tensor, tensor("blocks.1.att.time_state", false)});
    reject("state01.pth", changed, uuid); // Same UUID cannot overwrite different bytes.
    reject("state01.bin", data, uuid); // Same stem cannot change the source filename.
    reject("state01.pth", data, "../../escape");
    reject("bad.pth", "not a PTH");
    reject("empty.pth", "");
    reject("bad.pth", bytes({tensor("blocks.x.att.time_state")}));
    reject("gap.pth", bytes({tensor("blocks.1.att.time_state")}));
    reject("gap.pth", bytes({tensor("blocks.0.att.time_state"), tensor("blocks.2.att.time_state")}));
    reject("duplicate.pth", bytes({tensor("blocks.0.att.time_state"), tensor("blocks.00.att.time_state")}));
    auto invalid = tensor("blocks.0.att.time_state");
    invalid.shape = {2, 4, 3};
    invalid.data.resize(48);
    reject("shape.pth", bytes({invalid}));
    invalid = tensor("blocks.0.att.time_state"); invalid.bf16 = false; invalid.fp16 = true;
    reject("dtype.pth", bytes({invalid}));
    auto other = tensor("blocks.1.att.time_state"); other.shape = {1, 4, 4}; other.data.resize(32);
    reject("inconsistent.pth", bytes({tensor("blocks.0.att.time_state"), other}));
    auto handle = store.acquire(a.state_id);
    CHECK(handle && std::filesystem::exists(handle->path));
    const auto stored_path = handle->path;
    CHECK(store.erase(a.state_id));
    CHECK(!store.acquire(a.state_id) && std::filesystem::exists(stored_path));
    handle.reset();
    CHECK(!std::filesystem::exists(stored_path));
    CHECK(store.acquire(c.state_id));
    store.shutdown();
    CHECK(store.list().empty());
    std::filesystem::remove(fixture);
    std::cout << "uploaded state store tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    store.shutdown();
    std::filesystem::remove(fixture);
    std::cerr << error.what() << '\n';
    return 1;
  }
}
