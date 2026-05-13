#include "pth_tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace llm_infer {
namespace {

struct GlobalRef {
  std::string module;
  std::string name;
};

struct StorageRef {
  std::string key;
  std::uint64_t size = 0;
};

struct Value {
  enum class Kind {
    kNone,
    kMark,
    kInt,
    kBool,
    kString,
    kGlobal,
    kTuple,
    kDict,
    kStorage,
    kTensor,
    kOrderedDict,
  };

  Kind kind = Kind::kNone;
  std::int64_t i = 0;
  bool b = false;
  std::string s;
  GlobalRef global;
  StorageRef storage;
  TensorRecord tensor;
  std::vector<Value> items;
};

std::uint16_t read_u16(const std::vector<std::uint8_t>& bytes, std::size_t off) {
  return static_cast<std::uint16_t>(bytes[off]) |
         (static_cast<std::uint16_t>(bytes[off + 1]) << 8);
}

std::uint32_t read_u32(const std::vector<std::uint8_t>& bytes, std::size_t off) {
  return static_cast<std::uint32_t>(bytes[off]) |
         (static_cast<std::uint32_t>(bytes[off + 1]) << 8) |
         (static_cast<std::uint32_t>(bytes[off + 2]) << 16) |
         (static_cast<std::uint32_t>(bytes[off + 3]) << 24);
}

std::int32_t read_i32(const std::vector<std::uint8_t>& bytes, std::size_t off) {
  std::uint32_t u = read_u32(bytes, off);
  std::int32_t out = 0;
  std::memcpy(&out, &u, sizeof(out));
  return out;
}

std::string number(std::uint64_t value) {
  std::ostringstream oss;
  oss << value;
  return oss.str();
}

Value value_of_kind(Value::Kind kind) {
  Value value;
  value.kind = kind;
  return value;
}

bool is_bfloat16_storage(const GlobalRef& global) {
  return global.module == "torch" && global.name == "BFloat16Storage";
}

std::vector<std::int64_t> tuple_to_ints(const Value& value) {
  std::vector<std::int64_t> out;
  if (value.kind != Value::Kind::kTuple) {
    return out;
  }
  out.reserve(value.items.size());
  for (const Value& item : value.items) {
    if (item.kind != Value::Kind::kInt) {
      return {};
    }
    out.push_back(item.i);
  }
  return out;
}

std::uint64_t numel(const std::vector<std::int64_t>& shape) {
  std::uint64_t n = 1;
  for (std::int64_t dim : shape) {
    if (dim < 0) {
      return 0;
    }
    n *= static_cast<std::uint64_t>(dim);
  }
  return n;
}

bool is_contiguous(const std::vector<std::int64_t>& shape, const std::vector<std::int64_t>& stride) {
  if (shape.size() != stride.size()) {
    return false;
  }
  std::int64_t expected = 1;
  for (std::size_t idx = shape.size(); idx > 0; --idx) {
    const std::size_t i = idx - 1;
    if (shape[i] == 0) {
      return true;
    }
    if (shape[i] != 1 && stride[i] != expected) {
      return false;
    }
    expected *= shape[i];
  }
  return true;
}

template <typename T>
T load_scalar(const std::uint8_t* ptr) {
  T value;
  std::memcpy(&value, ptr, sizeof(T));
  return value;
}

double read_bf16_element_as_double(const std::uint8_t* base, std::uint64_t element_index) {
  const std::uint8_t* ptr = base + element_index * sizeof(std::uint16_t);
  return bf16_bits_to_float(load_scalar<std::uint16_t>(ptr));
}

class PickleTensorParser {
public:
  explicit PickleTensorParser(std::vector<std::uint8_t> bytes) : bytes_(std::move(bytes)) {}

  Result<std::vector<TensorRecord>> parse() {
    while (pos_ < bytes_.size()) {
      const std::uint8_t op = bytes_[pos_++];
      Status status = dispatch(op);
      if (!status.ok_status()) {
        return status;
      }
      if (stopped_) {
        return tensors_;
      }
    }
    return Status::error("pickle STOP not found");
  }

private:
  Status dispatch(std::uint8_t op) {
    switch (op) {
      case 0x80:
        return skip(1, "PROTO argument");
      case '}':
        return push(value_of_kind(Value::Kind::kDict));
      case '(':
        return push(value_of_kind(Value::Kind::kMark));
      case 'q':
        return memo_put(read_byte(), false);
      case 'r':
        return memo_put(read_u32_at_cursor(), true);
      case 'h':
        return memo_get(read_byte());
      case 'j':
        return memo_get(read_u32_at_cursor());
      case 'X':
        return push_string(read_sized_string(read_u32_at_cursor()));
      case 'c':
        return push_global();
      case 'K':
        return push_int(read_byte());
      case 'M':
        return push_int(read_u16_at_cursor());
      case 'J':
        return push_int(read_i32_at_cursor());
      case 0x8a:
        return push_int(read_long_at_cursor(read_byte(), "LONG1"));
      case 0x8b:
        return push_int(read_long_at_cursor(read_u32_at_cursor(), "LONG4"));
      case 't':
        return make_tuple_from_mark();
      case ')':
        return push(value_of_kind(Value::Kind::kTuple));
      case 0x85:
        return make_tuple_n(1);
      case 0x86:
        return make_tuple_n(2);
      case 0x87:
        return make_tuple_n(3);
      case 0x88:
        return push_bool(true);
      case 0x89:
        return push_bool(false);
      case 'Q':
        return make_persistent_id();
      case 'R':
        return reduce();
      case 's':
        return setitem();
      case 'u':
        return setitems();
      case 'b':
        return build();
      case '.':
        stopped_ = true;
        return Status::ok();
      default:
        return Status::error("unsupported pickle opcode 0x" + hex(op) + " at byte " + number(pos_ - 1));
    }
  }

  Status push(Value value) {
    stack_.push_back(std::move(value));
    return Status::ok();
  }

  Status push_int(std::int64_t value) {
    Value v;
    v.kind = Value::Kind::kInt;
    v.i = value;
    return push(std::move(v));
  }

  Status push_bool(bool value) {
    Value v;
    v.kind = Value::Kind::kBool;
    v.b = value;
    return push(std::move(v));
  }

  Status push_string(std::string value) {
    Value v;
    v.kind = Value::Kind::kString;
    v.s = std::move(value);
    return push(std::move(v));
  }

  Status push_global() {
    const std::string module = read_line();
    const std::string name = read_line();
    if (module.empty() || name.empty()) {
      return Status::error("malformed GLOBAL opcode");
    }
    Value v;
    v.kind = Value::Kind::kGlobal;
    v.global = GlobalRef{module, name};
    return push(std::move(v));
  }

  Status make_tuple_n(std::size_t n) {
    if (stack_.size() < n) {
      return Status::error("tuple opcode underflow");
    }
    Value v;
    v.kind = Value::Kind::kTuple;
    v.items.assign(stack_.end() - static_cast<std::ptrdiff_t>(n), stack_.end());
    stack_.resize(stack_.size() - n);
    return push(std::move(v));
  }

  Status make_tuple_from_mark() {
    const std::size_t mark = find_mark();
    if (mark == npos) {
      return Status::error("TUPLE without MARK");
    }
    Value v;
    v.kind = Value::Kind::kTuple;
    v.items.assign(stack_.begin() + static_cast<std::ptrdiff_t>(mark + 1), stack_.end());
    stack_.resize(mark);
    return push(std::move(v));
  }

  Status make_persistent_id() {
    if (stack_.empty() || stack_.back().kind != Value::Kind::kTuple) {
      return Status::error("BINPERSID expects a tuple persistent id");
    }
    Value pid = std::move(stack_.back());
    stack_.pop_back();
    if (pid.items.size() != 5 ||
        pid.items[0].kind != Value::Kind::kString ||
        pid.items[1].kind != Value::Kind::kGlobal ||
        pid.items[2].kind != Value::Kind::kString ||
        pid.items[4].kind != Value::Kind::kInt ||
        pid.items[0].s != "storage") {
      return Status::error("unsupported persistent id shape");
    }
    Value out;
    out.kind = Value::Kind::kStorage;
    out.storage.key = pid.items[2].s;
    out.storage.size = static_cast<std::uint64_t>(pid.items[4].i);
    if (!is_bfloat16_storage(pid.items[1].global)) {
      return Status::error(
          "unsupported storage dtype; this reader only supports torch.BFloat16Storage, got: " +
          pid.items[1].global.module + "." + pid.items[1].global.name);
    }
    return push(std::move(out));
  }

  Status reduce() {
    if (stack_.size() < 2) {
      return Status::error("REDUCE stack underflow");
    }
    Value args = std::move(stack_.back());
    stack_.pop_back();
    Value callable = std::move(stack_.back());
    stack_.pop_back();
    if (callable.kind != Value::Kind::kGlobal || args.kind != Value::Kind::kTuple) {
      return Status::error("unsupported REDUCE operands");
    }

    if (callable.global.module == "collections" && callable.global.name == "OrderedDict") {
      Value out;
      out.kind = Value::Kind::kOrderedDict;
      return push(std::move(out));
    }

    if (callable.global.module == "torch._utils" && callable.global.name == "_rebuild_tensor_v2") {
      if (args.items.size() < 4 ||
          args.items[0].kind != Value::Kind::kStorage ||
          args.items[1].kind != Value::Kind::kInt ||
          args.items[2].kind != Value::Kind::kTuple ||
          args.items[3].kind != Value::Kind::kTuple) {
        return Status::error("unsupported _rebuild_tensor_v2 arguments");
      }
      Value out;
      out.kind = Value::Kind::kTensor;
      out.tensor.dtype = TensorDType::kBFloat16;
      out.tensor.storage_key = args.items[0].storage.key;
      out.tensor.storage_size = args.items[0].storage.size;
      out.tensor.storage_offset = static_cast<std::uint64_t>(args.items[1].i);
      out.tensor.shape = tuple_to_ints(args.items[2]);
      out.tensor.stride = tuple_to_ints(args.items[3]);
      if (out.tensor.shape.size() != out.tensor.stride.size()) {
        return Status::error("tensor shape / stride rank mismatch");
      }
      return push(std::move(out));
    }

    return Status::error("unsupported REDUCE callable: " + callable.global.module + "." + callable.global.name);
  }

  Status setitems() {
    const std::size_t mark = find_mark();
    if (mark == npos || mark == 0 ||
        (stack_[mark - 1].kind != Value::Kind::kDict &&
         stack_[mark - 1].kind != Value::Kind::kOrderedDict)) {
      return Status::error("SETITEMS without dict / OrderedDict and MARK");
    }
    const std::size_t item_count = stack_.size() - mark - 1;
    if ((item_count % 2) != 0) {
      return Status::error("SETITEMS requires key/value pairs");
    }
    bool tensor_items = item_count > 0;
    for (std::size_t i = mark + 1; i < stack_.size(); i += 2) {
      if (stack_[i].kind != Value::Kind::kString || stack_[i + 1].kind != Value::Kind::kTensor) {
        tensor_items = false;
        break;
      }
    }
    if (tensor_items) {
      for (std::size_t i = mark + 1; i < stack_.size(); i += 2) {
        TensorRecord record = std::move(stack_[i + 1].tensor);
        record.name = stack_[i].s;
        tensors_.push_back(std::move(record));
      }
    }
    Value container = stack_[mark - 1];
    stack_.resize(mark - 1);
    return push(std::move(container));
  }

  Status setitem() {
    if (stack_.size() < 3) {
      return Status::error("SETITEM stack underflow");
    }
    Value value = std::move(stack_.back());
    stack_.pop_back();
    Value key = std::move(stack_.back());
    stack_.pop_back();
    (void)value;
    (void)key;
    if (stack_.back().kind != Value::Kind::kDict &&
        stack_.back().kind != Value::Kind::kOrderedDict) {
      return Status::error("SETITEM target must be dict / OrderedDict");
    }
    return Status::ok();
  }

  Status build() {
    if (stack_.size() < 2) {
      return Status::error("BUILD stack underflow");
    }
    Value state = std::move(stack_.back());
    stack_.pop_back();
    (void)state;
    return Status::ok();
  }

  Status memo_put(std::uint32_t index, bool) {
    if (stack_.empty()) {
      return Status::error("BINPUT stack underflow");
    }
    memo_[index] = stack_.back();
    return Status::ok();
  }

  Status memo_get(std::uint32_t index) {
    auto it = memo_.find(index);
    if (it == memo_.end()) {
      return Status::error("BINGET missing memo index " + number(index));
    }
    return push(it->second);
  }

  std::size_t find_mark() const {
    for (std::size_t idx = stack_.size(); idx > 0; --idx) {
      if (stack_[idx - 1].kind == Value::Kind::kMark) {
        return idx - 1;
      }
    }
    return npos;
  }

  Status skip(std::size_t n, const char* what) {
    if (pos_ + n > bytes_.size()) {
      return Status::error(std::string("truncated ") + what);
    }
    pos_ += n;
    return Status::ok();
  }

  std::uint8_t read_byte() {
    if (pos_ >= bytes_.size()) {
      throw std::runtime_error("truncated pickle byte");
    }
    return bytes_[pos_++];
  }

  std::uint16_t read_u16_at_cursor() {
    if (pos_ + 2 > bytes_.size()) {
      throw std::runtime_error("truncated pickle uint16");
    }
    const std::uint16_t out = read_u16(bytes_, pos_);
    pos_ += 2;
    return out;
  }

  std::uint32_t read_u32_at_cursor() {
    if (pos_ + 4 > bytes_.size()) {
      throw std::runtime_error("truncated pickle uint32");
    }
    const std::uint32_t out = read_u32(bytes_, pos_);
    pos_ += 4;
    return out;
  }

  std::int32_t read_i32_at_cursor() {
    if (pos_ + 4 > bytes_.size()) {
      throw std::runtime_error("truncated pickle int32");
    }
    const std::int32_t out = read_i32(bytes_, pos_);
    pos_ += 4;
    return out;
  }

  std::int64_t read_long_at_cursor(std::uint32_t n, const char* what) {
    if (pos_ + n > bytes_.size()) {
      throw std::runtime_error(std::string("truncated pickle ") + what);
    }
    if (n == 0) {
      return 0;
    }
    if (n > 8) {
      throw std::runtime_error(std::string("pickle ") + what + " does not fit int64");
    }
    std::uint64_t u = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
      u |= static_cast<std::uint64_t>(bytes_[pos_ + i]) << (8 * i);
    }
    const bool negative = (bytes_[pos_ + n - 1] & 0x80u) != 0;
    pos_ += n;
    if (negative && n < 8) {
      u |= (~0ull) << (8 * n);
    }
    std::int64_t out = 0;
    std::memcpy(&out, &u, sizeof(out));
    return out;
  }

  std::string read_sized_string(std::uint32_t n) {
    if (pos_ + n > bytes_.size()) {
      throw std::runtime_error("truncated pickle string");
    }
    std::string out(reinterpret_cast<const char*>(bytes_.data() + pos_), n);
    pos_ += n;
    return out;
  }

  std::string read_line() {
    const std::size_t start = pos_;
    while (pos_ < bytes_.size() && bytes_[pos_] != '\n') {
      ++pos_;
    }
    if (pos_ >= bytes_.size()) {
      return {};
    }
    std::string out(reinterpret_cast<const char*>(bytes_.data() + start), pos_ - start);
    ++pos_;
    return out;
  }

  static std::string hex(std::uint8_t value) {
    const char* digits = "0123456789abcdef";
    std::string out;
    out.push_back(digits[(value >> 4) & 0xfu]);
    out.push_back(digits[value & 0xfu]);
    return out;
  }

  static constexpr std::size_t npos = static_cast<std::size_t>(-1);

  std::vector<std::uint8_t> bytes_;
  std::size_t pos_ = 0;
  bool stopped_ = false;
  std::vector<Value> stack_;
  std::unordered_map<std::uint32_t, Value> memo_;
  std::vector<TensorRecord> tensors_;
};

std::string archive_prefix(const PthArchive& archive) {
  for (const PthEntry& entry : archive.entries()) {
    const std::string suffix = "/data.pkl";
    if (entry.name.size() >= suffix.size() &&
        entry.name.compare(entry.name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      return entry.name.substr(0, entry.name.size() - suffix.size());
    }
  }
  return {};
}

void update_minmax(TensorStats* stats, double value) {
  if (std::isnan(value)) {
    return;
  }
  if (!stats->has_value) {
    stats->min = value;
    stats->max = value;
    stats->has_value = true;
  } else {
    stats->min = std::min(stats->min, value);
    stats->max = std::max(stats->max, value);
  }
}

void scan_strided(
    const std::uint8_t* base,
    const std::vector<std::int64_t>& shape,
    const std::vector<std::int64_t>& stride,
    std::size_t dim,
    std::uint64_t offset,
    TensorStats* stats) {
  if (dim == shape.size()) {
    update_minmax(stats, read_bf16_element_as_double(base, offset));
    return;
  }
  for (std::int64_t i = 0; i < shape[dim]; ++i) {
    scan_strided(base, shape, stride, dim + 1, offset + static_cast<std::uint64_t>(i * stride[dim]), stats);
  }
}

void load_strided_float(
    const std::uint8_t* base,
    const std::vector<std::int64_t>& shape,
    const std::vector<std::int64_t>& stride,
    std::size_t dim,
    std::uint64_t offset,
    std::vector<float>* out) {
  if (dim == shape.size()) {
    out->push_back(static_cast<float>(read_bf16_element_as_double(base, offset)));
    return;
  }
  for (std::int64_t i = 0; i < shape[dim]; ++i) {
    load_strided_float(base, shape, stride, dim + 1, offset + static_cast<std::uint64_t>(i * stride[dim]), out);
  }
}

void load_strided_bf16(
    const std::uint8_t* base,
    const std::vector<std::int64_t>& shape,
    const std::vector<std::int64_t>& stride,
    std::size_t dim,
    std::uint64_t offset,
    std::vector<std::uint16_t>* out) {
  if (dim == shape.size()) {
    const std::uint8_t* ptr = base + offset * sizeof(std::uint16_t);
    out->push_back(load_scalar<std::uint16_t>(ptr));
    return;
  }
  for (std::int64_t i = 0; i < shape[dim]; ++i) {
    load_strided_bf16(base, shape, stride, dim + 1, offset + static_cast<std::uint64_t>(i * stride[dim]), out);
  }
}

}  // namespace

const char* dtype_name(TensorDType dtype) {
  switch (dtype) {
    case TensorDType::kBFloat16:
      return "bfloat16";
  }
  return "bfloat16";
}

std::uint64_t dtype_size_bytes(TensorDType dtype) {
  switch (dtype) {
    case TensorDType::kBFloat16:
      return 2;
  }
  return 2;
}

std::string shape_string(const std::vector<std::int64_t>& dims) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < dims.size(); ++i) {
    if (i != 0) {
      oss << ",";
    }
    oss << dims[i];
  }
  oss << "]";
  return oss.str();
}

float bf16_bits_to_float(std::uint16_t bits) {
  std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
  float out = 0.0f;
  std::memcpy(&out, &wide, sizeof(out));
  return out;
}

float f16_bits_to_float(std::uint16_t bits) {
  const std::uint32_t sign = (static_cast<std::uint32_t>(bits & 0x8000u)) << 16;
  const std::uint32_t exp = (bits >> 10) & 0x1fu;
  const std::uint32_t mant = bits & 0x03ffu;
  std::uint32_t out_bits = 0;
  if (exp == 0) {
    if (mant == 0) {
      out_bits = sign;
    } else {
      std::uint32_t m = mant;
      std::uint32_t e = 113;
      while ((m & 0x0400u) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x03ffu;
      out_bits = sign | (e << 23) | (m << 13);
    }
  } else if (exp == 0x1fu) {
    out_bits = sign | 0x7f800000u | (mant << 13);
  } else {
    out_bits = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  float out = 0.0f;
  std::memcpy(&out, &out_bits, sizeof(out));
  return out;
}

std::uint16_t float_to_bf16_bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<std::uint16_t>(bits >> 16);
}

std::uint16_t float_to_f16_bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  std::int32_t exp = static_cast<std::int32_t>((bits >> 23) & 0xffu) - 127;
  std::uint32_t mant = bits & 0x7fffffu;
  if (exp == 128) {
    return static_cast<std::uint16_t>(sign | (mant == 0 ? 0x7c00u : 0x7e00u));
  }
  if (exp > 15) {
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  }
  if (exp >= -14) {
    std::uint32_t half_exp = static_cast<std::uint32_t>(exp + 15);
    std::uint32_t half_mant = mant + 0x1000u;
    if (half_mant & 0x800000u) {
      half_mant = 0;
      ++half_exp;
      if (half_exp >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
      }
    }
    return static_cast<std::uint16_t>(sign | (half_exp << 10) | (half_mant >> 13));
  }
  if (exp < -24) {
    return static_cast<std::uint16_t>(sign);
  }
  mant |= 0x800000u;
  const int shift = -exp - 14;
  std::uint32_t half_mant = mant >> (shift + 13);
  const std::uint32_t round_bit = (mant >> (shift + 12)) & 1u;
  half_mant += round_bit;
  return static_cast<std::uint16_t>(sign | half_mant);
}

Result<std::vector<TensorRecord>> parse_pth_tensor_records(const PthArchive& archive) {
  const std::string prefix = archive_prefix(archive);
  if (prefix.empty()) {
    return Status::error("data.pkl entry not found");
  }
  const PthEntry* data = archive.find_entry(prefix + "/data.pkl");
  if (data == nullptr) {
    return Status::error("data.pkl entry not found");
  }
  auto bytes = archive.read_stored_entry(*data);
  if (!bytes.ok()) {
    return bytes.status();
  }
  try {
    PickleTensorParser parser(std::move(bytes.value()));
    return parser.parse();
  } catch (const std::exception& ex) {
    return Status::error(ex.what());
  }
}

Result<TensorStats> compute_tensor_stats(const PthArchive& archive, const TensorRecord& record) {
  const std::string prefix = archive_prefix(archive);
  if (prefix.empty()) {
    return Status::error("archive prefix not found");
  }
  const PthEntry* entry = archive.find_entry(prefix + "/data/" + record.storage_key);
  if (entry == nullptr) {
    return Status::error("storage entry not found for tensor: " + record.name);
  }
  auto storage = archive.read_stored_entry(*entry);
  if (!storage.ok()) {
    return storage.status();
  }
  const std::uint64_t dtype_bytes = dtype_size_bytes(record.dtype);
  if (record.storage_size * dtype_bytes != storage.value().size()) {
    return Status::error("storage byte size mismatch for tensor: " + record.name);
  }
  const std::uint64_t n = numel(record.shape);
  TensorStats stats;
  if (n == 0) {
    return stats;
  }
  if (is_contiguous(record.shape, record.stride)) {
    if (record.storage_offset + n > record.storage_size) {
      return Status::error("tensor data range exceeds storage: " + record.name);
    }
    for (std::uint64_t i = 0; i < n; ++i) {
      update_minmax(&stats, read_bf16_element_as_double(storage.value().data(), record.storage_offset + i));
    }
  } else {
    scan_strided(storage.value().data(), record.shape, record.stride, 0, record.storage_offset, &stats);
  }
  return stats;
}

Result<TensorData> load_bf16_tensor_select(
    const PthArchive& archive,
    const TensorRecord& record,
    bool need_bf16,
    bool need_f16,
    bool need_float) {
  const std::string prefix = archive_prefix(archive);
  if (prefix.empty()) {
    return Status::error("archive prefix not found");
  }
  const PthEntry* entry = archive.find_entry(prefix + "/data/" + record.storage_key);
  if (entry == nullptr) {
    return Status::error("storage entry not found for tensor: " + record.name);
  }
  auto storage = archive.read_stored_entry(*entry);
  if (!storage.ok()) {
    return storage.status();
  }
  const std::uint64_t dtype_bytes = dtype_size_bytes(record.dtype);
  if (record.storage_size * dtype_bytes != storage.value().size()) {
    return Status::error("storage byte size mismatch for tensor: " + record.name);
  }
  const std::uint64_t n = numel(record.shape);
  TensorData out;
  out.name = record.name;
  out.shape = record.shape;
  if (need_bf16) out.values_bf16.reserve(static_cast<std::size_t>(n));
  if (need_f16) out.values_f16.reserve(static_cast<std::size_t>(n));
  if (need_float) out.values.reserve(static_cast<std::size_t>(n));
  if (is_contiguous(record.shape, record.stride)) {
    if (record.storage_offset + n > record.storage_size) {
      return Status::error("tensor data range exceeds storage: " + record.name);
    }
    for (std::uint64_t i = 0; i < n; ++i) {
      const std::uint8_t* ptr = storage.value().data() + (record.storage_offset + i) * sizeof(std::uint16_t);
      const std::uint16_t bits = load_scalar<std::uint16_t>(ptr);
      if (need_bf16) out.values_bf16.push_back(bits);
      if (need_f16 || need_float) {
        const float value = bf16_bits_to_float(bits);
        if (need_f16) out.values_f16.push_back(float_to_f16_bits(value));
        if (need_float) out.values.push_back(value);
      }
    }
  } else {
    if (need_bf16) {
      load_strided_bf16(storage.value().data(), record.shape, record.stride, 0, record.storage_offset, &out.values_bf16);
    }
    if (need_float) {
      load_strided_float(storage.value().data(), record.shape, record.stride, 0, record.storage_offset, &out.values);
    }
    if (need_f16) {
      std::vector<float> tmp;
      const std::vector<float>* src = &out.values;
      if (!need_float) {
        tmp.reserve(static_cast<std::size_t>(n));
        load_strided_float(storage.value().data(), record.shape, record.stride, 0, record.storage_offset, &tmp);
        src = &tmp;
      }
      out.values_f16.reserve(src->size());
      for (float value : *src) {
        out.values_f16.push_back(float_to_f16_bits(value));
      }
    }
  }
  return out;
}

Result<TensorData> load_bf16_tensor_as_float(const PthArchive& archive, const TensorRecord& record) {
  return load_bf16_tensor_select(archive, record, true, true, true);
}

}  // namespace llm_infer
