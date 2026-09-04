#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "rwkv_server_backend.hpp"
#include "rwkv_tokenizer.hpp"

namespace {

constexpr int kDefaultLength = 16384;
constexpr std::uint32_t kDefaultSeed = 1380668983u;

struct Options {
  std::string fp16_model;
  std::string int8_model;
  std::string vocab;
  std::string output_prefix = "tmp/logit_mse_curve";
  std::string tune_cache;
  std::string text_path;
  int length = kDefaultLength;
  bool length_specified = false;
  int ppl_seq_len = 4096;
  int ppl_stride = 128;
  std::uint32_t seed = kDefaultSeed;
  bool seed_specified = false;
  bool ppl = false;
};

void print_usage() {
  std::cerr << "usage: rwkv_logit_mse_curve FP16_MODEL INT8_MODEL VOCAB "
               "[--length N] [--seed N | --text FILE] [--output-prefix PATH] "
               "[--tune-cache PATH] [--ppl] [--ppl-seq-len N] [--ppl-stride N]\n";
}

Options parse_options(int argc, char** argv) {
  if (argc < 4) {
    print_usage();
    throw std::runtime_error("missing model or vocabulary argument");
  }
  Options options;
  options.fp16_model = argv[1];
  options.int8_model = argv[2];
  options.vocab = argv[3];
  for (int index = 4; index < argc; ++index) {
    const std::string argument = argv[index];
    auto require_value = [&](const char* name) -> const char* {
      if (index + 1 >= argc) {
        print_usage();
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++index];
    };
    if (argument == "--length") {
      options.length = std::stoi(require_value("--length"));
      options.length_specified = true;
    } else if (argument == "--seed") {
      options.seed = static_cast<std::uint32_t>(std::stoul(require_value("--seed")));
      options.seed_specified = true;
    } else if (argument == "--text") {
      options.text_path = require_value("--text");
    } else if (argument == "--output-prefix") {
      options.output_prefix = require_value("--output-prefix");
    } else if (argument == "--tune-cache") {
      options.tune_cache = require_value("--tune-cache");
    } else if (argument == "--ppl") {
      options.ppl = true;
    } else if (argument == "--ppl-seq-len") {
      options.ppl_seq_len = std::stoi(require_value("--ppl-seq-len"));
    } else if (argument == "--ppl-stride") {
      options.ppl_stride = std::stoi(require_value("--ppl-stride"));
    } else {
      print_usage();
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  if (options.length <= 0) throw std::runtime_error("length must be positive");
  if (options.ppl_seq_len <= 0) throw std::runtime_error("--ppl-seq-len must be positive");
  if (options.ppl_stride <= 0) throw std::runtime_error("--ppl-stride must be positive");
  if (!options.text_path.empty() && options.seed_specified) {
    throw std::runtime_error("--text and --seed are mutually exclusive");
  }
  return options;
}

std::vector<std::int64_t> make_token_sequence(int length, int vocab_size, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> distribution(0, vocab_size - 1);
  std::vector<std::int64_t> tokens(static_cast<std::size_t>(length));
  for (std::int64_t& token : tokens) token = distribution(rng);
  return tokens;
}

std::vector<std::int64_t> make_text_token_sequence(
    const std::string& path, rwkv7_server::TrieTokenizer& tokenizer, int length) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot read text file: " + path);
  const std::string text(
      (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  const std::vector<int> encoded = tokenizer.encode(text);
  if (static_cast<int>(encoded.size()) < length) {
    throw std::runtime_error(
        "text file has only " + std::to_string(encoded.size()) +
        " tokens; need at least " + std::to_string(length));
  }
  std::vector<std::int64_t> tokens;
  tokens.reserve(static_cast<std::size_t>(length));
  for (int index = 0; index < length; ++index) {
    tokens.push_back(encoded[static_cast<std::size_t>(index)]);
  }
  return tokens;
}

void write_tokens(const std::string& path, const std::vector<std::int64_t>& tokens) {
  const std::filesystem::path output_path(path);
  if (!output_path.parent_path().empty()) std::filesystem::create_directories(output_path.parent_path());
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot write token sequence: " + path);
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    output << index + 1 << ' ' << tokens[index] << '\n';
  }
}

std::vector<float> run_model(
    const std::string& label,
    const std::string& model_path,
    const std::vector<std::int64_t>& tokens,
    const std::string& tune_cache) {
  std::cout << "mse_curve_start model=" << label << " length=" << tokens.size() << '\n';
  auto model = std::make_shared<rwkv7_server::ModelBackend>(
      model_path, false, true, "no-fc", tune_cache, false);
  const int vocab_size = model->vocab_size();
  for (std::int64_t token : tokens) {
    if (token < 0 || token >= vocab_size) {
      throw std::runtime_error("token sequence exceeds " + label + " vocabulary size");
    }
  }
  const std::size_t row_size = static_cast<std::size_t>(vocab_size);
  std::vector<float> all_logits(tokens.size() * row_size);
  auto state = model->create_state(1);
  rwkv7_server::DeviceLogits logits;

  model->forward_prefill({{tokens.front()}}, state, logits);
  for (std::size_t position = 0; position < tokens.size(); ++position) {
    if (position > 0) model->forward_decode({tokens[position]}, state, logits);
    if (logits.rows != 1 || logits.vocab_size != vocab_size) {
      throw std::runtime_error("unexpected logits shape from " + label);
    }
    float* destination = all_logits.data() + position * row_size;
    const cudaError_t status = cudaMemcpy(
        destination, logits.values.p, row_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
      throw std::runtime_error(std::string("copy logits failed: ") + cudaGetErrorString(status));
    }
    if ((position + 1) % 1024 == 0 || position + 1 == tokens.size()) {
      std::cout << "mse_curve_progress model=" << label << " position=" << position + 1 << '\n';
    }
  }
  model.reset();
  return all_logits;
}

struct CurvePoint {
  double mse = 0.0;
  double max_abs = 0.0;
  bool top1_equal = false;
};

struct PplPoint {
  int chunk = 0;
  std::size_t start_position = 0;
  std::size_t end_position = 0;
  std::size_t token_count = 0;
  double fp16_ppl = 0.0;
  double int8_ppl = 0.0;
};

std::vector<CurvePoint> compare_logits(
    const std::vector<float>& reference,
    const std::vector<float>& candidate,
    std::size_t row_size) {
  if (reference.size() != candidate.size() || row_size == 0 || reference.size() % row_size != 0) {
    throw std::runtime_error("incompatible logits buffers");
  }
  const std::size_t rows = reference.size() / row_size;
  std::vector<CurvePoint> curve(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    const float* ref = reference.data() + row * row_size;
    const float* got = candidate.data() + row * row_size;
    double sum_sq = 0.0;
    double max_abs = 0.0;
    std::size_t ref_best = 0;
    std::size_t got_best = 0;
    for (std::size_t index = 0; index < row_size; ++index) {
      const double difference = static_cast<double>(got[index]) - ref[index];
      sum_sq += difference * difference;
      max_abs = std::max(max_abs, std::abs(difference));
      if (ref[index] > ref[ref_best]) ref_best = index;
      if (got[index] > got[got_best]) got_best = index;
    }
    curve[row] = {sum_sq / static_cast<double>(row_size), max_abs, ref_best == got_best};
  }
  return curve;
}

double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) return 0.0;
  const std::size_t index = static_cast<std::size_t>(fraction * static_cast<double>(values.size() - 1));
  std::nth_element(values.begin(), values.begin() + index, values.end());
  return values[index];
}

void write_curve_csv(const std::string& path, const std::vector<CurvePoint>& curve) {
  const std::filesystem::path output_path(path);
  if (!output_path.parent_path().empty()) std::filesystem::create_directories(output_path.parent_path());
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot write curve CSV: " + path);
  output << "position,mse,rmse,max_abs,top1_equal\n";
  output << std::setprecision(10);
  for (std::size_t index = 0; index < curve.size(); ++index) {
    output << index + 1 << ',' << curve[index].mse << ',' << std::sqrt(curve[index].mse) << ','
           << curve[index].max_abs << ',' << (curve[index].top1_equal ? 1 : 0) << '\n';
  }
}

void print_summary(const std::vector<CurvePoint>& curve) {
  for (std::size_t limit : {std::size_t(1024), std::size_t(4096), std::size_t(16384)}) {
    if (limit > curve.size()) continue;
    std::vector<double> values;
    values.reserve(limit);
    double sum = 0.0;
    double max_mse = 0.0;
    std::size_t top1 = 0;
    for (std::size_t index = 0; index < limit; ++index) {
      sum += curve[index].mse;
      values.push_back(curve[index].mse);
      max_mse = std::max(max_mse, curve[index].mse);
      top1 += curve[index].top1_equal ? 1 : 0;
    }
    std::cout << "mse_curve_summary positions=" << limit
              << " mean_mse=" << (sum / static_cast<double>(limit))
              << " p50_mse=" << percentile(values, 0.50)
              << " p95_mse=" << percentile(values, 0.95)
              << " max_mse=" << max_mse
              << " top1_equal=" << top1 << '/' << limit << '\n';
  }
}

double token_nll(const float* logits, std::size_t vocab_size, std::int64_t target) {
  if (target < 0 || static_cast<std::size_t>(target) >= vocab_size) {
    throw std::runtime_error("PPL target token is outside the model vocabulary");
  }
  float max_logit = logits[0];
  for (std::size_t index = 1; index < vocab_size; ++index) {
    max_logit = std::max(max_logit, logits[index]);
  }
  double exp_sum = 0.0;
  for (std::size_t index = 0; index < vocab_size; ++index) {
    exp_sum += std::exp(static_cast<double>(logits[index]) - static_cast<double>(max_logit));
  }
  const double logsumexp = static_cast<double>(max_logit) + std::log(exp_sum);
  return logsumexp - static_cast<double>(logits[static_cast<std::size_t>(target)]);
}

double perplexity(
    const std::vector<float>& logits,
    std::size_t vocab_size,
    const std::vector<std::int64_t>& tokens,
    std::size_t begin,
    std::size_t end) {
  if (begin >= end || end >= tokens.size() || logits.size() != tokens.size() * vocab_size) {
    throw std::runtime_error("invalid PPL range");
  }
  double nll = 0.0;
  for (std::size_t position = begin; position < end; ++position) {
    nll += token_nll(logits.data() + position * vocab_size, vocab_size, tokens[position + 1]);
  }
  return std::exp(nll / static_cast<double>(end - begin));
}

std::vector<PplPoint> compute_ppl(
    const std::vector<float>& fp16,
    const std::vector<float>& int8,
    std::size_t vocab_size,
    const std::vector<std::int64_t>& tokens,
    int sequence_length,
    int stride) {
  if (sequence_length <= 0 || stride <= 0 ||
      tokens.size() < static_cast<std::size_t>(sequence_length + 1)) {
    throw std::runtime_error("PPL sequence requires one target token beyond the sequence length");
  }
  const std::size_t prediction_count = static_cast<std::size_t>(sequence_length);
  std::vector<PplPoint> points;
  for (std::size_t begin = 0; begin < prediction_count; begin += static_cast<std::size_t>(stride)) {
    const std::size_t end = std::min(prediction_count, begin + static_cast<std::size_t>(stride));
    points.push_back({
        static_cast<int>(points.size()) + 1,
        begin + 1,
        end + 1,
        end - begin,
        perplexity(fp16, vocab_size, tokens, begin, end),
        perplexity(int8, vocab_size, tokens, begin, end),
    });
  }
  return points;
}

void write_ppl_csv(const std::string& path, const std::vector<PplPoint>& points) {
  const std::filesystem::path output_path(path);
  if (!output_path.parent_path().empty()) std::filesystem::create_directories(output_path.parent_path());
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot write PPL CSV: " + path);
  output << "chunk,start_position,end_position,token_count,fp16_ppl,int8_ppl\n";
  output << std::setprecision(10);
  for (const auto& point : points) {
    output << point.chunk << ',' << point.start_position << ',' << point.end_position << ','
           << point.token_count << ',' << point.fp16_ppl << ',' << point.int8_ppl << '\n';
  }
}

void print_ppl_summary(
    const std::vector<float>& fp16,
    const std::vector<float>& int8,
    std::size_t vocab_size,
    const std::vector<std::int64_t>& tokens,
    int sequence_length,
    int stride,
    const std::string& output_prefix) {
  const std::vector<PplPoint> points = compute_ppl(
      fp16, int8, vocab_size, tokens, sequence_length, stride);
  const double fp16_ppl = perplexity(fp16, vocab_size, tokens, 0, static_cast<std::size_t>(sequence_length));
  const double int8_ppl = perplexity(int8, vocab_size, tokens, 0, static_cast<std::size_t>(sequence_length));
  std::cout << std::setprecision(10)
            << "ppl_summary tokens=" << sequence_length
            << " stride=" << stride
            << " fp16_ppl=" << fp16_ppl
            << " int8_ppl=" << int8_ppl
            << " absolute_diff=" << (int8_ppl - fp16_ppl)
            << " relative_diff_percent=" << ((int8_ppl / fp16_ppl - 1.0) * 100.0)
            << '\n';
  for (const auto& point : points) {
    std::cout << "ppl_chunk chunk=" << point.chunk
              << " start_position=" << point.start_position
              << " end_position=" << point.end_position
              << " token_count=" << point.token_count
              << " fp16_ppl=" << point.fp16_ppl
              << " int8_ppl=" << point.int8_ppl << '\n';
  }
  write_ppl_csv(output_prefix + ".ppl.csv", points);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    rwkv7_server::TrieTokenizer tokenizer;
    if (tokenizer.load(options.vocab) != rwkv7_server::kTokenizerSuccess) {
      throw std::runtime_error("failed to load vocabulary");
    }
    const int sequence_length = options.ppl
        ? (options.length_specified ? std::max(options.length, options.ppl_seq_len + 1)
                                    : options.ppl_seq_len + 1)
        : options.length;
    const std::vector<std::int64_t> tokens = options.text_path.empty()
        ? make_token_sequence(sequence_length, 65536, options.seed)
        : make_text_token_sequence(options.text_path, tokenizer, sequence_length);
    write_tokens(options.output_prefix + ".tokens", tokens);
    const std::vector<float> fp16 = run_model("FP16", options.fp16_model, tokens, {});
    const std::vector<float> int8 = run_model("INT8", options.int8_model, tokens, options.tune_cache);
    const std::size_t vocab_size = fp16.size() / tokens.size();
    const std::vector<CurvePoint> curve = compare_logits(fp16, int8, vocab_size);
    write_curve_csv(options.output_prefix + ".csv", curve);
    print_summary(curve);
    if (options.ppl) {
      print_ppl_summary(
          fp16, int8, vocab_size, tokens, options.ppl_seq_len, options.ppl_stride,
          options.output_prefix);
    }
    std::cout << "mse_curve_output csv=" << options.output_prefix << ".csv tokens="
              << options.output_prefix << ".tokens source="
              << (options.text_path.empty() ? "random" : options.text_path) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "logit MSE curve failed: " << error.what() << '\n';
    return 1;
  }
}
