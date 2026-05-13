#include "rwkv_sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace rwkv7_server {

SamplerPenaltyState make_sampler_penalties(int vocab_size) {
  SamplerPenaltyState state;
  state.penalties.resize(static_cast<size_t>(std::max(vocab_size, 0)), 0.0);
  return state;
}

int sample_repetition_temperature_topk_topp(
    const std::vector<float>& logits,
    SamplerPenaltyState& penalties,
    const GenerateOptions& options) {
  if (logits.empty()) {
    throw std::runtime_error("empty logits");
  }

  std::vector<std::pair<int, double>> candidates;
  candidates.reserve(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    double score = static_cast<double>(logits[i]);
    if (i < penalties.penalties.size()) {
      score -= penalties.penalties[i];
    }
    candidates.emplace_back(static_cast<int>(i), score);
  }

  if (options.temperature <= 0.0 || options.top_k == 1) {
    return static_cast<int>(std::max_element(
                                candidates.begin(),
                                candidates.end(),
                                [](const auto& a, const auto& b) { return a.second < b.second; })
                                ->first);
  }

  const int keep_k =
      std::max(1, std::min<int>(std::max(options.top_k, 1), static_cast<int>(candidates.size())));
  std::partial_sort(
      candidates.begin(),
      candidates.begin() + keep_k,
      candidates.end(),
      [](const auto& a, const auto& b) { return a.second > b.second; });
  candidates.resize(static_cast<size_t>(keep_k));

  double max_logit = -std::numeric_limits<double>::infinity();
  for (const auto& item : candidates) {
    max_logit = std::max(max_logit, item.second);
  }

  std::vector<double> probs(candidates.size(), 0.0);
  double sum = 0.0;
  for (size_t i = 0; i < candidates.size(); ++i) {
    probs[i] = std::exp((candidates[i].second - max_logit) / options.temperature);
    sum += probs[i];
  }
  if (sum <= 0.0) {
    return candidates.front().first;
  }
  for (double& p : probs) {
    p /= sum;
  }

  if (options.top_p > 0.0 && options.top_p < 1.0) {
    std::vector<size_t> order(probs.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return probs[a] > probs[b]; });
    double cumulative = 0.0;
    std::vector<double> filtered(probs.size(), 0.0);
    for (size_t idx : order) {
      cumulative += probs[idx];
      filtered[idx] = probs[idx];
      if (cumulative >= options.top_p) {
        break;
      }
    }
    const double filtered_sum = std::accumulate(filtered.begin(), filtered.end(), 0.0);
    if (filtered_sum > 0.0) {
      for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] = filtered[i] / filtered_sum;
      }
    }
  }

  static thread_local std::mt19937 rng(std::random_device{}());
  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  return candidates[static_cast<size_t>(dist(rng))].first;
}

void update_sampler_penalties(
    int token,
    SamplerPenaltyState& penalties,
    const GenerateOptions& options) {
  for (double& value : penalties.penalties) {
    value *= options.alpha_decay;
  }
  if (token >= 0 && static_cast<size_t>(token) < penalties.penalties.size()) {
    penalties.penalties[static_cast<size_t>(token)] += options.alpha_presence + options.alpha_frequency;
  }
}

}  // namespace rwkv7_server
