#include "nccl_dp.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace rwkv_dp {
namespace {
int worker_rank = 0, world = 1, device = 0;
ncclUniqueId unique_id{};
volatile sig_atomic_t interrupted = 0;
void on_signal(int signal) { interrupted = signal; }
void check(ncclResult_t result) {
  if (result != ncclSuccess)
    throw std::runtime_error(std::string("NCCL: ") +
                             ncclGetErrorString(result));
}
std::vector<int> parse_devices(const std::string &text) {
  std::vector<int> devices;
  std::set<int> seen;
  std::istringstream input(text);
  std::string word;
  while (std::getline(input, word, ',')) {
    if (word.empty() ||
        word.find_first_not_of("0123456789") != std::string::npos)
      throw std::invalid_argument(
          "--devices requires comma-separated GPU indices");
    int d = std::stoi(word);
    if (!seen.insert(d).second)
      throw std::invalid_argument("--devices contains a duplicate GPU");
    devices.push_back(d);
  }
  if (devices.empty() || text.back() == ',')
    throw std::invalid_argument(
        "--devices must not be empty or end with a comma");
  return devices;
}
} // namespace
int rank() { return worker_rank; }
int size() { return world; }
int launch(const std::string &list) {
  const auto devices = parse_devices(list);
  world = int(devices.size());
  int channel[2];
  if (pipe(channel))
    throw std::runtime_error("DP rendezvous pipe failed");
  std::vector<pid_t> children;
  const pid_t parent = getpid();
  std::cout.flush();
  std::cerr.flush();
  struct sigaction action{}, old_int{}, old_term{}, old_pipe{};
  action.sa_handler = on_signal;
  sigemptyset(&action.sa_mask);
  sigaction(SIGINT, &action, &old_int);
  sigaction(SIGTERM, &action, &old_term);
  action.sa_handler = SIG_IGN;
  sigaction(SIGPIPE, &action, &old_pipe);
  int result = 0;
  for (int r = 0; r < world; ++r) {
    const pid_t child = fork();
    if (child < 0) {
      result = 1;
      break;
    }
    if (!child) {
      // A killed launcher must not leave training processes/VRAM behind.
      prctl(PR_SET_PDEATHSIG, SIGKILL);
      if (getppid() != parent)
        _exit(1);
      sigaction(SIGINT, &old_int, nullptr);
      sigaction(SIGTERM, &old_term, nullptr);
      sigaction(SIGPIPE, &old_pipe, nullptr);
      close(channel[1]);
      // A single read is atomic for this <= PIPE_BUF rendezvous payload.
      const auto n = read(channel[0], &unique_id, sizeof(unique_id));
      close(channel[0]);
      if (n != sizeof(unique_id))
        _exit(1);
      worker_rank = r;
      device = devices[r];
      return -1;
    }
    children.push_back(child);
  }
  close(channel[0]);
  // No CUDA or NCCL calls occur until AFTER all forks.
  if (!result && ncclGetUniqueId(&unique_id) != ncclSuccess)
    result = 1;
  if (!result)
    for (size_t i = 0; i < children.size(); ++i)
      if (write(channel[1], &unique_id, sizeof(unique_id)) !=
          sizeof(unique_id)) {
        result = 1;
        break;
      }
  close(channel[1]);
  while (!children.empty()) {
    if (interrupted)
      result = 128 + interrupted;
    if (result)
      for (auto pid : children)
        kill(pid, SIGKILL);
    int status = 0;
    // Polling avoids a signal arriving between the interrupted check and a
    // blocking waitpid, which could otherwise delay Ctrl-C until an epoch ends.
    const pid_t pid = waitpid(-1, &status, WNOHANG);
    if (!pid) {
      usleep(10000);
      continue;
    }
    if (pid < 0) {
      if (errno == EINTR)
        continue;
      result = 1;
      break;
    }
    children.erase(std::remove(children.begin(), children.end(), pid),
                   children.end());
    if (!WIFEXITED(status) || WEXITSTATUS(status)) {
      if (!result)
        result =
            WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
      std::cerr << "DP worker " << pid
                << " failed; terminating remaining ranks\n";
    }
  }
  sigaction(SIGINT, &old_int, nullptr);
  sigaction(SIGTERM, &old_term, nullptr);
  sigaction(SIGPIPE, &old_pipe, nullptr);
  return result;
}
Group::Group() {
  rwkv_bf16_training::gpu_check(cudaSetDevice(device), "DP select GPU");
  check(ncclCommInitRank(&comm_, world, unique_id, worker_rank));
  scalar_.resize(1 + size_t(world), "DP collective scalars");
  std::fprintf(stderr, "DP rank=%d/%d device=%d\n", worker_rank, world, device);
}
Group::~Group() {
  if (comm_) {
    if (std::uncaught_exceptions())
      ncclCommAbort(comm_);
    else
      ncclCommDestroy(comm_);
  }
}
void Group::sum(float *values, size_t count, cudaStream_t stream) {
  check(
      ncclAllReduce(values, values, count, ncclFloat, ncclSum, comm_, stream));
}
float Group::sum_loss(float value, cudaStream_t stream) {
  using rwkv_bf16_training::gpu_check;
  gpu_check(cudaMemcpyAsync(scalar_.p, &value, sizeof(value),
                            cudaMemcpyHostToDevice, stream),
            "DP upload loss");
  sum(scalar_.p, 1, stream);
  gpu_check(cudaMemcpyAsync(&value, scalar_.p, sizeof(value),
                            cudaMemcpyDeviceToHost, stream),
            "DP download loss");
  gpu_check(cudaStreamSynchronize(stream), "DP loss reduction");
  return value;
}
float Group::sum_loss_and_times(float local_loss, float local_compute_s,
                                std::vector<float> &rank_compute_s,
                                cudaStream_t stream) {
  using rwkv_bf16_training::gpu_check;
  std::vector<float> buf(1 + size_t(world), 0.0f);
  buf[0] = local_loss;
  buf[1 + size_t(worker_rank)] = local_compute_s;
  gpu_check(cudaMemcpyAsync(scalar_.p, buf.data(), buf.size() * sizeof(float),
                            cudaMemcpyHostToDevice, stream),
            "DP upload loss and compute time");
  sum(scalar_.p, buf.size(), stream);
  gpu_check(cudaMemcpyAsync(buf.data(), scalar_.p, buf.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, stream),
            "DP download loss and compute times");
  gpu_check(cudaStreamSynchronize(stream), "DP loss/time reduction");
  rank_compute_s.assign(buf.begin() + 1, buf.end());
  return buf[0];
}
void Group::barrier(cudaStream_t stream) { (void)sum_loss(0, stream); }
std::vector<int> assign(const std::vector<size_t> &lengths, int ranks,
                        int micro_bsz, const std::vector<double> *inv_speeds,
                        std::vector<size_t> *out_raw_loads) {
  if (ranks < 1 || micro_bsz < 1)
    throw std::invalid_argument("invalid DP world size/micro batch");
  std::vector<double> scale(ranks, 1.0);
  const bool weighted =
      inv_speeds && int(inv_speeds->size()) == ranks;
  if (weighted) {
    for (int r = 0; r < ranks; ++r)
      if (std::isfinite((*inv_speeds)[r]) && (*inv_speeds)[r] > 0.0)
        scale[r] = std::clamp((*inv_speeds)[r], 0.5, 2.0);
  }
  std::vector<size_t> order(lengths.size()), load(ranks, 0), count(ranks, 0);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return lengths[a] > lengths[b]; });
  std::vector<int> owners(lengths.size());
  for (auto i : order) {
    // Cost = actual processed tokens + small initial sweep setup on first
    // sample; spilling past 1.5x micro_bsz (virtual-stream capacity) pays a
    // full extra weight sweep penalty.
    const size_t single_pass_cap =
        micro_bsz > 2 ? size_t(micro_bsz + micro_bsz / 2) : size_t(micro_bsz);
    auto next_raw = [&](int r) -> size_t {
      size_t penalty = 0;
      if (count[r] == 0)
        penalty = lengths[i] / 5;
      else if (count[r] % single_pass_cap == 0)
        penalty = lengths[i] * 5;
      return load[r] + lengths[i] + penalty;
    };
    int best = 0;
    if (!weighted) {
      for (int r = 1; r < ranks; ++r)
        if (next_raw(r) < next_raw(best))
          best = r;
    } else {
      double best_time = double(next_raw(0)) * scale[0];
      for (int r = 1; r < ranks; ++r) {
        const double t = double(next_raw(r)) * scale[r];
        if (t < best_time) {
          best = r;
          best_time = t;
        }
      }
    }
    owners[i] = best;
    load[best] = next_raw(best);
    ++count[best];
  }
  if (out_raw_loads)
    *out_raw_loads = std::move(load);
  return owners;
}
void update_speed_model(const std::vector<size_t> &raw_loads,
                        const std::vector<float> &rank_compute_s,
                        std::vector<double> &inv_speeds, bool &ready) {
  const size_t ranks = raw_loads.size();
  if (ranks < 2 || rank_compute_s.size() != ranks)
    return;
  std::vector<double> spu(ranks, 0.0);
  double sum_spu = 0.0;
  for (size_t r = 0; r < ranks; ++r) {
    if (raw_loads[r] == 0 || !std::isfinite(rank_compute_s[r]) ||
        rank_compute_s[r] <= 1.0e-3f)
      return;
    spu[r] = double(rank_compute_s[r]) / double(raw_loads[r]);
    sum_spu += spu[r];
  }
  const double mean_spu = sum_spu / double(ranks);
  if (!(mean_spu > 0.0) || !std::isfinite(mean_spu))
    return;
  if (inv_speeds.size() != ranks)
    inv_speeds.assign(ranks, 1.0);
  for (size_t r = 0; r < ranks; ++r) {
    const double rel = std::clamp(spu[r] / mean_spu, 0.65, 1.5);
    inv_speeds[r] = ready ? (0.82 * inv_speeds[r] + 0.18 * rel) : rel;
  }
  ready = true;
}
} // namespace rwkv_dp
