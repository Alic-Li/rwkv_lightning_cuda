#include "nccl_dp.hpp"
#include <algorithm>
#include <arpa/inet.h>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <numeric>
#include <sched.h>
#include <set>
#include <sstream>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace rwkv_dp {
namespace {
int worker_rank = 0, world = 1, device = 0;
ncclUniqueId unique_id{};
volatile sig_atomic_t interrupted = 0;
void on_signal(int signal) { interrupted = signal; }

struct InitPayload {
  ncclUniqueId id{};
  int32_t total_world = 1;
  int32_t rank_offset = 0;
};

struct PeerHello {
  int32_t node_rank = 0;
  int32_t local_world = 0;
};

bool write_exact(int fd, const void *buf, size_t len) {
  const auto *p = static_cast<const uint8_t *>(buf);
  size_t done = 0;
  while (done < len) {
    const ssize_t n = ::write(fd, p + done, len - done);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      return false;
    }
    if (n == 0)
      return false;
    done += size_t(n);
  }
  return true;
}

bool read_exact(int fd, void *buf, size_t len) {
  auto *p = static_cast<uint8_t *>(buf);
  size_t done = 0;
  while (done < len) {
    const ssize_t n = ::read(fd, p + done, len - done);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      return false;
    }
    if (n == 0)
      return false;
    done += size_t(n);
  }
  return true;
}

void set_nonblocking(int fd) {
  const int flags = fcntl(fd, F_GETFL, 0);
  if (flags >= 0)
    (void)fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

void parse_master_addr(const std::string &spec, std::string &host, int &port) {
  const auto colon = spec.rfind(':');
  if (colon == std::string::npos || colon == 0 || colon + 1 >= spec.size())
    throw std::invalid_argument(
        "RWKV_DP_MASTER_ADDR must be formatted as host:port");
  host = spec.substr(0, colon);
  port = std::stoi(spec.substr(colon + 1));
  if (port <= 0 || port > 65535)
    throw std::invalid_argument("invalid RWKV_DP_MASTER_ADDR port");
}

// Pin each DP worker to the NUMA-local CPU cores of its PCIe root complex
// BEFORE model weights or cudaHostAlloc staging buffers are first-touched.
void bind_numa_cpu_affinity(int cuda_dev) {
  char bdf[32] = {};
  if (cudaDeviceGetPCIBusId(bdf, sizeof(bdf), cuda_dev) != cudaSuccess)
    return;
  for (char &ch : bdf)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  std::ifstream f(std::string("/sys/bus/pci/devices/") + bdf + "/local_cpulist");
  std::string spec;
  if (!f || !std::getline(f, spec) || spec.empty())
    return;
  cpu_set_t mask;
  CPU_ZERO(&mask);
  std::istringstream ss(spec);
  std::string part;
  while (std::getline(ss, part, ',')) {
    if (part.empty())
      continue;
    const auto dash = part.find('-');
    int lo = std::atoi(part.c_str());
    int hi = (dash == std::string::npos)
                 ? lo
                 : std::atoi(part.c_str() + dash + 1);
    for (int c = std::max(0, lo); c <= hi && c < CPU_SETSIZE; ++c)
      CPU_SET(c, &mask);
  }
  if (CPU_COUNT(&mask) >= 2)
    (void)sched_setaffinity(0, sizeof(mask), &mask);
}
void check(ncclResult_t result) {
  if (result != ncclSuccess)
    throw std::runtime_error(std::string("GPU collective: ") +
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
  const int local_world = int(devices.size());
  int num_nodes = 1;
  int node_rank = 0;
  std::string master_host;
  int master_port = 0;
  if (const char *nn = std::getenv("RWKV_DP_NUM_NODES")) {
    if (*nn)
      num_nodes = std::stoi(nn);
  }
  if (num_nodes < 1 || num_nodes > 64)
    throw std::invalid_argument("invalid RWKV_DP_NUM_NODES");
  if (num_nodes > 1) {
    const char *nr = std::getenv("RWKV_DP_NODE_RANK");
    const char *ma = std::getenv("RWKV_DP_MASTER_ADDR");
    if (!nr || !*nr || !ma || !*ma)
      throw std::invalid_argument(
          "multi-node DP requires RWKV_DP_NODE_RANK and RWKV_DP_MASTER_ADDR");
    node_rank = std::stoi(nr);
    if (node_rank < 0 || node_rank >= num_nodes)
      throw std::invalid_argument("RWKV_DP_NODE_RANK out of range");
    parse_master_addr(ma, master_host, master_port);
  }

  world = local_world;
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
  for (int r = 0; r < local_world; ++r) {
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
      InitPayload payload{};
      const auto n = read(channel[0], &payload, sizeof(payload));
      close(channel[0]);
      if (n != sizeof(payload))
        _exit(1);
      unique_id = payload.id;
      world = payload.total_world;
      worker_rank = payload.rank_offset + r;
      device = devices[r];
      return -1;
    }
    children.push_back(child);
  }
  close(channel[0]);

  // No CUDA or NCCL calls occur until AFTER all forks.
  InitPayload local_payload{};
  std::vector<int> peer_socks;
  std::vector<bool> peer_ok;
  if (!result) {
    if (num_nodes == 1) {
      if (ncclGetUniqueId(&local_payload.id) != ncclSuccess)
        result = 1;
      local_payload.total_world = local_world;
      local_payload.rank_offset = 0;
    } else if (node_rank == 0) {
      if (ncclGetUniqueId(&local_payload.id) != ncclSuccess) {
        result = 1;
      } else {
        int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd < 0) {
          result = 1;
        } else {
          int opt = 1;
          setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
          sockaddr_in addr{};
          addr.sin_family = AF_INET;
          addr.sin_addr.s_addr = htonl(INADDR_ANY);
          addr.sin_port = htons(uint16_t(master_port));
          if (::bind(listen_fd, reinterpret_cast<sockaddr *>(&addr),
                     sizeof(addr)) < 0 ||
              ::listen(listen_fd, num_nodes) < 0) {
            std::cerr << "DP master bind/listen failed on port " << master_port
                      << '\n';
            result = 1;
          } else {
            std::vector<int> node_counts(num_nodes, 0);
            std::vector<int> node_fds(num_nodes, -1);
            node_counts[0] = local_world;
            for (int accepted = 1; accepted < num_nodes && !result && !interrupted;
                 ++accepted) {
              int cfd = ::accept(listen_fd, nullptr, nullptr);
              if (cfd < 0) {
                if (errno == EINTR) {
                  --accepted;
                  continue;
                }
                result = 1;
                break;
              }
              setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
              PeerHello hello{};
              if (!read_exact(cfd, &hello, sizeof(hello)) ||
                  hello.node_rank <= 0 || hello.node_rank >= num_nodes ||
                  hello.local_world <= 0 || node_fds[hello.node_rank] >= 0) {
                ::close(cfd);
                result = 1;
                break;
              }
              node_counts[hello.node_rank] = hello.local_world;
              node_fds[hello.node_rank] = cfd;
            }
            if (interrupted)
              result = 128 + interrupted;
            if (!result) {
              std::vector<int> offsets(num_nodes, 0);
              int total = 0;
              for (int n = 0; n < num_nodes; ++n) {
                offsets[n] = total;
                total += node_counts[n];
              }
              local_payload.total_world = total;
              local_payload.rank_offset = 0;
              for (int n = 1; n < num_nodes; ++n) {
                InitPayload peer_p{local_payload.id, total, offsets[n]};
                if (!write_exact(node_fds[n], &peer_p, sizeof(peer_p))) {
                  result = 1;
                  break;
                }
                set_nonblocking(node_fds[n]);
                peer_socks.push_back(node_fds[n]);
                node_fds[n] = -1;
              }
            }
            for (int fd : node_fds)
              if (fd >= 0)
                ::close(fd);
          }
          ::close(listen_fd);
        }
      }
    } else {
      // Worker node supervisor: connect to master with retry up to 120s.
      addrinfo hints{}, *res = nullptr;
      hints.ai_family = AF_INET;
      hints.ai_socktype = SOCK_STREAM;
      const std::string port_str = std::to_string(master_port);
      if (getaddrinfo(master_host.c_str(), port_str.c_str(), &hints, &res) != 0 ||
          !res) {
        std::cerr << "DP worker failed to resolve " << master_host << '\n';
        result = 1;
      } else {
        int sock = -1;
        for (int attempt = 0; attempt < 480 && !interrupted; ++attempt) {
          sock = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
          if (sock >= 0 &&
              ::connect(sock, res->ai_addr, res->ai_addrlen) == 0) {
            int opt = 1;
            setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
            break;
          }
          if (sock >= 0) {
            ::close(sock);
            sock = -1;
          }
          usleep(250000);
        }
        freeaddrinfo(res);
        if (interrupted)
          result = 128 + interrupted;
        if (sock < 0) {
          if (!result)
            result = 1;
          std::cerr << "DP worker timed out connecting to master "
                    << master_host << ":" << master_port << '\n';
        } else {
          PeerHello hello{node_rank, local_world};
          if (!write_exact(sock, &hello, sizeof(hello)) ||
              !read_exact(sock, &local_payload, sizeof(local_payload))) {
            ::close(sock);
            result = 1;
          } else {
            set_nonblocking(sock);
            peer_socks.push_back(sock);
          }
        }
      }
    }
  }
  peer_ok.assign(peer_socks.size(), false);

  if (!result)
    for (size_t i = 0; i < children.size(); ++i)
      if (!write_exact(channel[1], &local_payload, sizeof(local_payload))) {
        result = 1;
        break;
      }
  close(channel[1]);

  bool abort_sent = false;
  auto broadcast_abort = [&]() {
    if (abort_sent)
      return;
    abort_sent = true;
    const char c = 'A';
    for (int fd : peer_socks)
      if (fd >= 0)
        (void)::send(fd, &c, 1, MSG_NOSIGNAL);
  };

  while (!children.empty()) {
    if (interrupted && !result)
      result = 128 + interrupted;
    // Check peer node supervisors for remote crash/cancellation.
    for (size_t i = 0; i < peer_socks.size(); ++i) {
      int fd = peer_socks[i];
      if (fd < 0)
        continue;
      char msg = 0;
      const ssize_t r = ::recv(fd, &msg, 1, MSG_DONTWAIT);
      if (r == 1) {
        if (msg == 'O') {
          peer_ok[i] = true;
        } else if (msg == 'A') {
          if (!result)
            result = 1;
          std::cerr << "DP remote node aborted; terminating local ranks\n";
        }
      } else if (r == 0) {
        if (!peer_ok[i]) {
          if (!result)
            result = 1;
          std::cerr << "DP remote node disconnected unexpectedly; terminating local ranks\n";
        }
        ::close(fd);
        peer_socks[i] = -1;
      } else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
        if (!peer_ok[i] && !result)
          result = 1;
        ::close(fd);
        peer_socks[i] = -1;
      }
    }
    if (result) {
      broadcast_abort();
      for (auto pid : children)
        kill(pid, SIGKILL);
    }
    int status = 0;
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
  if (result) {
    broadcast_abort();
  } else {
    const char ok = 'O';
    for (int fd : peer_socks)
      if (fd >= 0)
        (void)::send(fd, &ok, 1, MSG_NOSIGNAL);
  }
  for (int fd : peer_socks)
    if (fd >= 0)
      ::close(fd);
  sigaction(SIGINT, &old_int, nullptr);
  sigaction(SIGTERM, &old_term, nullptr);
  sigaction(SIGPIPE, &old_pipe, nullptr);
  return result;
}
Group::Group() {
  rwkv_bf16_training::gpu_check(cudaSetDevice(device), "DP select GPU");
  bind_numa_cpu_affinity(device);
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
        scale[r] = std::clamp((*inv_speeds)[r], 0.35, 3.0);
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
    const double rel = std::clamp(spu[r] / mean_spu, 0.35, 3.0);
    inv_speeds[r] = ready ? (0.75 * inv_speeds[r] + 0.25 * rel) : rel;
  }
  ready = true;
}
} // namespace rwkv_dp
