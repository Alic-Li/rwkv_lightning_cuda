#include "../online_rl/server.hpp"
int main(int argc, char **argv) {
  return rwkv_online_rl::run(argc, argv, false);
}
