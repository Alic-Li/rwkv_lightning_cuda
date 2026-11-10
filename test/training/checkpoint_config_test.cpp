#include "rwkv/bf16_training/checkpoint_config.hpp"
#include <iostream>
#include <stdexcept>

void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}

int main() {
  try {
    Json::Value config;
    config["base_fingerprint"] = "test-backbone";
    config["training_dtype"] = "bfloat16";
    config["kind"] = "online_rl";
    config["rank"] = 8;
    config["alpha"] = 8;
    config["targets"] = "all";
    const auto original = config;
    // Reproduce the old writer's ordering, including its second-write failure.
    const auto first = config;
    (void)config["seed"].asUInt64();
    (void)config["planned_steps"];
    require(first.size() == 6 && config.size() == 8,
            "missing-key reads did not reproduce config pollution");
    require(first != config, "polluted config unexpectedly equals clean config");
    using rwkv_bf16_miss::checkpoint_config_matches;
    require(checkpoint_config_matches(first, original), "clean config rejected");
    require(checkpoint_config_matches(config, original), "legacy null keys rejected");
    require(config.size() == 8 && original.size() == 6, "comparison mutated inputs");
    for (const char *key : {"seed", "planned_steps"}) {
      auto saved = original;
      saved[key] = Json::nullValue;
      require(checkpoint_config_matches(saved, original), "single null key rejected");
      saved[key] = 0;
      require(!checkpoint_config_matches(saved, original), "real optional value ignored");
      auto expected = saved;
      require(checkpoint_config_matches(saved, expected), "explicit value rejected");
      saved[key] = Json::nullValue;
      require(!checkpoint_config_matches(saved, expected), "required value ignored");
    }
    for (const char *key : {"base_fingerprint", "training_dtype", "kind", "rank", "alpha", "targets"}) {
      auto saved = config;
      saved[key] = "mismatch";
      require(!checkpoint_config_matches(saved, original), "config mismatch ignored");
    }
    auto extra = config;
    extra["unexpected"] = Json::nullValue;
    require(!checkpoint_config_matches(extra, original), "unrelated null key ignored");
    auto cli = original;
    cli.removeMember("kind");
    extra = cli;
    extra["seed"] = Json::nullValue;
    require(!checkpoint_config_matches(extra, cli), "CLI config validation relaxed");
    // Const reads used by Trainer must leave all five snapshots identical.
    const auto &readonly = original;
    for (int i = 0; i < 5; ++i) {
      auto saved = readonly;
      require(readonly["seed"].asUInt64() == 0, "missing seed default changed");
      require(readonly["planned_steps"].isNull(), "missing scheduler default changed");
      require(saved == readonly && saved.size() == 6, "const read mutated config");
    }
    std::cout << "Checkpoint config regression passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
