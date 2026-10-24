"""CPU regression for string-stopped, length-limited multi-turn RL episodes."""
import argparse
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "online_rl", Path(__file__).resolve().parents[1] / "tools/online_rl.py")
online_rl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(online_rl)


def completion(tokens, reason="stop", matched="</tool_call>", truncation=None):
    row = dict(tokens=tokens, logprobs=[-1.0] * len(tokens), finish_reason=reason,
               truncated=reason == "length")
    if matched is not None:
        row["matched_stop"] = matched
    if truncation:
        row["truncation_reason"] = truncation
    return row


class FakeAPI:
    def __init__(self, batches):
        self.batches = list(batches)
        self.requests = []

    def call(self, path, body):
        assert path == "/v1/completions"
        self.requests.append(copy.deepcopy(body))
        batch = self.batches.pop(0)
        return dict(policy_version="v", completions=batch, cached_prompt_tokens=0,
                    prefill_tokens=sum(map(len, body["prompts"])),
                    generated_tokens=sum(len(c["tokens"]) for c in batch), seconds=1.0)


class RolloutTest(unittest.TestCase):
    def collect(self, api, rounds, **kwargs):
        options = dict(group_size=2, version="v", max_tokens=1024, ctx=4096,
                       temperature=1.0, seed=3, judge=online_rl.token_f1)
        options.update(kwargs)
        return online_rl.collect_rollouts(api, rounds, **options)

    def test_truncated_episode_does_not_replay_more_observations(self):
        api = FakeAPI([
            [completion([10, 11]), completion([20, 21, 22], "length", None, "max_tokens")],
            [completion([12], matched="\n\nUser:")],
        ])
        result = self.collect(api, [([1, 2], [10, 11]), ([3], [12])])
        self.assertEqual(api.requests[1]["prompts"], [[1, 2, 10, 11, 3]])
        self.assertNotIn("stop", api.requests[0])  # inherit the server's defaults
        self.assertEqual(api.requests[0]["max_tokens"], 1024)
        self.assertEqual(result["completed_turns"], [2, 1])
        self.assertEqual(result["truncated"], [False, True])
        self.assertEqual(result["truncation_reasons"], [None, "max_tokens"])
        self.assertEqual(result["turns"][1], [
            dict(tokens=[1, 2], loss_mask=[0, 0]),
            dict(tokens=[20, 21, 22], loss_mask=[1, 1, 1]),
        ])
        self.assertEqual(result["logprob_sums"], [-3.0, -3.0])
        self.assertEqual(result["generated"], 6)
        self.assertEqual(result["rewards"], [1.0, 0.0])

    def test_context_uses_actual_lengths_not_worst_case_cap(self):
        api = FakeAPI([[completion([10]), completion([20, 21])], [completion([11])]])
        result = self.collect(api, [([1, 2], [10]), ([3], [11])], ctx=5)
        # The 1024-token safety cap does not prevent a short, stopped turn.
        self.assertEqual(len(api.requests), 2)
        self.assertEqual(api.requests[1]["prompts"], [[1, 2, 10, 3]])
        self.assertEqual(result["truncated"], [False, True])
        self.assertEqual(result["truncation_reasons"][1], "context_length")
        self.assertEqual(len(result["turns"][1]), 2)  # no overflowing observation

    def test_all_truncated_stops_the_group(self):
        api = FakeAPI([[completion([10], "length", None, "max_tokens"),
                        completion([20], "length", None, "context_length")]])
        result = self.collect(api, [([1], [10]), ([2], [11])], stop=["END"])
        self.assertEqual(len(api.requests), 1)
        self.assertEqual(api.requests[0]["stop"], ["END"])
        self.assertEqual(result["completed_turns"], [1, 1])
        self.assertEqual(result["truncated"], [True, True])

    def test_stop_cli_json_and_explicit_disable(self):
        self.assertEqual(online_rl.parse_stops(r'["</tool_call>", "\n\nUser:"]'),
                         ["</tool_call>", "\n\nUser:"])
        self.assertEqual(online_rl.parse_stops("[]"), [])
        self.assertEqual(online_rl.parse_stops('["结束"]'), ["结束"])
        for value in ('[""]', 'null', '"END"', '[1]', '[', '["' + 'x' * 257 + '"]',
                      '["' + '中' * 86 + '"]', r'["\ud800"]'):
            with self.assertRaises(argparse.ArgumentTypeError):
                online_rl.parse_stops(value)


if __name__ == "__main__":
    unittest.main()
