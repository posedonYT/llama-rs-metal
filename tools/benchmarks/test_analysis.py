"""Statistical reporting tests: pairing, direction and confound exclusion."""

import unittest

from summarize import summarize


def record(block, engine, speed, confounded=False):
    return dict(
        model="fixture",
        track="matched_f32",
        engine=engine,
        block=block,
        confounded=confounded,
        cells=[
            dict(
                workload="p128-d64",
                runs=[
                    dict(
                        prefill_tokens_per_second=speed,
                        decode_tokens_per_second=speed,
                        request_ttft_seconds=1 / speed,
                        reset_seconds=1 / speed,
                    )
                ]
                * 2,
            )
        ],
    )


class AnalysisTests(unittest.TestCase):
    def test_exact_ratio_and_latency_direction(self):
        rows = [
            r for b in range(4) for r in [record(b, "rust", 20), record(b, "mlx", 10)]
        ]
        for metric in summarize(rows)[0]["metrics"].values():
            self.assertAlmostEqual(metric["rust_speed_ratio"], 2)
            self.assertEqual(metric["paired_bootstrap95"], [2.0, 2.0])
            self.assertEqual(metric["valid_paired_blocks"], 4)

    def test_confounded_pair_is_preserved_but_excluded(self):
        rows = [
            r for b in range(3) for r in [record(b, "rust", 20), record(b, "mlx", 10)]
        ]
        rows += [record(3, "rust", 1000, True), record(3, "mlx", 10)]
        metric = summarize(rows)[0]["metrics"]["decode_tokens_per_second"]
        self.assertAlmostEqual(metric["rust_speed_ratio"], 2)
        self.assertEqual(metric["confounded_blocks"], [3])
        self.assertEqual(metric["engines"]["rust"]["samples"], 8)
        self.assertEqual(metric["valid_paired_engines"]["rust"]["samples"], 6)

    def test_unpaired_observation_does_not_create_a_ratio(self):
        result = summarize([record(0, "rust", 20), record(1, "mlx", 10)])[0]
        self.assertIsNone(
            result["metrics"]["decode_tokens_per_second"]["rust_speed_ratio"]
        )


if __name__ == "__main__":
    unittest.main()
