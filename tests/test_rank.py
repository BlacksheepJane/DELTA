import unittest

from delta.rank import (
    compute_delta_rank,
    estimate_projection_storage_bytes,
    packed_payload_bits,
)


class RankAccountingTest(unittest.TestCase):
    def test_packed_payload_bits(self):
        self.assertEqual(packed_payload_bits(4), 4.0)
        self.assertEqual(packed_payload_bits(8), 8.0)
        self.assertAlmostEqual(packed_payload_bits(3), 3.2)
        self.assertAlmostEqual(packed_payload_bits(5), 32.0 / 6.0)
        self.assertAlmostEqual(packed_payload_bits(6), 6.4)

    def test_rank_is_grouped_and_within_matrix(self):
        rank = compute_delta_rank(4096, 4096, 0.4, 4, 4)
        self.assertEqual(rank % 64, 0)
        self.assertGreater(rank, 0)
        self.assertLessEqual(rank, 4096)

    def test_estimated_storage_respects_projection_budget(self):
        out_features, in_features = 4096, 11008
        ratio = 0.4
        rank = compute_delta_rank(out_features, in_features, ratio, 6, 4)
        compressed = estimate_projection_storage_bytes(
            out_features, in_features, rank, 6, 4
        )
        dense_fp16 = out_features * in_features * 2
        self.assertLessEqual(compressed, dense_fp16 * ratio)
        next_rank = min(rank + 64, min(out_features, in_features))
        if next_rank > rank:
            self.assertGreater(
                estimate_projection_storage_bytes(
                    out_features, in_features, next_rank, 6, 4
                ),
                compressed,
            )


if __name__ == "__main__":
    unittest.main()
