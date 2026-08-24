import unittest

import torch

from delta.quantization import (
    DIRECTION_CANDIDATES,
    SINQLinearFactor,
    _ActivationAwareScorer,
    _quantize_candidate,
    activation_aware_full_weight_mse,
)
from delta.rank import BIT_CANDIDATES
from delta.svd import (
    build_svdllm_factors,
    decompose_svdllm,
    factors_from_svdllm_decomposition,
)


class QuantizationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.weight = torch.randn(64, 64)

    def test_all_bits_and_directions_roundtrip(self):
        for bits in BIT_CANDIDATES:
            for params in DIRECTION_CANDIDATES:
                candidate = _quantize_candidate(
                    self.weight,
                    bits,
                    params,
                    device="cpu",
                    metadata_dtype=torch.float16,
                )
                reconstructed = candidate["reconstructed"]
                self.assertEqual(tuple(reconstructed.shape), tuple(self.weight.shape))
                self.assertTrue(torch.isfinite(reconstructed).all())
                self.assertEqual(candidate["meta"]["scale"].dtype, torch.float16)
                self.assertEqual(candidate["meta"]["zero"].dtype, torch.float16)

    def test_factor_spec_preserves_non_tensor_metadata(self):
        candidate = _quantize_candidate(
            self.weight,
            4,
            {"optimize": False, "axis": 0},
            device="cpu",
            metadata_dtype=torch.float16,
        )
        factor = SINQLinearFactor(
            candidate["qweight"],
            candidate["meta"],
            candidate["params"],
            nbits=4,
            rank=64,
            score=0.1,
        )
        rebuilt = SINQLinearFactor.empty_from_spec(factor.to_spec())
        rebuilt.load_state_dict(factor.state_dict(), strict=True)
        self.assertEqual(rebuilt.params, factor.params)
        self.assertEqual(rebuilt.meta["axis"], 0)
        self.assertEqual(rebuilt.meta["shape"], (64, 64))
        torch.testing.assert_close(rebuilt.dequantize(), factor.dequantize())

        inputs = torch.randn(3, 64)
        expected = rebuilt(inputs)
        state_keys = set(rebuilt.state_dict())
        rebuilt.cache_dequantized(device=torch.device("cpu"), dtype=torch.float32)
        actual = rebuilt(inputs)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(set(rebuilt.state_dict()), state_keys)
        rebuilt.clear_dequantized_cache()
        self.assertIsNone(rebuilt._cached_weight)

    def test_shared_svd_decomposition_matches_direct_build(self):
        cholesky = torch.tril(torch.randn(64, 64))
        cholesky.diagonal().add_(8.0)
        decomposition = decompose_svdllm(self.weight, cholesky)
        for rank in (7, 19, 37):
            expected_u, expected_v = build_svdllm_factors(
                self.weight, cholesky, rank
            )
            actual_u, actual_v = factors_from_svdllm_decomposition(
                decomposition, rank
            )
            torch.testing.assert_close(actual_u, expected_u)
            torch.testing.assert_close(actual_v, expected_v)

    def test_cached_activation_score_matches_full_weight_formula(self):
        cholesky = torch.tril(torch.randn(64, 64))
        cholesky.diagonal().add_(8.0)
        scorer = _ActivationAwareScorer(self.weight, cholesky)
        for rank in (7, 19, 37):
            quant_u = torch.randn(64, rank)
            quant_v = torch.randn(rank, 64)
            expected = activation_aware_full_weight_mse(
                self.weight, quant_u, quant_v, cholesky
            )
            actual = scorer.score(quant_u, quant_v)
            self.assertAlmostEqual(actual, expected, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
