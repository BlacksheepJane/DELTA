"""Activation-aware SINQ quantization used by DELTA."""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from ._vendor.sinq.quantizer import Quantizer
from .rank import BIT_CANDIDATES, DEFAULT_GROUP_SIZE, compute_delta_rank
from .svd import decompose_svdllm, factors_from_svdllm_decomposition


DIRECTION_CANDIDATES = (
    {"optimize": False, "axis": 0},
    {"optimize": False, "axis": 1},
    {"optimize": True, "axis": 1},
)
COARSE_BIT_CANDIDATES = (3, 4, 6, 8)
MAX_UV_BIT_GAP = 2


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _dtype_from_name(name: str) -> torch.dtype:
    value = getattr(torch, str(name).replace("torch.", ""), None)
    if not isinstance(value, torch.dtype):
        raise ValueError("Unknown torch dtype: {}".format(name))
    return value


def _encode_json_value(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return {"__type__": "torch.dtype", "value": _dtype_name(value)}
    if isinstance(value, tuple):
        return {
            "__type__": "tuple",
            "items": [_encode_json_value(item) for item in value],
        }
    if isinstance(value, list):
        return [_encode_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode_json_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("Unsupported metadata value: {!r}".format(type(value)))


def _decode_json_value(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__type__") == "torch.dtype":
        return _dtype_from_name(value["value"])
    if isinstance(value, dict) and value.get("__type__") == "tuple":
        return tuple(_decode_json_value(item) for item in value["items"])
    if isinstance(value, list):
        return [_decode_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _decode_json_value(item) for key, item in value.items()}
    return value


def _cast_metadata_value(value: Any, dtype: Optional[torch.dtype]) -> Any:
    if dtype is None:
        return value
    if torch.is_tensor(value):
        return value.to(dtype) if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: _cast_metadata_value(item, dtype) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cast_metadata_value(item, dtype) for item in value)
    if isinstance(value, list):
        return [_cast_metadata_value(item, dtype) for item in value]
    return value


def cast_quant_metadata(meta: Dict[str, Any], dtype: torch.dtype) -> Dict[str, Any]:
    return {key: _cast_metadata_value(value, dtype) for key, value in meta.items()}


def _metadata_from_factor(factor: "SINQLinearFactor") -> Dict[str, Any]:
    meta = dict(factor.meta)
    for name, tensor in factor._buffers.items():
        if name.startswith("meta_"):
            meta[name[len("meta_") :]] = tensor
    return meta


class SINQLinearFactor(nn.Module):
    """One packed linear factor plus all metadata required to dequantize it."""

    def __init__(
        self,
        qweight: torch.Tensor,
        meta: Dict[str, Any],
        params: Dict[str, Any],
        nbits: int,
        rank: int,
        score: Optional[float] = None,
        dense_score: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.params = {"optimize": bool(params["optimize"]), "axis": int(params["axis"])}
        self.nbits = int(nbits)
        self.rank = int(rank)
        self.score = None if score is None else float(score)
        self.dense_score = None if dense_score is None else float(dense_score)
        self._cached_weight = None
        self.register_buffer("qweight", qweight)
        self.meta = {}
        for key, value in meta.items():
            if torch.is_tensor(value):
                self.register_buffer("meta_" + key, value)
            else:
                self.meta[key] = value

    def dequantize(self) -> torch.Tensor:
        return Quantizer.dequantize(self.qweight, _metadata_from_factor(self))

    def cache_dequantized(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        target_device = self.qweight.device if device is None else torch.device(device)
        self._cached_weight = self.dequantize().to(
            device=target_device, dtype=dtype
        )

    def clear_dequantized_cache(self) -> None:
        self._cached_weight = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        compute_dtype = (
            torch.float32
            if inputs.dtype in (torch.float16, torch.bfloat16)
            else inputs.dtype
        )
        cached = self._cached_weight
        if (
            cached is not None
            and cached.device == inputs.device
            and cached.dtype == compute_dtype
        ):
            weight = cached
        else:
            weight = self.dequantize().to(
                device=inputs.device, dtype=compute_dtype
            )
        output = inputs.to(compute_dtype).matmul(weight.t())
        return output.to(inputs.dtype)

    def to_spec(self) -> Dict[str, Any]:
        tensor_meta = {}
        for name, tensor in self._buffers.items():
            if name.startswith("meta_"):
                tensor_meta[name[len("meta_") :]] = {
                    "shape": list(tensor.shape),
                    "dtype": _dtype_name(tensor.dtype),
                }
        return {
            "qweight": {
                "shape": list(self.qweight.shape),
                "dtype": _dtype_name(self.qweight.dtype),
            },
            "tensor_meta": tensor_meta,
            "meta": _encode_json_value(self.meta),
            "params": dict(self.params),
            "nbits": self.nbits,
            "rank": self.rank,
            "score": self.score,
            "dense_score": self.dense_score,
        }

    @classmethod
    def empty_from_spec(cls, spec: Dict[str, Any]) -> "SINQLinearFactor":
        qweight_spec = spec["qweight"]
        qweight = torch.empty(
            qweight_spec["shape"], dtype=_dtype_from_name(qweight_spec["dtype"])
        )
        meta = _decode_json_value(spec["meta"])
        for key, tensor_spec in spec["tensor_meta"].items():
            meta[key] = torch.empty(
                tensor_spec["shape"], dtype=_dtype_from_name(tensor_spec["dtype"])
            )
        return cls(
            qweight=qweight,
            meta=meta,
            params=spec["params"],
            nbits=spec["nbits"],
            rank=spec["rank"],
            score=spec.get("score"),
            dense_score=spec.get("dense_score"),
        )


def activation_aware_full_weight_mse(
    weight: torch.Tensor,
    quant_u: torch.Tensor,
    quant_v: torch.Tensor,
    cholesky: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Score ||(W - qUqV)C||_F^2 / ||WC||_F^2."""
    device = weight.device
    weight = weight.detach().to(device=device, dtype=torch.float32)
    quant_u = quant_u.detach().to(device=device, dtype=torch.float32)
    quant_v = quant_v.detach().to(device=device, dtype=torch.float32)
    cholesky = cholesky.detach().to(device=device, dtype=torch.float32)
    reconstructed = quant_u.matmul(quant_v)
    reference = weight.matmul(cholesky)
    error = (weight - reconstructed).matmul(cholesky)
    return (error.square().sum() / (reference.square().sum() + eps)).item()


class _ActivationAwareScorer:
    """Reuse the reference activation and score low-rank factors efficiently."""

    def __init__(
        self,
        weight: torch.Tensor,
        cholesky: torch.Tensor,
        eps: float = 1e-12,
    ) -> None:
        self.device = weight.device
        self.cholesky = cholesky.detach().to(
            device=self.device, dtype=torch.float32
        )
        self.reference = weight.detach().to(torch.float32).matmul(self.cholesky)
        self.denominator = self.reference.square().sum() + eps

    def prepare_u(self, quant_u: torch.Tensor) -> torch.Tensor:
        return quant_u.detach().to(device=self.device, dtype=torch.float32)

    def prepare_v(self, quant_v: torch.Tensor) -> torch.Tensor:
        quant_v = quant_v.detach().to(device=self.device, dtype=torch.float32)
        return quant_v.matmul(self.cholesky)

    def score_prepared(
        self,
        prepared_u: torch.Tensor,
        prepared_v: torch.Tensor,
    ) -> float:
        error = self.reference - prepared_u.matmul(prepared_v)
        return (error.square().sum() / self.denominator).item()

    def score(self, quant_u: torch.Tensor, quant_v: torch.Tensor) -> float:
        return self.score_prepared(
            self.prepare_u(quant_u),
            self.prepare_v(quant_v),
        )


def _quantize_candidate(
    weight: torch.Tensor,
    nbits: int,
    params: Dict[str, Any],
    device: str,
    metadata_dtype: torch.dtype,
) -> Dict[str, Any]:
    qweight, meta = Quantizer.quantize(
        weight,
        layer_activations=None,
        nbits=int(nbits),
        channel_wise=True,
        group_size=DEFAULT_GROUP_SIZE,
        optimize=bool(params["optimize"]),
        round_zero=False,
        axis=int(params["axis"]),
        bitpack=True,
        compute_dtype=torch.float16,
        view_as_float=False,
        device=device,
        tiling_mode="1D",
        method="dual",
    )
    meta = cast_quant_metadata(meta, metadata_dtype)
    return {
        "qweight": qweight,
        "meta": meta,
        "params": dict(params),
        "reconstructed": Quantizer.dequantize(qweight, meta),
    }


def _quantize_factor_candidates(
    weight: torch.Tensor,
    nbits: int,
    device: str,
    metadata_dtype: torch.dtype,
) -> List[Dict[str, Any]]:
    return [
        _quantize_candidate(weight, nbits, params, device, metadata_dtype)
        for params in DIRECTION_CANDIDATES
    ]


def _adjacent_bits(bits: int) -> Tuple[int, ...]:
    index = BIT_CANDIDATES.index(int(bits))
    return BIT_CANDIDATES[max(0, index - 1) : min(len(BIT_CANDIDATES), index + 2)]


@torch.no_grad()
def search_delta_factors(
    weight: torch.Tensor,
    cholesky: torch.Tensor,
    target_ratio: float,
    device: str,
    metadata_dtype: torch.dtype = torch.float16,
) -> Tuple[SINQLinearFactor, SINQLinearFactor, Dict[str, Any]]:
    """Search U/V bit widths and SINQ directions against the full weight."""
    out_features, in_features = weight.shape
    best = None
    factor_cache = {}
    evaluated = set()
    decomposition = decompose_svdllm(weight, cholesky)
    scorer = _ActivationAwareScorer(weight, cholesky)

    def evaluate_pair(ubits: int, vbits: int) -> None:
        nonlocal best
        pair = (int(ubits), int(vbits))
        if pair in evaluated:
            return
        evaluated.add(pair)
        rank = compute_delta_rank(
            out_features, in_features, target_ratio, pair[0], pair[1]
        )
        if rank not in factor_cache:
            factor_cache[rank] = factors_from_svdllm_decomposition(
                decomposition, rank, output_dtype=torch.float32
            )
        u_factor, v_factor = factor_cache[rank]
        dense_score = scorer.score(u_factor, v_factor)
        u_candidates = _quantize_factor_candidates(
            u_factor, pair[0], device, metadata_dtype
        )
        v_candidates = _quantize_factor_candidates(
            v_factor, pair[1], device, metadata_dtype
        )
        prepared_u = [
            scorer.prepare_u(candidate["reconstructed"])
            for candidate in u_candidates
        ]
        prepared_v = [
            scorer.prepare_v(candidate["reconstructed"])
            for candidate in v_candidates
        ]
        for u_index, u_candidate in enumerate(u_candidates):
            for v_index, v_candidate in enumerate(v_candidates):
                score = scorer.score_prepared(
                    prepared_u[u_index], prepared_v[v_index]
                )
                if best is None or score < best["score"]:
                    best = {
                        "score": score,
                        "dense_score": dense_score,
                        "rank": rank,
                        "ubits": pair[0],
                        "vbits": pair[1],
                        "u": u_candidate,
                        "v": v_candidate,
                    }

    for bits in COARSE_BIT_CANDIDATES:
        evaluate_pair(bits, bits)
    if best is not None:
        coarse_best = (best["ubits"], best["vbits"])
        for ubits in _adjacent_bits(coarse_best[0]):
            for vbits in _adjacent_bits(coarse_best[1]):
                if abs(ubits - vbits) <= MAX_UV_BIT_GAP:
                    evaluate_pair(ubits, vbits)
    if best is None:
        raise RuntimeError("DELTA search did not produce a valid quantization")

    u_factor = SINQLinearFactor(
        qweight=best["u"]["qweight"],
        meta=best["u"]["meta"],
        params=best["u"]["params"],
        nbits=best["ubits"],
        rank=best["rank"],
        score=best["score"],
        dense_score=best["dense_score"],
    )
    v_factor = SINQLinearFactor(
        qweight=best["v"]["qweight"],
        meta=best["v"]["meta"],
        params=best["v"]["params"],
        nbits=best["vbits"],
        rank=best["rank"],
        score=best["score"],
        dense_score=best["dense_score"],
    )
    choice = {
        "rank": best["rank"],
        "score": best["score"],
        "dense_score": best["dense_score"],
        "u": {"nbits": best["ubits"], "params": best["u"]["params"]},
        "v": {"nbits": best["vbits"], "params": best["v"]["params"]},
    }
    return u_factor, v_factor, choice
