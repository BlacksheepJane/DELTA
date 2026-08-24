"""SVD-LLM Cholesky-whitened low-rank decomposition."""

from typing import Tuple

import torch


SVDLLMDecomposition = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def decompose_svdllm(
    weight: torch.Tensor,
    cholesky: torch.Tensor,
) -> SVDLLMDecomposition:
    """Compute the rank-independent SVD-LLM decomposition."""
    device = weight.device
    cholesky = cholesky.to(device=device, dtype=torch.float32)
    try:
        cholesky_inv = torch.linalg.inv(cholesky)
    except RuntimeError:
        jitter = 1e-3 * torch.eye(cholesky.shape[0], device=device)
        cholesky = cholesky + jitter
        cholesky_inv = torch.linalg.inv(cholesky)

    whitened_weight = weight.float().matmul(cholesky)
    left, singular_values, right_t = torch.linalg.svd(
        whitened_weight, full_matrices=False
    )
    return left, singular_values, right_t, cholesky_inv


def factors_from_svdllm_decomposition(
    decomposition: SVDLLMDecomposition,
    rank: int,
    output_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Truncate a shared decomposition and build balanced U/V factors."""
    left, singular_values, right_t, cholesky_inv = decomposition
    rank = min(int(rank), int(singular_values.numel()))
    if rank <= 0:
        raise ValueError("rank must be positive")

    sqrt_s = torch.sqrt(singular_values[:rank])
    u_factor = left[:, :rank] * sqrt_s.unsqueeze(0)
    v_factor = sqrt_s.unsqueeze(1) * right_t[:rank, :].matmul(cholesky_inv)
    return (
        u_factor.cpu().to(output_dtype),
        v_factor.cpu().to(output_dtype),
    )


def build_svdllm_factors(
    weight: torch.Tensor,
    cholesky: torch.Tensor,
    rank: int,
    output_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build balanced U/V factors using the original SVD-LLM formulation."""
    decomposition = decompose_svdllm(weight, cholesky)
    return factors_from_svdllm_decomposition(
        decomposition, rank, output_dtype=output_dtype
    )
