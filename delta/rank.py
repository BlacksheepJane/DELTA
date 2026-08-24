"""Rank and storage accounting for DELTA projections."""

import math
from typing import Optional


BIT_CANDIDATES = (3, 4, 5, 6, 8)
DEFAULT_GROUP_SIZE = 64
DEFAULT_METADATA_BYTES_PER_GROUP = 4.0


def packed_payload_bits(nbits: int) -> float:
    """Return the effective payload bits used by the tested SINQ bit packer."""
    nbits = int(nbits)
    if nbits == 3:
        return 32.0 / 10.0
    if nbits == 5:
        return 32.0 / 6.0
    if nbits == 6:
        return 32.0 / 5.0
    if nbits in (4, 8):
        return float(nbits)
    raise ValueError("Unsupported bit width: {}".format(nbits))


def effective_bits_per_value(
    nbits: int,
    quant_group_size: int = DEFAULT_GROUP_SIZE,
    metadata_bytes_per_group: float = DEFAULT_METADATA_BYTES_PER_GROUP,
) -> float:
    if quant_group_size <= 0:
        raise ValueError("quant_group_size must be positive")
    return packed_payload_bits(nbits) + (
        8.0 * float(metadata_bytes_per_group) / float(quant_group_size)
    )


def compute_delta_rank(
    out_features: int,
    in_features: int,
    target_ratio: float,
    ubits: int,
    vbits: int,
    rank_group_size: int = DEFAULT_GROUP_SIZE,
    quant_group_size: int = DEFAULT_GROUP_SIZE,
    u_metadata_bytes_per_group: Optional[float] = None,
    v_metadata_bytes_per_group: Optional[float] = None,
) -> int:
    """Compute a grouped rank under DELTA's projection storage budget."""
    if out_features <= 0 or in_features <= 0:
        raise ValueError("matrix dimensions must be positive")
    if not 0.0 < target_ratio <= 1.0:
        raise ValueError("target_ratio must be in (0, 1]")
    if rank_group_size <= 0:
        raise ValueError("rank_group_size must be positive")
    if ubits not in BIT_CANDIDATES or vbits not in BIT_CANDIDATES:
        raise ValueError("bits must be selected from {}".format(BIT_CANDIDATES))

    u_overhead = (
        DEFAULT_METADATA_BYTES_PER_GROUP
        if u_metadata_bytes_per_group is None
        else float(u_metadata_bytes_per_group)
    )
    v_overhead = (
        DEFAULT_METADATA_BYTES_PER_GROUP
        if v_metadata_bytes_per_group is None
        else float(v_metadata_bytes_per_group)
    )
    u_eff_bits = effective_bits_per_value(ubits, quant_group_size, u_overhead)
    v_eff_bits = effective_bits_per_value(vbits, quant_group_size, v_overhead)
    raw_rank = (
        16.0 * out_features * in_features * target_ratio
        / (out_features * u_eff_bits + in_features * v_eff_bits)
    )
    rank = int(math.floor(raw_rank / rank_group_size) * rank_group_size)
    rank = max(rank_group_size, rank)
    return min(rank, out_features, in_features)


def estimate_projection_storage_bytes(
    out_features: int,
    in_features: int,
    rank: int,
    ubits: int,
    vbits: int,
    quant_group_size: int = DEFAULT_GROUP_SIZE,
) -> float:
    """Estimate packed U/V payload and fp16 scale/zero metadata storage."""
    u_bits = effective_bits_per_value(ubits, quant_group_size)
    v_bits = effective_bits_per_value(vbits, quant_group_size)
    return (out_features * rank * u_bits + rank * in_features * v_bits) / 8.0
