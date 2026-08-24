"""Command-line entry point for DELTA compression of LLaMA-7B."""

import argparse
import concurrent.futures
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .checkpoint import CHECKPOINT_FORMAT, save_delta_checkpoint
from .data import get_calibration_data
from .modeling import DeltaLinear, set_submodule
from .profiling import PROJECTION_NAMES, profile_cholesky
from .quantization import search_delta_factors
from .rank import BIT_CANDIDATES, DEFAULT_GROUP_SIZE


def log(message: str) -> None:
    print("[{}] {}".format(time.strftime("%F %T"), message), flush=True)


def parse_devices(value: Optional[str]) -> List[str]:
    if not torch.cuda.is_available():
        raise RuntimeError("DELTA compression requires CUDA")
    if value is None or not value.strip():
        return ["cuda:{}".format(index) for index in range(torch.cuda.device_count())]
    devices = []
    for item in value.split(","):
        item = item.strip()
        if item:
            devices.append(item if item.startswith("cuda:") else "cuda:{}".format(int(item)))
    if not devices:
        raise ValueError("--devices did not contain a CUDA device")
    return devices


def validate_llama7b_config(config) -> None:
    expected = {
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
    }
    mismatches = {
        key: (getattr(config, key, None), expected_value)
        for key, expected_value in expected.items()
        if getattr(config, key, None) != expected_value
    }
    if mismatches:
        raise ValueError(
            "This release supports HuggyLLaMA LLaMA-7B only; config mismatches: {}".format(
                mismatches
            )
        )


def _load_base_model(model_path: str):
    local_only = Path(model_path).expanduser().exists()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=False,
        local_files_only=local_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
        torch_dtype=torch.float16,
        trust_remote_code=False,
        local_files_only=local_only,
    )
    validate_llama7b_config(model.config)
    return model.eval(), tokenizer


def _profile_for_layer(profiling: Dict, layer_index: int) -> Dict[str, torch.Tensor]:
    if layer_index in profiling:
        return profiling[layer_index]
    key = str(layer_index)
    if key in profiling:
        return profiling[key]
    raise KeyError("Profiling data is missing layer {}".format(layer_index))


@torch.no_grad()
def compress_one_layer(
    layer_index: int,
    layer,
    layer_profile: Dict[str, torch.Tensor],
    ratio: float,
    device: str,
) -> Tuple[int, torch.nn.Module, Dict[str, Dict]]:
    torch.cuda.set_device(torch.device(device))
    layer = layer.to(device)
    layer_choices = {}
    for projection_name in PROJECTION_NAMES:
        original = layer.get_submodule(projection_name)
        weight = original.weight.detach().float().to(device)
        cholesky = layer_profile[projection_name].to(device)
        u_factor, v_factor, choice = search_delta_factors(
            weight=weight,
            cholesky=cholesky,
            target_ratio=ratio,
            device=device,
            metadata_dtype=torch.float16,
        )
        replacement = DeltaLinear(
            u_factor=u_factor,
            v_factor=v_factor,
            in_features=original.in_features,
            out_features=original.out_features,
            bias=original.bias,
        )
        set_submodule(layer, projection_name, replacement)
        choice.update(
            {
                "layer": layer_index,
                "projection": projection_name,
                "in_features": original.in_features,
                "out_features": original.out_features,
            }
        )
        layer_choices[projection_name] = choice
        del weight, cholesky, original, replacement, u_factor, v_factor
        torch.cuda.empty_cache()
    return layer_index, layer.cpu(), layer_choices


@torch.no_grad()
def compress_model(
    model,
    profiling: Dict,
    ratio: float,
    devices: List[str],
    max_workers: Optional[int] = None,
    max_layers: Optional[int] = None,
) -> Dict[str, Dict]:
    layers = model.model.layers
    layer_count = len(layers) if max_layers is None else min(int(max_layers), len(layers))
    worker_count = len(devices) if max_workers is None else int(max_workers)
    worker_count = max(1, min(worker_count, len(devices), layer_count))
    choices = {}
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        for layer_index in range(layer_count):
            futures.append(
                executor.submit(
                    compress_one_layer,
                    layer_index,
                    layers[layer_index],
                    _profile_for_layer(profiling, layer_index),
                    ratio,
                    devices[layer_index % len(devices)],
                )
            )
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Compress LLaMA layers",
        ):
            layer_index, compressed_layer, layer_choices = future.result()
            layers[layer_index] = compressed_layer
            for projection_name, choice in layer_choices.items():
                choices["model.layers.{}.{}".format(layer_index, projection_name)] = choice
    return choices


def _choice_summary(choices: Dict[str, Dict]) -> str:
    pairs = Counter(
        (choice["u"]["nbits"], choice["v"]["nbits"])
        for choice in choices.values()
    )
    return ", ".join(
        "{}b/{}b={}".format(pair[0], pair[1], count)
        for pair, count in sorted(pairs.items())
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compress HuggyLLaMA LLaMA-7B with DELTA."
    )
    parser.add_argument("--model-path", required=True, help="Local LLaMA-7B directory or Hugging Face model ID.")
    parser.add_argument("--ratio", required=True, type=float, help="Target retained projection storage ratio in (0, 1].")
    parser.add_argument("--output-dir", required=True, help="New directory for the DELTA checkpoint.")
    parser.add_argument("--devices", default=None, help="Comma-separated logical CUDA ids; defaults to all visible GPUs.")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--profiling-path", default=None, help="Load profiling if it exists; otherwise compute and save it here.")
    parser.add_argument("--calib-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--wikitext-path", default=None)
    parser.add_argument("--max-layers", type=int, default=None, help="Smoke-test option: compress only the first N layers.")
    parser.add_argument("--max-shard-size", default="2GB")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.ratio <= 1.0:
        raise SystemExit("--ratio must be in (0, 1]")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit("--output-dir must be absent or empty: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    devices = parse_devices(args.devices)

    log("Loading LLaMA-7B from {}".format(args.model_path))
    model, tokenizer = _load_base_model(args.model_path)
    model.seqlen = args.seq_len
    profiling_path = None if args.profiling_path is None else Path(args.profiling_path).expanduser().resolve()
    if profiling_path is not None and profiling_path.exists():
        log("Loading Cholesky profiling from {}".format(profiling_path))
        profiling = torch.load(str(profiling_path), map_location="cpu", weights_only=True)
    else:
        log("Collecting Cholesky profiling with {} samples".format(args.calib_samples))
        calibration = get_calibration_data(
            tokenizer=tokenizer,
            nsamples=args.calib_samples,
            seqlen=args.seq_len,
            seed=args.seed,
            cache_dir=args.cache_dir,
            wikitext_path=args.wikitext_path,
        )
        profiling = profile_cholesky(
            model=model,
            calibration_batches=calibration,
            device=devices[0],
            max_layers=args.max_layers,
        )
        if profiling_path is not None:
            profiling_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(profiling, str(profiling_path))
            log("Saved profiling to {}".format(profiling_path))

    log(
        "Compressing ratio={} on devices={} with SVD-LLM Cholesky".format(
            args.ratio, ",".join(devices)
        )
    )
    choices = compress_model(
        model=model,
        profiling=profiling,
        ratio=args.ratio,
        devices=devices,
        max_workers=args.max_workers,
        max_layers=args.max_layers,
    )
    choices_payload = {
        "format": "delta-compression-choices-v1",
        "target_ratio": args.ratio,
        "bit_candidates": list(BIT_CANDIDATES),
        "group_size": DEFAULT_GROUP_SIZE,
        "layers": choices,
    }
    compression_config = {
        "method": "svdllm_cholesky",
        "target_ratio": args.ratio,
        "calibration_dataset": "wikitext2",
        "calibration_samples": args.calib_samples,
        "seed": args.seed,
        "sequence_length": args.seq_len,
        "group_size": DEFAULT_GROUP_SIZE,
        "bit_candidates": list(BIT_CANDIDATES),
        "metadata_dtype": "float16",
        "compressed_layers": len(choices) // len(PROJECTION_NAMES),
    }
    log("Selected bit pairs: {}".format(_choice_summary(choices)))
    save_delta_checkpoint(
        model=model,
        tokenizer=tokenizer,
        output_dir=str(output_dir),
        choices=choices_payload,
        compression_config=compression_config,
        max_shard_size=args.max_shard_size,
    )
    log("Saved {} checkpoint to {}".format(CHECKPOINT_FORMAT, output_dir))


if __name__ == "__main__":
    main()
