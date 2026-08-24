"""Self-contained safetensors checkpoint I/O for DELTA models."""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import load_sharded_checkpoint, no_init_weights

from .modeling import DeltaLinear, set_submodule


CHECKPOINT_FORMAT = "delta-safetensors-v1"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def collect_delta_module_specs(model) -> Dict[str, Dict[str, Any]]:
    return {
        name: module.to_spec()
        for name, module in model.named_modules()
        if isinstance(module, DeltaLinear)
    }


def save_delta_checkpoint(
    model,
    tokenizer,
    output_dir: str,
    choices: Dict[str, Any],
    compression_config: Dict[str, Any],
    max_shard_size: str = "2GB",
) -> None:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    module_specs = collect_delta_module_specs(model)
    if not module_specs:
        raise ValueError("Refusing to save a checkpoint without DeltaLinear modules")

    delta_config = {
        "format": CHECKPOINT_FORMAT,
        "compression": compression_config,
        "modules": module_specs,
    }
    _write_json(output_path / "delta_config.json", delta_config)
    _write_json(output_path / "compression_choices.json", choices)
    model.save_pretrained(
        str(output_path),
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    tokenizer.save_pretrained(str(output_path))


def _load_safe_weights(model, checkpoint_path: Path) -> None:
    single_file = checkpoint_path / "model.safetensors"
    index_file = checkpoint_path / "model.safetensors.index.json"
    if single_file.exists():
        state_dict = load_file(str(single_file), device="cpu")
        incompatible = model.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("Unexpected checkpoint key mismatch: {}".format(incompatible))
        return
    if index_file.exists():
        load_sharded_checkpoint(
            model,
            str(checkpoint_path),
            strict=True,
            prefer_safe=True,
        )
        return
    raise FileNotFoundError("No safetensors weights found in {}".format(checkpoint_path))


def load_delta_checkpoint(checkpoint_dir: str) -> Tuple[Any, Any]:
    checkpoint_path = Path(checkpoint_dir).expanduser().resolve()
    delta_config_path = checkpoint_path / "delta_config.json"
    if not delta_config_path.exists():
        raise FileNotFoundError(
            "Not a DELTA directory checkpoint: {} is missing".format(delta_config_path)
        )
    delta_config = json.loads(delta_config_path.read_text())
    if delta_config.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported DELTA checkpoint format: {}".format(delta_config.get("format")))

    config = AutoConfig.from_pretrained(str(checkpoint_path), local_files_only=True)
    if config.model_type != "llama":
        raise ValueError("DELTA checkpoint must contain a LLaMA config")
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=getattr(config, "torch_dtype", torch.float16),
        )
    for module_path, module_spec in delta_config["modules"].items():
        set_submodule(model, module_path, DeltaLinear.empty_from_spec(module_spec))
    _load_safe_weights(model, checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_path), trust_remote_code=False, local_files_only=True
    )
    return model, tokenizer
