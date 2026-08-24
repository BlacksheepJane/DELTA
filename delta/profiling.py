"""Collect SVD-LLM Cholesky profiling matrices for LLaMA blocks."""

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from tqdm import tqdm


PROJECTION_NAMES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


class _CaptureComplete(Exception):
    pass


def _move_entry_modules(model, device: str) -> None:
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.norm = model.model.norm.to(device)


def _release_entry_modules(model) -> None:
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()


def _capture_first_layer_inputs(
    model,
    calibration_batches: List[dict],
    device: str,
) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
    layers = model.model.layers
    _move_entry_modules(model, device)
    layers[0] = layers[0].to(device)
    inputs = []
    attention_masks = []
    position_ids = []

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self.module = module

        def forward(self, hidden_states, *args, **kwargs):
            inputs.append(hidden_states.detach().cpu())
            attention_mask = kwargs.get("attention_mask")
            position_id = kwargs.get("position_ids")
            attention_masks.append(
                None if attention_mask is None else attention_mask.detach().cpu()
            )
            position_ids.append(
                None if position_id is None else position_id.detach().cpu()
            )
            raise _CaptureComplete

    original_first_layer = layers[0]
    layers[0] = Catcher(original_first_layer)
    try:
        for batch in tqdm(calibration_batches, desc="Capture calibration inputs"):
            try:
                model(**{key: value.to(device) for key, value in batch.items()})
            except _CaptureComplete:
                pass
    finally:
        layers[0] = original_first_layer.cpu()
        _release_entry_modules(model)
        torch.cuda.empty_cache()
    return inputs, attention_masks, position_ids


def _positive_cholesky(covariance: torch.Tensor, device: str) -> torch.Tensor:
    covariance = covariance.double().to(device)
    try:
        return torch.linalg.cholesky(covariance).cpu()
    except RuntimeError:
        minimum = torch.linalg.eigvalsh(covariance)[0]
        jitter = (-minimum + 1e-3) * torch.eye(
            covariance.shape[0], device=device, dtype=covariance.dtype
        )
        return torch.linalg.cholesky(covariance + jitter).cpu()


@torch.no_grad()
def profile_cholesky(
    model,
    calibration_batches: List[dict],
    device: str = "cuda:0",
    max_layers: Optional[int] = None,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Collect one Cholesky factor per LLaMA projection input covariance."""
    if not calibration_batches:
        raise ValueError("calibration_batches must not be empty")
    layers = model.model.layers
    layer_count = len(layers) if max_layers is None else min(int(max_layers), len(layers))
    inputs, attention_masks, position_ids = _capture_first_layer_inputs(
        model, calibration_batches, device
    )
    profiling = {}

    for layer_index in tqdm(range(layer_count), desc="Profile LLaMA layers"):
        layer = layers[layer_index].to(device)
        outputs = []
        handles = []
        projection_modules = {
            name: layer.get_submodule(name) for name in PROJECTION_NAMES
        }

        def accumulate(module, module_inputs, _output):
            activations = module_inputs[0].detach().float()
            if activations.dim() == 2:
                activations = activations.unsqueeze(0)
            module._delta_covariance.add_(
                torch.matmul(activations.transpose(1, 2), activations).sum(dim=0)
            )

        for module in projection_modules.values():
            module._delta_covariance = torch.zeros(
                module.in_features,
                module.in_features,
                dtype=torch.float32,
                device=device,
            )
            handles.append(module.register_forward_hook(accumulate))

        for sample_index, hidden_states in enumerate(inputs):
            kwargs = {}
            if attention_masks[sample_index] is not None:
                kwargs["attention_mask"] = attention_masks[sample_index].to(device)
            if position_ids[sample_index] is not None:
                kwargs["position_ids"] = position_ids[sample_index].to(device)
            outputs.append(layer(hidden_states.to(device), **kwargs)[0].detach().cpu())

        for handle in handles:
            handle.remove()
        layer_profile = {}
        for name, module in projection_modules.items():
            layer_profile[name] = _positive_cholesky(module._delta_covariance, device)
            del module._delta_covariance
        profiling[layer_index] = layer_profile
        layers[layer_index] = layer.cpu()
        inputs = outputs
        torch.cuda.empty_cache()
    return profiling
