"""Small model components used by DELTA checkpoints."""

from typing import Any, Dict, Optional

import torch
from torch import nn

from .quantization import SINQLinearFactor


class DeltaLinear(nn.Module):
    """A packed low-rank projection that applies V followed by U."""

    def __init__(
        self,
        u_factor: SINQLinearFactor,
        v_factor: SINQLinearFactor,
        in_features: int,
        out_features: int,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.u_factor = u_factor
        self.v_factor = v_factor
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.u_factor(self.v_factor(inputs))
        if self.bias is not None:
            output = output + self.bias.to(device=output.device, dtype=output.dtype)
        return output

    def to_spec(self) -> Dict[str, Any]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "has_bias": self.bias is not None,
            "bias_dtype": None if self.bias is None else str(self.bias.dtype).replace("torch.", ""),
            "u_factor": self.u_factor.to_spec(),
            "v_factor": self.v_factor.to_spec(),
        }

    @classmethod
    def empty_from_spec(cls, spec: Dict[str, Any]) -> "DeltaLinear":
        bias = None
        if spec.get("has_bias"):
            dtype = getattr(torch, spec["bias_dtype"])
            bias = torch.empty(spec["out_features"], dtype=dtype)
        return cls(
            u_factor=SINQLinearFactor.empty_from_spec(spec["u_factor"]),
            v_factor=SINQLinearFactor.empty_from_spec(spec["v_factor"]),
            in_features=spec["in_features"],
            out_features=spec["out_features"],
            bias=bias,
        )


def set_submodule(root: nn.Module, path: str, module: nn.Module) -> None:
    parent_path, separator, child_name = path.rpartition(".")
    parent = root.get_submodule(parent_path) if separator else root
    setattr(parent, child_name, module)
