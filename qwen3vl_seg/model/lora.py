"""Minimal LoRA implementation for the Qwen3-VL text transformer."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Linear layer with a frozen pretrained weight plus trainable LoRA delta."""

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 32,
        alpha: int | float = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = float(alpha) / max(rank, 1)
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        for parameter in self.linear.parameters():
            parameter.requires_grad_(False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.linear(self.dropout(x), self.lora_A)
        return self.linear(x) + F.linear(hidden, self.lora_B) * self.scaling

    def merge(self) -> None:
        with torch.no_grad():
            self.linear.weight.add_(self.lora_B @ self.lora_A, alpha=self.scaling)
            self.lora_A.zero_()
            self.lora_B.zero_()


def apply_lora(
    root: nn.Module,
    target_attr_names: list[str],
    rank: int = 32,
    alpha: int | float = 32,
    dropout: float = 0.0,
) -> int:
    """Replace matching nn.Linear children with LoRALinear wrappers.

    Only direct children whose attribute name is in ``target_attr_names`` are
    wrapped, so vision blocks and decoder weights remain untouched.
    """
    replaced = 0

    def visit(module: nn.Module) -> None:
        nonlocal replaced
        for name, child in module.named_children():
            if name in target_attr_names and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                replaced += 1
            else:
                visit(child)

    visit(root)
    return replaced


def merge_lora(root: nn.Module) -> int:
    """Merge LoRA deltas into the wrapped linear and unwrap it."""
    merged = 0

    def visit(module: nn.Module) -> None:
        nonlocal merged
        for name, child in module.named_children():
            if isinstance(child, LoRALinear):
                child.merge()
                child.linear.weight.requires_grad_(True)
                if child.linear.bias is not None:
                    child.linear.bias.requires_grad_(True)
                setattr(module, name, child.linear)
                merged += 1
            else:
                visit(child)

    visit(root)
    return merged


def count_lora_parameters(root: nn.Module) -> dict[str, Any]:
    total = 0
    trainable = 0
    modules = 0
    for module in root.modules():
        if isinstance(module, LoRALinear):
            modules += 1
            for parameter in module.parameters():
                total += parameter.numel()
                if parameter.requires_grad:
                    trainable += parameter.numel()
    return {"modules": modules, "parameters": total, "trainable": trainable}
