"""Hooks that expose intermediate Qwen3-VL features to the mask decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ModelFeatureHooks:
    """Context manager that captures visual deep-stack and final text states."""

    base: Any
    visual_deepstack: list[torch.Tensor] | None = None
    visual_pooler: torch.Tensor | None = None
    text_last_hidden: torch.Tensor | None = None
    _handles: list[Any] | None = None

    def __enter__(self) -> "ModelFeatureHooks":
        self._handles = [
            self.base.model.visual.register_forward_hook(self._on_visual),
            self.base.model.language_model.register_forward_hook(self._on_text),
        ]
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in self._handles or []:
            handle.remove()
        self._handles = []

    def _on_visual(self, module: Any, args: tuple[Any, ...], output: Any) -> None:
        self.visual_pooler = getattr(output, "pooler_output", None)
        self.visual_deepstack = list(getattr(output, "deepstack_features", None) or [])

    def _on_text(self, module: Any, args: tuple[Any, ...], output: Any) -> None:
        self.text_last_hidden = getattr(output, "last_hidden_state", None)


def _image_rows(mm_token_type_ids: torch.Tensor) -> list[int]:
    rows: list[int] = []
    for batch_index in range(mm_token_type_ids.shape[0]):
        if bool((mm_token_type_ids[batch_index] == 1).any()):
            rows.append(batch_index)
    return rows


def extract_per_image_features(
    hooks: ModelFeatureHooks,
    hidden_states: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    merge_size: int = 2,
) -> list[dict[str, torch.Tensor]]:
    """Return visual/multimodal features grouped per input image.

    Each returned entry contains:
      - visual: list of 4 tensors with shape (1, C, H, W)
      - mm: tensor with shape (1, C, H, W)
      - grid: [H, W] of the merged LLM visual grid
    """
    if image_grid_thw is None or image_grid_thw.numel() == 0:
        return []
    grids = image_grid_thw.tolist()
    counts = [t * h * w // (merge_size * merge_size) for t, h, w in grids]
    scales = [*hooks.visual_deepstack, hooks.visual_pooler]
    scales = [scale for scale in scales if scale is not None]

    offset = 0
    per_scale_slices = []
    for count in counts:
        chunk = []
        for scale in scales:
            slice_ = scale[offset : offset + count]
            t, h, w = grids[len(per_scale_slices)]
            merged_h, merged_w = h // merge_size, w // merge_size
            chunk.append(
                slice_.view(merged_h, merged_w, -1).permute(2, 0, 1).unsqueeze(0)
            )
        per_scale_slices.append(chunk)
        offset += count

    rows = _image_rows(mm_token_type_ids)
    if len(rows) != len(per_scale_slices):
        raise RuntimeError(
            f"image rows {len(rows)} do not match image grids {len(per_scale_slices)}"
        )

    results = []
    for grid_index, batch_index in enumerate(rows):
        positions = (mm_token_type_ids[batch_index] == 1).nonzero().flatten()
        if positions.numel() != counts[grid_index]:
            raise RuntimeError(
                f"row {batch_index} has {positions.numel()} visual token positions, "
                f"expected {counts[grid_index]}"
            )
        hidden = hidden_states[batch_index, positions, :]
        t, h, w = grids[grid_index]
        merged_h, merged_w = h // merge_size, w // merge_size
        mm = (
            hidden.view(merged_h, merged_w, -1)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .contiguous()
        )
        results.append({"visual": per_scale_slices[grid_index], "mm": mm, "grid": [h, w]})
    return results
