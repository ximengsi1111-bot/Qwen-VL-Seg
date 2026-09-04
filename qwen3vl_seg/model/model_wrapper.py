"""End-to-end wrapper around Qwen3-VL and the box-guided mask decoder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

from qwen3vl_seg.model.losses import segmentation_losses
from qwen3vl_seg.model.mask_decoder import BoxGuidedMaskDecoder, MaskDecoderConfig
from qwen3vl_seg.model.prompt_format import MASK_TOKEN, ensure_seg_special_tokens
from qwen3vl_seg.model.vision_hooks import ModelFeatureHooks, extract_per_image_features


@dataclass
class Qwen3VLSegConfig:
    model_path: str
    tokenizer_path: str | None = None
    decoder: MaskDecoderConfig = field(default_factory=MaskDecoderConfig)
    iou_loss_weight: float = 0.2
    ignore_mask_token_in_ce: bool = True


def _batch_row(value: Any, batch_index: int) -> Any:
    if torch.is_tensor(value) and value.ndim >= 2 and value.shape[0] == 1:
        return value[0]
    if isinstance(value, list):
        return value[batch_index]
    return value


class Qwen3VLSegForSegmentation(nn.Module):
    def __init__(
        self,
        config: Qwen3VLSegConfig,
        base_model: Qwen3VLForConditionalGeneration,
        processor: Any | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.base = base_model
        self.processor = processor or AutoProcessor.from_pretrained(
            config.tokenizer_path or config.model_path
        )
        self.tokenizer = self.processor.tokenizer
        self.special_token_ids = ensure_seg_special_tokens(self.tokenizer)
        old_vocab = self.base.config.text_config.vocab_size
        new_vocab = len(self.tokenizer)
        if new_vocab > old_vocab:
            self.base.resize_token_embeddings(new_vocab)
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(MASK_TOKEN)
        self.decoder = BoxGuidedMaskDecoder(config.decoder).to(self.base.dtype)
        self.decoder_dtype = next(self.decoder.parameters()).dtype

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        tokenizer_path: str | None = None,
        torch_dtype: torch.dtype | str | None = None,
        attn_implementation: str | None = None,
        **kwargs: Any,
    ) -> "Qwen3VLSegForSegmentation":
        processor_path = tokenizer_path or model_path
        processor = AutoProcessor.from_pretrained(processor_path)
        model_kwargs: dict[str, Any] = {"attn_implementation": attn_implementation}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        model_kwargs.update(kwargs)
        base = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        config = Qwen3VLSegConfig(model_path=model_path, tokenizer_path=processor_path)
        return cls(config=config, base_model=base, processor=processor)

    def forward(
        self,
        batch: dict[str, Any],
        labels: torch.Tensor | None = None,
        return_text_logits: bool = False,
        **base_kwargs: Any,
    ) -> dict[str, Any]:
        labels = labels if labels is not None else batch.get("labels")
        input_ids = batch["input_ids"].to(self.base.device)
        attention_mask = batch["attention_mask"].to(self.base.device)
        mm_token_type_ids = batch["mm_token_type_ids"].to(self.base.device)
        pixel_values = batch.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.base.device)
        image_grid_thw = batch.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(self.base.device)
        if labels is not None:
            labels = labels.to(self.base.device)

        with ModelFeatureHooks(self.base) as hooks:
            base_output = self.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                return_dict=True,
                use_cache=False,
                **base_kwargs,
            )

        hidden_states = hooks.text_last_hidden
        outputs: dict[str, Any] = {"text_loss": base_output.loss}
        if return_text_logits and base_output.logits is not None:
            outputs["text_logits"] = base_output.logits

        has_decoder_targets = bool(batch.get("has_seg", False))
        if (
            pixel_values is None
            or image_grid_thw is None
            or hidden_states is None
            or not has_decoder_targets
        ):
            if labels is not None:
                outputs["total_loss"] = outputs["text_loss"]
            return outputs

        per_image = extract_per_image_features(
            hooks,
            hidden_states,
            mm_token_type_ids,
            image_grid_thw,
        )
        image_rows = [
            b
            for b in range(mm_token_type_ids.shape[0])
            if bool((mm_token_type_ids[b] == 1).any())
        ]
        boxes = batch["boxes"]
        if boxes.ndim == 2:
            boxes = boxes.unsqueeze(0)
        masks = batch["masks"]
        if masks.ndim == 3:
            masks = masks.unsqueeze(0)
        boxes = boxes.to(self.base.device)
        masks = masks.to(self.base.device)
        num_instances = batch["num_instances"]
        if isinstance(num_instances, int):
            num_instances = [num_instances]

        decoder_outputs: list[dict[str, Any]] = []
        for image_index, row in enumerate(image_rows):
            count = int(num_instances[row])
            if count == 0:
                continue
            feature = per_image[image_index]
            mask_positions = (input_ids[row] == self.mask_token_id).nonzero().flatten()
            if mask_positions.numel() != count:
                raise RuntimeError(
                    f"row {row} has {mask_positions.numel()} mask tokens, expected {count}"
                )
            seg_features = hidden_states[row, mask_positions, :].unsqueeze(0)
            row_boxes = _batch_row(boxes, row)[:count].unsqueeze(0)
            row_masks = _batch_row(masks, row)[:count].unsqueeze(0)
            image_tensor = batch["image"]
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            decoder_input = {
                "visual_features": feature["visual"],
                "mm_features": feature["mm"].to(seg_features.dtype),
                "seg_features": seg_features.to(self.decoder_dtype),
                "boxes": row_boxes.to(self.decoder_dtype),
                "image": image_tensor.to(
                    device=self.base.device, dtype=self.decoder_dtype
                ),
            }
            decoded = self.decoder(**decoder_input)
            losses = segmentation_losses(
                decoded["mask_logits"],
                decoded["iou_scores"],
                row_masks.to(decoded["mask_logits"].device),
            )
            decoder_outputs.append(
                {
                    "mask_logits": decoded["mask_logits"],
                    "iou_scores": decoded["iou_scores"],
                    "gt_masks": row_masks,
                    "losses": losses,
                }
            )

        if decoder_outputs:
            seg_loss = torch.stack([entry["losses"]["seg_loss"] for entry in decoder_outputs]).mean()
            iou_loss = torch.stack([entry["losses"]["iou_mse_loss"] for entry in decoder_outputs]).mean()
            outputs["seg_loss"] = seg_loss
            outputs["iou_mse_loss"] = iou_loss
            outputs["mask_logits"] = torch.cat([entry["mask_logits"] for entry in decoder_outputs], dim=0)
            outputs["iou_scores"] = torch.cat([entry["iou_scores"] for entry in decoder_outputs], dim=0)
            if base_output.loss is not None:
                outputs["total_loss"] = (
                    base_output.loss + seg_loss + self.config.iou_loss_weight * iou_loss
                )
            else:
                outputs["total_loss"] = seg_loss + self.config.iou_loss_weight * iou_loss
        elif labels is not None:
            outputs["total_loss"] = outputs["text_loss"]
        return outputs
