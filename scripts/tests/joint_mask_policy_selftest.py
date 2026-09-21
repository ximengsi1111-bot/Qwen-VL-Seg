"""CPU self-test for the joint token+mask GRPO policy.

Covers the two halves that meet in :class:`OnPolicySegGRPOTrainer`:

1. ``capture_mask_actions_from_policy_forward`` -- sample a mask action from a
   (no_grad) policy forward and score it against the GT mask.
2. ``joint_mask_policy_loss`` -- re-score the sampled action on a grad-enabled
   forward and check the gradient reaches the decoder **and** the Qwen3-VL
   hidden states.

Run from the repository root (CPU only, no checkpoints required)::

    PYTHONPATH=. python scripts/tests/joint_mask_policy_selftest.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

from qwen3vl_seg.grpo.joint_mask_policy import (
    MaskActionRecord,
    capture_mask_actions_from_policy_forward,
    joint_mask_policy_loss,
    load_gt_masks,
    mask_logprob_from_logits,
    resolve_mask_decoder_dtype,
)
from qwen3vl_seg.model.mask_decoder import BoxGuidedMaskDecoder, MaskDecoderConfig
from qwen3vl_seg.model.prompt_format import MASK_TOKEN
from qwen3vl_seg.model.vision_hooks import ModelFeatureHooks

HIDDEN = 2560
GRID_H, GRID_W = 6, 8
MERGED = (GRID_H // 2, GRID_W // 2)
SEQ_LEN = 20
MM_POSITIONS = list(range(2, 14))
MASK_POSITIONS = [15, 17]
MASK_TOKEN_ID = 7


class _FakeTokenizer:
    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == MASK_TOKEN, token
        return MASK_TOKEN_ID


class _FakeSample:
    def __init__(self, completion: str, solution: dict) -> None:
        self.messages = [{"role": "assistant", "content": completion}]
        self.extra = {"solution": json.dumps(solution)}


def _fixtures(tmpdir: str):
    image_path = os.path.join(tmpdir, "image.png")
    image = np.zeros((GRID_H * 16, GRID_W * 16, 3), dtype=np.uint8)
    image[40:100, 30:90] = 200
    Image.fromarray(image).save(image_path)

    mask_paths = []
    for index in range(2):
        mask = np.zeros((GRID_H * 16, GRID_W * 16), dtype=np.uint8)
        mask[20 + index * 10 : 80 + index * 10, 20 + index * 20 : 90 + index * 20] = 255
        path = os.path.join(tmpdir, f"mask{index}.png")
        Image.fromarray(mask, mode="L").save(path)
        mask_paths.append(path)

    solution = {
        "bbox_1000": [[100, 200, 300, 400], [150, 250, 350, 450]],
        "image_path": image_path,
        "mask_paths": mask_paths,
        "image_size": [GRID_H * 16, GRID_W * 16],
    }
    completion = json.dumps(
        [
            {"bbox_2d": [100, 200, 300, 400], "label": "x", "mask": "<mask_start><mask_token><mask_end>"},
            {"bbox_2d": [150, 250, 350, 450], "label": "x", "mask": "<mask_start><mask_token><mask_end>"},
        ]
    )
    return _FakeSample(completion, solution)


def _model_inputs():
    input_ids = torch.zeros(1, SEQ_LEN, dtype=torch.long)
    input_ids[0, MASK_POSITIONS] = MASK_TOKEN_ID
    mm_token_type_ids = torch.zeros(1, SEQ_LEN, dtype=torch.long)
    mm_token_type_ids[0, MM_POSITIONS] = 1
    return {
        "input_ids": input_ids,
        "mm_token_type_ids": mm_token_type_ids,
        "image_grid_thw": torch.tensor([[1, GRID_H, GRID_W]]),
    }


def _hooks(requires_grad: bool):
    hooks = ModelFeatureHooks(base=None)
    count = MERGED[0] * MERGED[1]
    scales = [torch.randn(count, HIDDEN, requires_grad=requires_grad) for _ in range(4)]
    hooks.visual_deepstack = scales[:3]
    hooks.visual_pooler = scales[3]
    hooks.text_last_hidden = torch.randn(1, SEQ_LEN, HIDDEN, requires_grad=requires_grad)
    return hooks


def _check_logprob() -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, 7, 9)
    target = (torch.rand(3, 7, 9) > 0.5).float()
    prob = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    manual = (target * torch.log(prob) + (1 - target) * torch.log(1 - prob))
    assert torch.allclose(
        mask_logprob_from_logits(logits, target, "mean"), manual.flatten(1).mean(1).mean(), atol=1e-6
    )
    assert torch.allclose(
        mask_logprob_from_logits(logits, target, "sum"), manual.flatten(1).sum(1).mean(), atol=1e-5
    )
    assert resolve_mask_decoder_dtype(None) is torch.float32
    assert resolve_mask_decoder_dtype("bf16") is torch.bfloat16
    print("[ok] bernoulli log-prob + dtype mapping")


def _check_gt_binarisation(tmpdir: str) -> None:
    path = os.path.join(tmpdir, "gt.png")
    array = np.zeros((17, 19), dtype=np.uint8)
    array[5:12, 3:15] = 255
    array[2:4, 2:4] = 128
    Image.fromarray(array, mode="L").save(path)
    masks = load_gt_masks({"mask_paths": [path]}, 6, 8)
    assert masks[0].shape == (6, 8) and set(np.unique(masks[0]).tolist()) <= {0, 1}
    reference = Image.open(path).convert("L").resize((8, 6), Image.BILINEAR)
    reference = (np.asarray(reference, dtype=np.float32) > 127.5).astype(np.uint8)
    assert np.array_equal(masks[0], reference)
    print("[ok] GT mask binarised at decoder resolution (SFT-identical)")


def _run_end_to_end(tmpdir: str) -> None:
    torch.manual_seed(0)
    os.environ.setdefault("GRPO_MASK_POLICY_DTYPE", "float32")
    decoder = BoxGuidedMaskDecoder(MaskDecoderConfig()).to(torch.float32)
    decoder.requires_grad_(True)
    sample = _fixtures(tmpdir)
    model_inputs = _model_inputs()

    # ---- 1. behaviour policy: sample the mask action (no_grad) -------------
    # Same feature tensors for both halves: in the trainer the behaviour policy
    # and the current policy see the same forward, so the ratio must stay ~1.
    hooks = _hooks(requires_grad=True)
    with torch.no_grad():
        records, stats = capture_mask_actions_from_policy_forward(
            decoder=decoder,
            tokenizer=_FakeTokenizer(),
            hooks=hooks,
            model_inputs=model_inputs,
            samples=[sample],
            device=torch.device("cpu"),
            sample_masks=True,
            logprob_reduction="mean",
            reward_mode="deterministic",
        )
    record = records[0]
    assert record.valid, record.failure
    assert record.sampled_mask.shape == (2, MERGED[0] * 4, MERGED[1] * 4), record.sampled_mask.shape
    assert 0.0 <= record.mask_iou <= 1.0 and 0.0 <= record.deterministic_mask_iou <= 1.0
    assert stats[0]["mask_iou"] == record.deterministic_mask_iou
    assert stats[0]["sampled_mask_iou"] == record.mask_iou
    assert stats[0]["decode_fail"] == 0.0
    record.advantage = 0.75
    print(
        f"[ok] sampled mask action: shape={tuple(record.sampled_mask.shape)} "
        f"sampled_iou={record.mask_iou:.4f} det_iou={record.deterministic_mask_iou:.4f} "
        f"old_logp={record.old_mask_logp:.4f}"
    )

    # ---- 2. current policy: re-score the action with gradients -------------
    for parameter in decoder.parameters():
        parameter.grad = None
    loss, metrics = joint_mask_policy_loss(
        decoder=decoder,
        hooks=hooks,
        model_inputs=model_inputs,
        records=records,
        device=torch.device("cpu"),
        logprob_reduction="mean",
        clip_epsilon=0.2,
    )
    assert loss is not None and torch.isfinite(loss), loss
    assert metrics["joint_masks/grad_to_llm"] == 1.0
    assert abs(metrics["joint_masks/ratio_mean"] - 1.0) < 1e-4, metrics["joint_masks/ratio_mean"]
    loss.backward()

    decoder_grad = sum(
        float(parameter.grad.abs().sum()) for parameter in decoder.parameters() if parameter.grad is not None
    )
    hidden_grad = float(hooks.text_last_hidden.grad.abs().sum())
    visual_grad = sum(float(scale.grad.abs().sum()) for scale in hooks.visual_deepstack)
    assert decoder_grad > 0, decoder_grad
    assert hidden_grad > 0, hidden_grad
    assert visual_grad > 0, visual_grad
    print(
        f"[ok] joint loss={loss.item():.6f} ratio={metrics['joint_masks/ratio_mean']:.6f} "
        f"grad_decoder={decoder_grad:.3f} grad_hidden={hidden_grad:.3f} grad_visual={visual_grad:.3f}"
    )

    # ---- 3. degenerate cases are skipped, not crashed ---------------------
    empty = MaskActionRecord(row=0, image_index=0, failure="parse_fail")
    skipped, _ = joint_mask_policy_loss(
        decoder=decoder,
        hooks=hooks,
        model_inputs=model_inputs,
        records=[empty],
        device=torch.device("cpu"),
    )
    assert skipped is None
    print("[ok] failed samples are skipped instead of crashing")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        _check_logprob()
        _check_gt_binarisation(tmpdir)
        _run_end_to_end(tmpdir)
    print("ALL JOINT MASK POLICY SELF-TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())