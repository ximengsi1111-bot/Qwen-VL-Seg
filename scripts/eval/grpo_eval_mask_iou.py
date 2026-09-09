import os, sys, json, torch
import numpy as np
from types import SimpleNamespace
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, "/mingli01/project/xiyongkai/qwen3vl-seg")
from qwen3vl_seg.model.prompt_format import build_user_prompt
from qwen3vl_seg.model.prompt_parser import parse_generated_boxes_and_masks
from qwen3vl_seg.grpo.reward import SegIoUReward

BASE = "/mingli01/data/xyk/model/grpo-policy"
ADAPTERS = {
    "baseline(no GRPO)": None,
    "grpo adapter checkpoint-500": "/mingli01/data/xyk/grpo/train_out/v0-20260908-124037/checkpoint-500",
    "grpo adapter checkpoint-1000": "/mingli01/data/xyk/grpo/train_out/v0-20260908-124037/checkpoint-1000",
}
DATA = "/mingli01/data/xyk/grpo/rl_eval.jsonl"
DEVICE = "cuda"


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (ua + ub - inter) if (ua + ub - inter) > 0 else 0.0


def load(adapter):
    model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.to(DEVICE).eval()
    proc = AutoProcessor.from_pretrained(BASE)
    return model, proc


def generate(model, proc, s):
    ns = SimpleNamespace(kind=s["kind"], label=s.get("label", ""),
                         expression=s.get("expression", ""),
                         image_size=s["image_size"], bboxes=s["bboxes"])
    user_text = build_user_prompt(ns)
    img = os.path.join("/mingli01/data/xyk", s["image_path"])
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img}, {"type": "text", "text": user_text}]}]
    enc = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                   return_tensors="pt", return_dict=True)
    enc = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=96, do_sample=False)
    text = proc.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return text


with open(DATA) as f:
    samples = [json.loads(l) for l in f if l.strip()]

rw = SegIoUReward(box_weight=1.0, mask_weight=0.0)  # wrapper is loaded lazily; weights irrelevant here

for name, adapter in ADAPTERS.items():
    model, proc = load(adapter)
    box_ious, mask_ious, mask_fail = [], [], 0
    for s in samples:
        text = generate(model, proc, s)
        preds = parse_generated_boxes_and_masks(text)
        gt = json.loads(s["solution"])
        # box iou
        best = 0.0
        if preds and gt.get("bbox_1000"):
            best = max(box_iou(p["bbox_2d"], g) for p in preds for g in gt["bbox_1000"])
        box_ious.append(best)
        # mask iou (frozen stage2 decoder, consistent with GRPO reward)
        msgs = list(s["messages"]) + [{"role": "assistant", "content": text}]
        try:
            mi = rw._mask_iou(text, gt, msgs)
        except Exception as e:
            mi = 0.0
            mask_fail += 1
        mask_ious.append(mi)
    print(f"[{name}] n={len(box_ious)} mean_box_iou={np.mean(box_ious):.4f} "
          f"mean_mask_iou={np.mean(mask_ious):.4f} mask_fail={mask_fail}")
    del model, proc
    torch.cuda.empty_cache()
