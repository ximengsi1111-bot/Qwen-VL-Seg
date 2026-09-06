"""Evaluate box IoU for the GRPO policy (with/without LoRA adapter)."""
import json, os, sys, torch
import numpy as np
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, "/mingli01/project/xiyongkai/qwen3vl-seg")
from qwen3vl_seg.model.prompt_format import build_user_prompt
from qwen3vl_seg.model.prompt_parser import parse_generated_boxes_and_masks

BASE = "/mingli01/data/xyk/model/grpo-policy"
ADAPTER = "/mingli01/data/xyk/grpo/smoke_out/v5-20260906-122205/checkpoint-64"
DATA = "/mingli01/data/xyk/grpo/rl_smoke.jsonl"
MAX_PIXELS = 1048576
DEVICE = "cuda"


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
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


def eval_iou(model, proc, samples, name):
    ious = []
    for s in samples:
        ns = SimpleNamespace(kind=s["kind"], label=s.get("label",""),
                             expression=s.get("expression",""),
                             image_size=s["image_size"], bboxes=s["bboxes"])
        user_text = build_user_prompt(ns)
        img = os.path.join("/mingli01/data/xyk", s["image_path"])
        messages = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":user_text}]}]
        enc = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True)
        enc = {k:(v.to(DEVICE) if torch.is_tensor(v) else v) for k,v in enc.items()}
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=96, do_sample=False)
        text = proc.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        preds = parse_generated_boxes_and_masks(text)
        sol = json.loads(s["solution"]); gt = sol["bbox_1000"]
        if preds and gt:
            best = max(box_iou(p["bbox_2d"], g) for p in preds for g in gt)
        else:
            best = 0.0
        ious.append(best)
    print(f"[{name}] n={len(ious)} mean_box_iou={np.mean(ious):.4f} raw={[round(x,3) for x in ious]}")


from types import SimpleNamespace
with open(DATA) as f:
    samples = [json.loads(l) for l in f if l.strip()]
# baseline (no adapter)
m0, p0 = load(None)
eval_iou(m0, p0, samples, "baseline(no GRPO)")
# with GRPO adapter
m1, p1 = load(ADAPTER)
eval_iou(m1, p1, samples, "grpo adapter")