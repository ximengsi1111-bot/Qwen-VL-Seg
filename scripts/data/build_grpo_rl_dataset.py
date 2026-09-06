"""Convert the GRPO prompt pool into ms-swift RLHF dataset format.

Rows carry messages (image+prompt), response (gold target text), a
rejected_response placeholder (satisfies ms-swift RLHF template), solution
(GT json for the reward) plus custom columns for reward kwargs.
"""
import json
import os
from types import SimpleNamespace

from qwen3vl_seg.model.prompt_format import build_user_prompt, build_target_text, _scale_box

SRC = "/mingli01/data/xyk/grpo/prompts.jsonl"
DATA_ROOT = "/mingli01/data/xyk"
OUT = "/mingli01/data/xyk/grpo/rl_dataset.jsonl"


def _ns(d):
    return SimpleNamespace(
        kind=d["kind"],
        label=d.get("label", ""),
        expression=d.get("expression", ""),
        image_size=d["image_size"],
        bboxes=d["bboxes"],
    )


def main():
    rows = []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            ns = _ns(d)
            user_text = build_user_prompt(ns)
            target_text = build_target_text(ns)
            image_abs = os.path.join(DATA_ROOT, d["image_path"])
            masks_abs = [os.path.join(DATA_ROOT, m) for m in d["mask_paths"]]
            h, w = (int(v) for v in d["image_size"])
            bbox_1000 = [_scale_box(list(b), w, h) for b in d["bboxes"]]
            rows.append({
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_abs},
                        {"type": "text", "text": user_text},
                    ],
                }],
                "response": target_text,
                "solution": json.dumps({
                    "bbox_1000": bbox_1000,
                    "image_path": image_abs,
                    "mask_paths": masks_abs,
                    "image_size": [h, w],
                }),
                "image_path": image_abs,
                "mask_paths": masks_abs,
                "bboxes": d["bboxes"],
                "image_size": [h, w],
                "source": d["source"],
                "kind": d["kind"],
            })
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("rows:", len(rows), "->", OUT)


if __name__ == "__main__":
    main()