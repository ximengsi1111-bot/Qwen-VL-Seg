import json, os, sys, torch
sys.path.insert(0, "/mingli01/project/xiyongkai/qwen3vl-seg")
from qwen3vl_seg.grpo.reward import SegIoUReward

with open("/mingli01/data/xyk/grpo/rl_smoke.jsonl") as f:
    s0 = json.loads(f.readline())
comp = s0["response"]          # gold target text (bbox_2d 0-1000 + mask placeholder)
solution = s0["solution"]
# messages: user (image+prompt) + assistant (completion)
user_text = s0["messages"][0]["content"][1]["text"]
msgs = [{
    "role": "user",
    "content": [{"type": "image", "image": s0["image_path"]}, {"type": "text", "text": user_text}],
}, {"role": "assistant", "content": comp}]

rw = SegIoUReward(box_weight=0.5, mask_weight=0.5)
# NOTE: solution/messages passed as lists (batched)
out = rw([comp], [solution], messages=[msgs])
print("reward(gold, box+mask):", out)
# also box-only view
rw2 = SegIoUReward(box_weight=1.0, mask_weight=0.0)
print("reward(gold, box only):", rw2([comp], [solution], messages=[msgs]))