import json, os, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageEnhance
from types import SimpleNamespace
sys.path.insert(0, "/mingli01/project/xiyongkai/qwen3vl-seg")
from qwen3vl_seg.model.prompt_format import build_user_prompt, build_target_text, _scale_box

ROOT = "/mingli01/data/xyk"
SRC = ROOT + "/grpo/hard_ids_02_05.jsonl"
AUG = ROOT + "/grpo/aug"
OUT = ROOT + "/grpo/rl_hard.jsonl"
WORKERS = int(os.environ.get("WORKERS", "16"))
LIMIT = int(os.environ.get("LIMIT", "0"))


def _ns(d, bboxes):
    return SimpleNamespace(kind=d["kind"], label=d.get("label", ""),
                           expression=d.get("expression", ""),
                           image_size=d["image_size"], bboxes=bboxes)


def emit(rows, d, img_abs, masks_abs, bboxes):
    ns = _ns(d, bboxes)
    user_text = build_user_prompt(ns)
    target_text = build_target_text(ns)
    h, w = (int(v) for v in d["image_size"])
    bbox_1000 = [_scale_box(list(b), w, h) for b in bboxes]
    rows.append({
        "messages": [{"role": "user", "content": [
            {"type": "image", "image": img_abs}, {"type": "text", "text": user_text}]}],
        "response": target_text,
        "solution": json.dumps({"bbox_1000": bbox_1000, "image_path": img_abs,
                                "mask_paths": masks_abs, "image_size": [h, w]}),
        "image_path": img_abs, "mask_paths": masks_abs, "bboxes": bboxes,
        "image_size": [h, w], "source": d["source"], "kind": d["kind"],
    })


def process_sample(d):
    rows = []
    img = Image.open(os.path.join(ROOT, d["image_path"])).convert("RGB")
    W, H = img.size
    masks_abs = [os.path.join(ROOT, m) for m in d["mask_paths"]]
    bboxes = d["bboxes"]
    sid = d["sample_id"]
    # hflip (image + masks + bboxes)
    fimg = img.transpose(Image.FLIP_LEFT_RIGHT)
    fp = AUG + f"/images/{sid}_hflip.jpg"; fimg.save(fp, quality=95)
    fmasks = []
    for j, mp in enumerate(d["mask_paths"]):
        m = Image.open(os.path.join(ROOT, mp)).convert("L").transpose(Image.FLIP_LEFT_RIGHT)
        fmp = AUG + f"/masks/{sid}_hflip_{j}.png"; m.save(fmp); fmasks.append(fmp)
    fb = [[W - b[2], b[1], W - b[0], b[3]] for b in bboxes]
    emit(rows, d, fp, fmasks, fb)
    # color jitter (image only)
    cimg = ImageEnhance.Color(ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(1.2)).enhance(1.1)).enhance(1.2)
    cp = AUG + f"/images/{sid}_color.jpg"; cimg.save(cp, quality=95)
    emit(rows, d, cp, masks_abs, bboxes)
    # gaussian noise (image only)
    arr = np.clip(np.asarray(img).astype(np.float32) + np.random.normal(0, 8, (H, W, 3)), 0, 255).astype(np.uint8)
    npth = AUG + f"/images/{sid}_noise.jpg"; Image.fromarray(arr).save(npth, quality=95)
    emit(rows, d, npth, masks_abs, bboxes)
    return rows


def main():
    os.makedirs(AUG + "/images", exist_ok=True)
    os.makedirs(AUG + "/masks", exist_ok=True)
    samples = [json.loads(l) for l in open(SRC) if l.strip()]
    if LIMIT:
        samples = samples[:LIMIT]
    print(f"samples: {len(samples)} workers: {WORKERS}", flush=True)
    rows = []
    done = 0
    with Pool(WORKERS) as pool:
        for r in pool.imap_unordered(process_sample, samples, chunksize=8):
            rows.extend(r)
            done += 1
            if done % 2000 == 0:
                print(f"{done} done", flush=True)
    with open(OUT, "w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("RL rows:", len(rows), "->", OUT)


if __name__ == "__main__":
    main()
