"""Convert the stage2-100k checkpoint base model into a standard ms-swift policy.

base.* (LoRA merged) -> AutoModelForImageTextToText, with seg mask tokens.
Base model already has vocab 151936 (== checkpoint embed rows); do NOT shrink.
"""
import torch
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoProcessor

SRC_DS = "/file_storage01/home/mingli/data/xyk/checkpoints/small/stage2-100k/ds"
BASE = "/mingli01/models/Qwen3-VL-4B-Instruct"
OUT = "/mingli01/data/xyk/model/grpo-policy"
SEG = ("<mask_start>", "<mask_token>", "<mask_end>")


def main():
    ds = Path(SRC_DS)
    latest = (ds / "latest").read_text().strip()
    ckpt = ds / latest / "mp_rank_00_model_states.pt"
    print("loading ckpt:", ckpt)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    mod = state.get("module", state)
    base_state = {k[5:]: v for k, v in mod.items() if k.startswith("base.")}
    print("base keys:", len(base_state))

    model = AutoModelForImageTextToText.from_pretrained(BASE, dtype="bfloat16")
    processor = AutoProcessor.from_pretrained(BASE)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        added = tokenizer.add_special_tokens({"additional_special_tokens": list(SEG)})
        print("added seg tokens ids:", added)

    embed_rows = model.get_input_embeddings().weight.shape[0]
    print("model embed rows:", embed_rows, "tokenizer len:", len(tokenizer))
    if tokenizer is not None and len(tokenizer) > embed_rows:
        model.resize_token_embeddings(len(tokenizer))
        print("resized UP to", len(tokenizer))

    missing, unexpected = model.load_state_dict(base_state, strict=False)
    print("missing:", len(missing), "unexpected:", len(unexpected))
    if unexpected:
        print("unexpected sample:", unexpected[:5])
    if missing:
        print("missing sample:", missing[:5])

    model.save_pretrained(OUT)
    processor.save_pretrained(OUT)
    print("saved:", OUT)


if __name__ == "__main__":
    main()