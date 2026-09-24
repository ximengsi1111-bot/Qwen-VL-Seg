#!/usr/bin/env python3
"""Build GT / baseline / GRPO comparison sheets from runner visualizations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def _load_predictions(directory: Path) -> dict[str, dict]:
    rows = {}
    for path in sorted(directory.glob("predictions.rank*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows[row["sample_id"]] = row
    return rows


def _font(size: int, bold: bool = False):
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for root in [Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts")]:
        for name in names:
            path = root / name
            if path.exists():
                return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    tile = Image.new("RGB", size, "white")
    fitted = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    tile.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return tile


def _render_sample(
    sample: dict,
    baseline_pred: dict,
    final_pred: dict,
    baseline_viz: Path,
    final_viz: Path,
    panel_size: tuple[int, int],
    full_width: int,
) -> tuple[Image.Image, Image.Image]:
    baseline_img = Image.open(baseline_viz).convert("RGB")
    final_img = Image.open(final_viz).convert("RGB")
    bw, bh = baseline_img.size
    fw, fh = final_img.size
    baseline_pred_img = baseline_img.crop((0, 0, bw // 2, bh))
    baseline_gt_img = baseline_img.crop((bw // 2, 0, bw, bh))
    final_pred_img = final_img.crop((0, 0, fw // 2, fh))

    header_h = 72
    label_h = 34
    tile_w, tile_h = panel_size
    row_w = tile_w * 3
    row_h = header_h + label_h + tile_h
    row = Image.new("RGB", (row_w, row_h), "white")
    draw = ImageDraw.Draw(row)
    title_font = _font(22, bold=True)
    small_font = _font(16)
    label_font = _font(18, bold=True)

    expression = str(sample.get("expression", ""))
    b = float(baseline_pred.get("miou", 0.0))
    f = float(final_pred.get("miou", 0.0))
    title = f'"{expression}"   Baseline mIoU={b:.3f}   GRPO-2083 mIoU={f:.3f}   Delta={f-b:+.3f}'
    draw.text((12, 10), title, fill="black", font=title_font)
    draw.text((12, 42), sample["sample_id"], fill=(80, 80, 80), font=small_font)

    tiles = [
        (baseline_gt_img, "Ground truth"),
        (baseline_pred_img, "Baseline"),
        (final_pred_img, "checkpoint-2083"),
    ]
    for idx, (image, label) in enumerate(tiles):
        x = idx * tile_w
        draw.rectangle([x, header_h, x + tile_w - 1, header_h + label_h - 1], fill=(238, 238, 238))
        draw.text((x + 10, header_h + 7), label, fill="black", font=label_font)
        row.paste(_fit_panel(image, panel_size), (x, header_h + label_h))

    full_w = max(1, int(full_width))
    scale = full_w / row_w
    full = row.resize((full_w, max(1, int(row_h * scale))), Image.Resampling.LANCZOS)
    return row, full


def _sheet(panels: list[Image.Image], title: str, width: int) -> Image.Image:
    title_h = 54
    gap = 12
    scaled = []
    for panel in panels:
        scale = width / panel.width
        scaled.append(panel.resize((width, int(panel.height * scale)), Image.Resampling.LANCZOS))
    out = Image.new("RGB", (width, title_h + sum(p.height + gap for p in scaled)), "white")
    draw = ImageDraw.Draw(out)
    draw.text((16, 12), title, fill="black", font=_font(28, bold=True))
    y = title_h
    for panel in scaled:
        out.paste(panel, (0, y))
        y += panel.height + gap
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--final-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--panel-width", type=int, default=420)
    parser.add_argument("--panel-height", type=int, default=300)
    parser.add_argument("--full-width", type=int, default=1800)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    final_dir = Path(args.final_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = output_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    baseline_preds = _load_predictions(baseline_dir)
    final_preds = _load_predictions(final_dir)
    with Path(args.manifest).open(encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    panels = []
    full_panels = []
    for sample in samples:
        sid = sample["sample_id"]
        baseline_viz = baseline_dir / "viz" / f"{sid}_cmp.png"
        final_viz = final_dir / "viz" / f"{sid}_cmp.png"
        if not baseline_viz.is_file() or not final_viz.is_file():
            raise FileNotFoundError(f"missing visualization for {sid}")
        panel, full = _render_sample(
            sample,
            baseline_preds[sid],
            final_preds[sid],
            baseline_viz,
            final_viz,
            (args.panel_width, args.panel_height),
            args.full_width,
        )
        panels.append(panel)
        full_panels.append(full)
        full.save(individual_dir / f"{sid}.png")

    for idx in range(0, len(panels), 6):
        part = panels[idx:idx + 6]
        sheet = _sheet(part, f"PhraseCut comparison: GT / baseline / checkpoint-2083 ({idx+1}-{idx+len(part)})", 1320)
        sheet.save(output_dir / f"comparison_part{idx//6 + 1}.png")

    all_sheet = _sheet(panels, "PhraseCut comparison: GT / baseline / checkpoint-2083 (12 samples)", 1320)
    all_sheet.save(output_dir / "comparison_all.png")
    print(json.dumps({"samples": len(panels), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
