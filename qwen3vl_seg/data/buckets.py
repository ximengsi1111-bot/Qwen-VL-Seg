"""Resolution bucket helpers for right/bottom padding batching."""

from __future__ import annotations

from PIL import Image


BUCKET_SIZES = (256, 384, 512, 640, 768)


def choose_bucket(
    height: int, width: int, bucket_sizes: tuple[int, ...] | list[int] | None = None
) -> int:
    """Smallest bucket >= max(H, W).

    ``bucket_sizes`` may be overridden (e.g. ``(256, 384, 512, 640)``) to
    control how much right/bottom padding is applied. Defaults to ``BUCKET_SIZES``.
    Images larger than the largest bucket use the largest bucket.
    """
    sizes = tuple(bucket_sizes) if bucket_sizes else BUCKET_SIZES
    largest = max(int(height), int(width))
    for size in sizes:
        if size >= largest:
            return size
    return sizes[-1]


def pad_right_bottom(image: Image.Image, bucket: int) -> tuple[Image.Image, int, int]:
    """Pad an RGB image on the right/bottom to a bucket-sized canvas."""
    width, height = image.size
    canvas = Image.new("RGB", (bucket, bucket), (0, 0, 0))
    canvas.paste(image, (0, 0))
    return canvas, bucket - width, bucket - height


def pad_mask_canvas(mask: Image.Image, bucket: int) -> Image.Image:
    """Place a grayscale mask at the top-left of a bucket-sized canvas."""
    width, height = mask.size
    canvas = Image.new("L", (bucket, bucket), 0)
    canvas.paste(mask, (0, 0))
    return canvas
