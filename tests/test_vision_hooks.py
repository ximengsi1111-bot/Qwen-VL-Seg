from dataclasses import dataclass

import torch

from qwen3vl_seg.model.vision_hooks import extract_per_image_features


@dataclass
class _Hooks:
    visual_deepstack: list[torch.Tensor]
    visual_pooler: torch.Tensor
    text_last_hidden: torch.Tensor | None = None


def test_extract_per_image_features_2d_shapes() -> None:
    # Grid (1, 6, 8) yields 3x4 merged tokens per image.
    merge_size = 2
    merged_h, merged_w = 3, 4
    count = merged_h * merged_w
    grids = torch.tensor([[1, 6, 8], [1, 6, 8]])
    hidden_dim = 5
    pooler = torch.randn(2 * count, hidden_dim)
    deep1 = torch.randn(2 * count, hidden_dim)
    deep2 = torch.randn(2 * count, hidden_dim)
    deep3 = torch.randn(2 * count, hidden_dim)
    hooks = _Hooks([deep1, deep2, deep3], pooler)

    seq_len = 30
    hidden = torch.randn(2, seq_len, hidden_dim)
    mm = torch.zeros(2, seq_len, dtype=torch.long)
    mm[0, 10 : 10 + count] = 1
    mm[1, 5 : 5 + count] = 1
    features = extract_per_image_features(
        hooks,
        hidden,
        mm,
        grids,
        merge_size=merge_size,
    )
    assert len(features) == 2
    for feature in features:
        assert len(feature["visual"]) == 4
        for tensor in feature["visual"]:
            assert tensor.shape == (1, hidden_dim, merged_h, merged_w)
        assert feature["mm"].shape == (1, hidden_dim, merged_h, merged_w)
