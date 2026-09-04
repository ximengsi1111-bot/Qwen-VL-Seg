import numpy as np
import pytest

from qwen3vl_seg.eval.metrics import (
    aggregate_ciou,
    aggregate_instance_iou,
    bbox_iou,
    mask_iou,
    match_masks,
)


def test_mask_iou_identity() -> None:
    mask = np.ones((8, 8), dtype=np.uint8)
    assert mask_iou(mask, mask) == 1.0


def test_bbox_iou_cases() -> None:
    assert bbox_iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert bbox_iou([0, 0, 1, 1], [1, 0, 2, 1]) == 0.0
    assert bbox_iou([0, 0, 2, 2], [1, 1, 3, 3]) == 1.0 / 7.0


def test_hungarian_match() -> None:
    a = np.ones((4, 4), dtype=np.uint8)
    b = np.zeros((4, 4), dtype=np.uint8)
    pred_idx, gt_idx, total = match_masks([a, b], [b, a])
    assert sorted(pred_idx.tolist()) == [0, 1]
    assert sorted(gt_idx.tolist()) == [0, 1]
    assert total > 1.0


def test_aggregates() -> None:
    scores = aggregate_instance_iou([1.0, 0.5, 0.2])
    assert scores["miou"] == pytest.approx(0.5666666, abs=1e-5)
    assert scores["p@0.5"] == pytest.approx(2.0 / 3.0)
    assert aggregate_ciou([(np.ones((2, 2)), np.ones((2, 2)))]) == 1.0
