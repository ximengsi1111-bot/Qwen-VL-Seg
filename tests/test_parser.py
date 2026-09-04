from qwen3vl_seg.model.prompt_parser import parse_generated_boxes_and_masks


def test_parse_single_record() -> None:
    text = '```json\n[{"bbox_2d": [1, 2, 30, 40], "label": "dog", "mask": "<mask_start><mask_token><mask_end>"}]\n```'
    records = parse_generated_boxes_and_masks(text)
    assert len(records) == 1
    assert records[0]["label"] == "dog"
    assert records[0]["bbox_2d"] == [1.0, 2.0, 30.0, 40.0]


def test_parse_invalid_returns_empty() -> None:
    assert parse_generated_boxes_and_masks("I cannot segment this image.") == []
    assert parse_generated_boxes_and_masks("") == []

