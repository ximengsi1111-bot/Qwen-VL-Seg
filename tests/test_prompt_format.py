from qwen3vl_seg.data.schema import Sample, SampleKind
from qwen3vl_seg.model.prompt_format import (
    MASK_END,
    MASK_START,
    MASK_TOKEN,
    build_target_records,
    build_target_text,
    build_user_prompt,
)


def _sample(kind: SampleKind, label: str, bboxes: list[list[float]]) -> Sample:
    return Sample(
        sample_id="sample-1",
        split="train",
        image_path="img.jpg",
        expression="white shirt" if kind is SampleKind.PHRASAL else label,
        kind=kind,
        label=label,
        bboxes=bboxes,
        mask_paths=[f"mask-{i}.png" for i in range(len(bboxes))],
        image_size=[480, 640],
    )


def test_category_prompt_and_records() -> None:
    sample = _sample(SampleKind.CATEGORY_SINGLE, "dog", [[10.0, 20.0, 310.0, 420.0]])
    prompt = build_user_prompt(sample)
    assert "dog" in prompt
    assert "JSON format" in prompt
    records = build_target_records(sample)
    assert records[0]["bbox_2d"] == [16, 42, 484, 875]
    assert records[0]["mask"] == f"{MASK_START}{MASK_TOKEN}{MASK_END}"


def test_phrasal_prompt_uses_expression() -> None:
    sample = _sample(SampleKind.PHRASAL, "person", [[0.0, 0.0, 100.0, 100.0]])
    prompt = build_user_prompt(sample)
    assert "white shirt" in prompt


def test_multi_target_text_is_json_array() -> None:
    sample = _sample(
        SampleKind.CATEGORY_MULTI,
        "person",
        [[0.0, 0.0, 320.0, 480.0], [320.0, 0.0, 640.0, 480.0]],
    )
    text = build_target_text(sample)
    assert text.startswith("[")
    assert text.endswith("]")
    assert text.count(MASK_START) == 2
