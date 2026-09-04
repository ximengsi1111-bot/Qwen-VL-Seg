from qwen3vl_seg.data.schema import Sample, SampleKind, parse_sample


def test_sample_roundtrip() -> None:
    raw = {
        "sample_id": "refcoco_train_000123",
        "split": "train",
        "image_path": "datasets/raw/coco2017/train2017/0001.jpg",
        "expression": "the dog on the left",
        "kind": "phrasal",
        "label": "dog",
        "bboxes": [[1.0, 2.0, 30.0, 40.0]],
        "mask_paths": ["datasets/small/public/masks/0001.png"],
        "image_size": [480, 640],
        "source": "refcoco:train",
        "created_at": "2026-09-02",
    }
    sample = parse_sample(raw)
    assert isinstance(sample, Sample)
    assert sample.kind is SampleKind.PHRASAL
    assert sample.to_dict()["kind"] == "phrasal"

