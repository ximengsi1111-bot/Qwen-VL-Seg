from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SampleKind(str, Enum):
    CATEGORY_SINGLE = "category_single"
    CATEGORY_MULTI = "category_multi"
    PHRASAL = "phrasal"
    DESCRIPTIVE = "descriptive"


@dataclass
class Sample:
    sample_id: str
    split: str
    image_path: str
    expression: str
    kind: SampleKind
    label: str
    bboxes: list[list[float]] = field(default_factory=list)
    mask_paths: list[str] = field(default_factory=list)
    image_size: list[int] = field(default_factory=list)
    source: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload


def parse_sample(row: dict[str, Any]) -> Sample:
    return Sample(
        sample_id=str(row["sample_id"]),
        split=str(row["split"]),
        image_path=str(row["image_path"]),
        expression=str(row["expression"]),
        kind=SampleKind(row["kind"]),
        label=str(row.get("label", "")),
        bboxes=[list(map(float, b)) for b in row.get("bboxes", [])],
        mask_paths=[str(p) for p in row.get("mask_paths", [])],
        image_size=[int(v) for v in row.get("image_size", [])],
        source=str(row.get("source", "")),
        created_at=str(row.get("created_at", "")),
    )

