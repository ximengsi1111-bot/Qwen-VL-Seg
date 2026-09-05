"""Bucket-aware sampler that keeps each batch within one resolution bucket."""

from __future__ import annotations

import random
from collections import defaultdict

from torch.utils.data import Sampler

from qwen3vl_seg.data.buckets import choose_bucket


class BucketOrderSampler(Sampler[int]):
    """Yield indices grouped by bucket; DataLoader batches consecutive indices.

    Within each bucket the order is shuffled. Trailing indices that would not
    form a full batch are dropped so a batch never mixes two buckets.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        seed: int = 42,
        bucket_sizes: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        rng = random.Random(seed)
        buckets: dict[int, list[int]] = defaultdict(list)
        for idx, sample in enumerate(getattr(dataset, "samples", [])):
            h, w = (int(v) for v in sample.image_size)
            buckets[choose_bucket(h, w, bucket_sizes)].append(idx)

        self.order: list[int] = []
        for key in sorted(buckets):
            indices = list(buckets[key])
            rng.shuffle(indices)
            drop = len(indices) % self.batch_size
            if drop:
                indices = indices[: len(indices) - drop]
            self.order.extend(indices)

    def __iter__(self):
        yield from self.order

    def __len__(self) -> int:
        return len(self.order)


class BucketProcessSampler(Sampler[int]):
    """DDP-aware sampler: same-bucket global batches, split across ranks.

    Every ``batch_size * world_size`` consecutive indices come from the same
    bucket; rank ``rank`` receives its own ``batch_size`` slice. Leftover
    indices that cannot form a full global batch are dropped.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 42,
        bucket_sizes: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        group_size = self.batch_size * max(1, int(world_size))
        rng = random.Random(seed)
        buckets: dict[int, list[int]] = defaultdict(list)
        for idx, sample in enumerate(getattr(dataset, "samples", [])):
            h, w = (int(v) for v in sample.image_size)
            buckets[choose_bucket(h, w, bucket_sizes)].append(idx)

        global_order: list[int] = []
        for key in sorted(buckets):
            indices = list(buckets[key])
            rng.shuffle(indices)
            drop = len(indices) % group_size
            if drop:
                indices = indices[: len(indices) - drop]
            global_order.extend(indices)

        rank_order: list[int] = []
        for start in range(0, len(global_order), group_size):
            group = global_order[start : start + group_size]
            rank_order.extend(
                group[self.batch_size * rank : self.batch_size * (rank + 1)]
            )
        self.order = rank_order

    def __iter__(self):
        yield from self.order

    def __len__(self) -> int:
        return len(self.order)
