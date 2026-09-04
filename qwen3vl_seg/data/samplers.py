"""Deterministic distributed sample ordering for resumable training."""

from __future__ import annotations

from torch.utils.data import Sampler


class StepIndexSampler(Sampler[int]):
    """Generate one local batch index per micro-batch across optimizer steps.

    The global order only depends on ``seed`` and the global micro-batch index,
    so a checkpoint can resume from the exact sample position without replaying
    a full epoch.
    """

    def __init__(
        self,
        dataset_size: int,
        world_size: int,
        rank: int,
        start_step: int,
        total_steps: int,
        gradient_accumulation_steps: int,
        seed: int = 42,
    ) -> None:
        self.dataset_size = dataset_size
        self.world_size = world_size
        self.rank = rank
        self.start_micro = start_step * gradient_accumulation_steps
        self.num_micro = max(0, total_steps - start_step) * gradient_accumulation_steps
        self.seed = seed

    def __iter__(self):
        for local_micro in range(self.num_micro):
            global_micro = self.start_micro + local_micro
            source = self.seed + global_micro * self.world_size + self.rank
            yield source % self.dataset_size

    def __len__(self) -> int:
        return self.num_micro
