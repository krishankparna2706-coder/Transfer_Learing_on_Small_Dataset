"""Training batches: size min(cap, N), shuffled from the run's loader stream, no singleton tail.

If the last batch would hold a single image, it is merged into the previous batch. Every support
image is used exactly once per epoch and none is ever dropped.
"""

from __future__ import annotations

from collections.abc import Iterator

from torch.utils.data import Sampler

from fewshotlab.seeds import numpy_rng


def plan_batches(order: list[int], batch_size: int) -> list[list[int]]:
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    return batches


class NoSingletonBatchSampler(Sampler[list[int]]):
    def __init__(self, n: int, cap: int, *, seed_labels: tuple[str | int, ...]) -> None:
        if n < 2:
            raise ValueError("training needs at least 2 support images")
        self.n = n
        self.batch_size = min(cap, n)
        self.seed_labels = tuple(seed_labels)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def batches(self) -> list[list[int]]:
        order = numpy_rng(*self.seed_labels, self.epoch).permutation(self.n).tolist()
        return plan_batches(order, self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.batches())

    def __len__(self) -> int:
        return len(plan_batches(list(range(self.n)), self.batch_size))
