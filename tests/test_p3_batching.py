"""P3: training batches never drop an image and never end with a singleton."""

from __future__ import annotations

import pytest

from fewshotlab.engine.batching import NoSingletonBatchSampler


@pytest.mark.parametrize("n", [2, 6, 33, 65, 102, 306, 510, 1020])
def test_every_image_once_and_no_singleton(n):
    sampler = NoSingletonBatchSampler(n, 32, seed_labels=("loader", 11))
    for epoch in range(3):
        sampler.set_epoch(epoch)
        batches = list(sampler)
        assert sorted(i for batch in batches for i in batch) == list(range(n))
        assert all(len(batch) >= 2 for batch in batches)
        assert max(len(batch) for batch in batches) <= min(32, n) + 1
        assert len(batches) == len(sampler)


def test_neu_one_shot_is_a_single_batch_of_six():
    assert [len(b) for b in NoSingletonBatchSampler(6, 32, seed_labels=("loader", 11))] == [6]


def test_a_singleton_tail_is_merged():
    assert [len(b) for b in NoSingletonBatchSampler(33, 32, seed_labels=("loader", 11))] == [33]
    assert [len(b) for b in NoSingletonBatchSampler(65, 32, seed_labels=("x",))] == [32, 33]


def test_order_is_seeded_per_epoch():
    sampler = NoSingletonBatchSampler(40, 32, seed_labels=("loader", 11))
    sampler.set_epoch(0)
    first = list(sampler)
    assert first == list(sampler)
    sampler.set_epoch(1)
    assert list(sampler) != first


def test_fewer_than_two_images_is_rejected():
    with pytest.raises(ValueError):
        NoSingletonBatchSampler(1, 32, seed_labels=("x",))
