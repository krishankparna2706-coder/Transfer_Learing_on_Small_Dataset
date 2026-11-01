"""P2: datasets accept support/validation only; augmentation is per sample and per epoch."""

from __future__ import annotations

import pytest
import torch

from data_fixtures import make_dataset
from fewshotlab.data.datasets import ManifestImageDataset
from fewshotlab.data.transforms import build_transform
from fewshotlab.errors import ProtocolRefusal


@pytest.fixture
def fixture(tmp_path):
    records, _ = make_dataset(tmp_path / "raw", per_split={"train": 2, "val": 1, "test": 1})
    return tmp_path / "raw", records


def test_test_records_are_refused(fixture):
    root, records = fixture
    with pytest.raises(ProtocolRefusal, match="sealed evaluation gate"):
        ManifestImageDataset(records, root, build_transform("fixture", "eval"))


def test_eval_dataset_is_deterministic(fixture):
    root, records = fixture
    val = [r for r in records if r.split == "val"]
    dataset = ManifestImageDataset(val, root, build_transform("fixture", "eval"))
    x0, label, index = dataset[0]
    assert x0.shape == (3, 64, 64) and label == val[0].class_id and index == 0
    assert torch.equal(x0, dataset[0][0])


def test_augmentation_depends_on_epoch_and_image_only(fixture):
    root, records = fixture
    train = [r for r in records if r.split == "train"]
    dataset = ManifestImageDataset(
        train,
        root,
        build_transform("fixture", "train"),
        augment=True,
        augmentation_labels=("augment", 11),
    )
    views = set()
    for epoch in range(8):
        dataset.set_epoch(epoch)
        views.add(dataset[0][0].numpy().tobytes())
    assert len(views) > 1  # flips vary across epochs
    dataset.set_epoch(3)
    first = dataset[0][0]
    clone = ManifestImageDataset(
        train,
        root,
        build_transform("fixture", "train"),
        augment=True,
        augmentation_labels=("augment", 11),
    )
    clone.set_epoch(3)
    assert torch.equal(first, clone[0][0])


def test_stage_and_transform_must_match(fixture):
    root, records = fixture
    train = [r for r in records if r.split == "train"]
    with pytest.raises(ValueError):
        ManifestImageDataset(train, root, build_transform("fixture", "train"))
    with pytest.raises(ValueError):
        ManifestImageDataset(
            train,
            root,
            build_transform("fixture", "eval"),
            augment=True,
            augmentation_labels=("a",),
        )
