"""Torch datasets over manifest records: support (train) and validation images only.

The held-out test split cannot enter these datasets. It is reachable only through the sealed
evaluation gate (P5).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import Dataset

from fewshotlab.data.imageio import open_image
from fewshotlab.data.manifest import ImageRecord
from fewshotlab.data.transforms import ImageTransform
from fewshotlab.errors import ProtocolRefusal
from fewshotlab.seeds import numpy_rng


class ManifestImageDataset(Dataset):
    """Images from manifest records with one fixed transform.

    With ``augment=True`` each sample's augmentation generator is derived from
    ``(*augmentation_labels, epoch, image_id)``. The result therefore depends only on the run's
    augmentation stream, the epoch and the image, never on worker scheduling or batch order.
    """

    def __init__(
        self,
        records: Sequence[ImageRecord],
        data_root: str | Path,
        transform: ImageTransform,
        *,
        augment: bool = False,
        augmentation_labels: tuple[str | int, ...] | None = None,
    ) -> None:
        if any(r.split == "test" for r in records):
            raise ProtocolRefusal(
                "test images can only be loaded through the sealed evaluation gate (P5)"
            )
        if augment and (transform.spec.stage != "train" or not augmentation_labels):
            raise ValueError("augmentation needs a training transform and augmentation labels")
        if not augment and transform.spec.stage != "eval":
            raise ValueError("a non-augmented dataset needs the deterministic evaluation transform")
        self.records = list(records)
        self.data_root = Path(data_root)
        self.transform = transform
        self.augment = augment
        self.augmentation_labels = tuple(augmentation_labels or ())
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        record = self.records[index]
        image = open_image(self.data_root / record.relative_path)
        rng = (
            numpy_rng(*self.augmentation_labels, self.epoch, record.image_id)
            if self.augment
            else None
        )
        return self.transform(image, rng), record.class_id, index
