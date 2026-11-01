"""Preprocessing contracts for protocol v1, identical in the CLI, the reports and the app.

Every path decodes safely, applies the EXIF orientation, converts to RGB (grayscale replicated,
alpha dropped), scales to [0, 1] float32, resizes with bilinear interpolation and antialiasing,
and normalizes with the ImageNet mean/std. Training augmentation draws its parameters from an
explicit NumPy generator, in a fixed order (crop, horizontal flip, vertical flip), so it never
shares a random stream with initialization or data loading. Evaluation transforms are
deterministic. A transform's full parameter set, plus the torchvision and Pillow versions, is
hashed into its transform ID.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import PIL
import torch
import torchvision
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from fewshotlab.data.imageio import to_rgb
from fewshotlab.hashing import sha256_json

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
STAGES = ("train", "eval")
IMPLEMENTATION = "fewshotlab.data.transforms v1 (float32 tensors, torchvision functional ops)"


@dataclass(frozen=True)
class TransformSpec:
    family: str  # "flowers", "neu" or "fixture"
    stage: str  # "train" or "eval"
    output_size: int = 224
    resize_shorter_to: int | None = None  # resize the shorter side, then center crop
    full_frame: bool = False  # resize the whole image to output_size x output_size
    random_resized_crop: tuple[float, float, float, float] | None = None  # scale lo/hi, ratio lo/hi
    hflip_p: float = 0.0
    vflip_p: float = 0.0
    interpolation: str = "bilinear"
    antialias: bool = True
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    color: str = "RGB; grayscale replicated to 3 channels; alpha dropped"
    orientation: str = "EXIF orientation applied before any resize"

    def signature(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "implementation": IMPLEMENTATION,
            "torchvision": torchvision.__version__,
            "pillow": PIL.__version__,
        }

    @property
    def transform_id(self) -> str:
        return sha256_json(self.signature())[:16]


def spec_for(dataset: str, stage: str, *, fixture_size: int = 64) -> TransformSpec:
    """The protocol v1 contract for a dataset and stage."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    if dataset == "flowers102":
        if stage == "train":
            return TransformSpec(
                "flowers", "train", random_resized_crop=(0.7, 1.0, 3 / 4, 4 / 3), hflip_p=0.5
            )
        return TransformSpec("flowers", "eval", resize_shorter_to=256)
    if dataset == "neu_cls":
        if stage == "train":
            return TransformSpec("neu", "train", full_frame=True, hflip_p=0.5, vflip_p=0.5)
        return TransformSpec("neu", "eval", full_frame=True)
    if dataset.startswith("fixture"):
        if stage == "train":
            return TransformSpec(
                "fixture", "train", output_size=fixture_size, full_frame=True, hflip_p=0.5
            )
        return TransformSpec("fixture", "eval", output_size=fixture_size, full_frame=True)
    raise ValueError(f"no transform contract for dataset {dataset!r}")


def pil_to_tensor01(image: Image.Image) -> torch.Tensor:
    """(3, H, W) float32 in [0, 1] from any PIL mode (grayscale replicated)."""
    array = np.asarray(to_rgb(image), dtype=np.uint8)
    return torch.from_numpy(array.copy()).permute(2, 0, 1).to(torch.float32).div_(255.0)


def rrc_params(
    height: int,
    width: int,
    scale: tuple[float, float],
    ratio: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    """torchvision's RandomResizedCrop.get_params algorithm, driven by an explicit generator."""
    area = height * width
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target_area = area * rng.uniform(scale[0], scale[1])
        aspect = math.exp(rng.uniform(log_ratio[0], log_ratio[1]))
        w = int(round(math.sqrt(target_area * aspect)))
        h = int(round(math.sqrt(target_area / aspect)))
        if 0 < w <= width and 0 < h <= height:
            top = int(rng.integers(0, height - h + 1))
            left = int(rng.integers(0, width - w + 1))
            return top, left, h, w
    in_ratio = width / height
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:
        w, h = width, height
    return (height - h) // 2, (width - w) // 2, h, w


class ImageTransform:
    """A callable preprocessing contract: ``transform(image, rng=None) -> tensor``."""

    def __init__(self, spec: TransformSpec) -> None:
        self.spec = spec

    @property
    def transform_id(self) -> str:
        return self.spec.transform_id

    def __call__(self, image: Image.Image, rng: np.random.Generator | None = None) -> torch.Tensor:
        return self.apply_tensor(pil_to_tensor01(image), rng)

    def apply_tensor(self, x: torch.Tensor, rng: np.random.Generator | None = None) -> torch.Tensor:
        """The contract on an already-decoded (3, H, W) float tensor in [0, 1].

        ``__call__`` and the corruption study both go through this one code path.
        """
        spec = self.spec
        if spec.stage == "train" and rng is None:
            raise ValueError("training transforms need an explicit random generator")
        if spec.stage == "eval" and rng is not None:
            raise ValueError("evaluation transforms are deterministic; do not pass a generator")
        if x.ndim != 3 or x.shape[0] != 3:
            raise ValueError(f"expected a (3, H, W) tensor, got {tuple(x.shape)}")
        mode = InterpolationMode.BILINEAR
        size = [spec.output_size, spec.output_size]
        if spec.random_resized_crop is not None:
            lo, hi, r_lo, r_hi = spec.random_resized_crop
            top, left, h, w = rrc_params(x.shape[1], x.shape[2], (lo, hi), (r_lo, r_hi), rng)
            x = F.resized_crop(
                x, top, left, h, w, size, interpolation=mode, antialias=spec.antialias
            )
        elif spec.full_frame:
            x = F.resize(x, size, interpolation=mode, antialias=spec.antialias)
        else:
            x = F.resize(x, [spec.resize_shorter_to], interpolation=mode, antialias=spec.antialias)
            x = F.center_crop(x, size)
        if spec.stage == "train":
            if spec.hflip_p and rng.random() < spec.hflip_p:
                x = F.hflip(x)
            if spec.vflip_p and rng.random() < spec.vflip_p:
                x = F.vflip(x)
        x = F.normalize(x, list(spec.mean), list(spec.std))
        return x.contiguous()


def build_transform(dataset: str, stage: str, *, fixture_size: int = 64) -> ImageTransform:
    return ImageTransform(spec_for(dataset, stage, fixture_size=fixture_size))
