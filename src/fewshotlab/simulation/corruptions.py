"""Controlled synthetic corruptions for the NEU robustness study and replay (protocol v1).

Corruptions are applied to decoded pixel values in [0, 1], at the image's native resolution,
before the dataset's evaluation resize and ImageNet normalization. The clean parent is never
modified. Derivatives inherit the parent's split and never enter training or model selection.

Conditions (level 1 / level 2):
* brightness: multiply by 0.6 / 1.4, then clip to [0, 1].
* gaussian_blur: sigma 1 / 2 px; separable Gaussian, kernel 2*ceil(3*sigma)+1, reflect padding.
* gaussian_noise: SD 0.03 / 0.08; zero mean, one draw per decoded pixel, then clip to [0, 1].

Grayscale sources (NEU) are corrupted as one channel and replicated to RGB afterwards, so the noise
stays gray. Noise seeds come from SHA-256 of the JSON label list
``[parent_id, condition, severity, 703]``; Python's ``hash()`` is never used.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_f

from fewshotlab.data.imageio import is_grayscale_content, open_image, to_rgb
from fewshotlab.data.transforms import ImageTransform
from fewshotlab.hashing import sha256_bytes
from fewshotlab.seeds import SEED_DERIVATION, derive_seed

IMPLEMENTATION = "fewshotlab.simulation.corruptions v1"
CORRUPTION_SEED = 703
CONDITIONS: tuple[tuple[str, int], ...] = (
    ("clean", 0),
    ("brightness", 1),
    ("brightness", 2),
    ("gaussian_blur", 1),
    ("gaussian_blur", 2),
    ("gaussian_noise", 1),
    ("gaussian_noise", 2),
)
PARAMETERS: dict[tuple[str, int], dict[str, float]] = {
    ("clean", 0): {},
    ("brightness", 1): {"factor": 0.6},
    ("brightness", 2): {"factor": 1.4},
    ("gaussian_blur", 1): {"sigma": 1.0},
    ("gaussian_blur", 2): {"sigma": 2.0},
    ("gaussian_noise", 1): {"sd": 0.03},
    ("gaussian_noise", 2): {"sd": 0.08},
}


def protocol() -> dict[str, Any]:
    """Everything the seal records about the corruption protocol."""
    return {
        "implementation": IMPLEMENTATION,
        "conditions": [
            {"condition": c, "severity": s, "parameters": PARAMETERS[(c, s)]} for c, s in CONDITIONS
        ],
        "pixel_space": "decoded [0, 1] values at native resolution, before resize/normalization",
        "clipping": "brightness and noise outputs clipped to [0, 1]",
        "blur": "separable Gaussian; kernel size 2*ceil(3*sigma)+1; reflect padding",
        "noise": "zero-mean Gaussian, one draw per decoded pixel (one channel for grayscale)",
        "seed": {
            "labels": ["<parent_id>", "<condition>", "<severity>", CORRUPTION_SEED],
            "derivation": SEED_DERIVATION,
            "generator": "numpy.random.PCG64",
        },
    }


def condition_id(condition: str, severity: int) -> str:
    return "clean" if condition == "clean" else f"{condition}-{severity}"


def decode01(source: str | Path | bytes) -> torch.Tensor:
    """Decoded image as float32 (C, H, W) in [0, 1].

    C = 1 when the image has grayscale content (NEU), else 3. Decoding matches the clean model
    path exactly: the same safe decoder, EXIF orientation and RGB conversion.
    """
    image = open_image(source)
    rgb = np.asarray(to_rgb(image), dtype=np.uint8)
    array = rgb[:, :, :1] if is_grayscale_content(image) else rgb
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).to(torch.float32).div_(255.0)


def noise_seed(parent_id: str, condition: str, severity: int) -> int | None:
    """The generator seed of a derivative (noise only; the other conditions use no randomness)."""
    if condition != "gaussian_noise":
        return None
    return derive_seed(parent_id, condition, severity, CORRUPTION_SEED)


def gaussian_kernel1d(sigma: float) -> torch.Tensor:
    radius = math.ceil(3 * sigma)
    x = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    return (kernel / kernel.sum()).to(torch.float32)


def _blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = gaussian_kernel1d(sigma)
    radius = len(kernel) // 2
    channels = x.shape[0]
    batch = torch_f.pad(x[None], (radius, radius, radius, radius), mode="reflect")
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    batch = torch_f.conv2d(batch, horizontal, groups=channels)
    batch = torch_f.conv2d(batch, vertical, groups=channels)
    return batch[0]


def corrupt(
    x01: torch.Tensor, condition: str, severity: int, parent_id: str
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return the derivative and its record (parameters, seed, implementation, checksum)."""
    key = (condition, severity)
    if key not in PARAMETERS:
        raise ValueError(f"unknown corruption {condition!r} severity {severity}")
    params = PARAMETERS[key]
    seed = None
    if condition == "clean":
        y = x01.clone()
    elif condition == "brightness":
        y = (x01 * params["factor"]).clamp(0.0, 1.0)
    elif condition == "gaussian_blur":
        y = _blur(x01, params["sigma"])
    else:
        seed = noise_seed(parent_id, condition, severity)
        rng = np.random.Generator(np.random.PCG64(seed))
        noise = rng.normal(0.0, params["sd"], size=tuple(x01.shape)).astype(np.float32)
        y = (x01 + torch.from_numpy(noise)).clamp(0.0, 1.0)
    y = y.contiguous()
    record = {
        "parent_image_id": parent_id,
        "derivative_id": f"{parent_id}#{condition_id(condition, severity)}",
        "condition": condition,
        "severity": severity,
        "parameters": params,
        "seed": seed,
        "implementation": IMPLEMENTATION,
        "checksum_sha256": sha256_bytes(y.numpy().astype(np.float32).tobytes()),
    }
    return y, record


def to_model_input(y01: torch.Tensor, transform: ImageTransform) -> torch.Tensor:
    """Apply the dataset's deterministic evaluation contract to a [0, 1] tensor."""
    if transform.spec.stage != "eval":
        raise ValueError("the corruption study uses the deterministic evaluation contract")
    x = y01.expand(3, -1, -1).contiguous() if y01.shape[0] == 1 else y01
    return transform.apply_tensor(x)
