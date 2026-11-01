"""P5: the corruption generator, checked against hand values and independent oracles (CPU)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from PIL import Image
from scipy import ndimage

from fewshotlab.data.imageio import open_image
from fewshotlab.data.transforms import IMAGENET_MEAN, IMAGENET_STD, build_transform
from fewshotlab.hashing import sha256_bytes, sha256_json
from fewshotlab.seeds import derive_seed
from fewshotlab.simulation import corruptions as C


def _image(path, mode, size=(40, 32), seed=0):
    rng = np.random.default_rng(seed)
    shape = (size[1], size[0]) if mode == "L" else (size[1], size[0], 3)
    Image.fromarray(rng.integers(0, 256, size=shape, dtype=np.uint8), mode=mode).save(path)
    return path


@pytest.mark.parametrize(
    "dataset,mode,name", [("neu_cls", "L", "x.bmp"), ("flowers102", "RGB", "x.png")]
)
def test_clean_condition_equals_the_evaluation_contract(tmp_path, dataset, mode, name):
    path = _image(tmp_path / name, mode, size=(300, 260))
    transform = build_transform(dataset, "eval")
    x01 = C.decode01(path)
    assert x01.shape[0] == (1 if mode == "L" else 3)
    clean, info = C.corrupt(x01, "clean", 0, "img")
    assert torch.equal(C.to_model_input(clean, transform), transform(open_image(path)))
    assert info["seed"] is None and info["derivative_id"] == "img#clean"


def test_parent_is_never_modified_and_every_record_is_complete(tmp_path):
    x01 = C.decode01(_image(tmp_path / "x.png", "RGB"))
    before = x01.clone()
    for condition, severity in C.CONDITIONS:
        derived, info = C.corrupt(x01, condition, severity, "p1")
        assert torch.equal(x01, before)
        assert derived.shape == x01.shape and derived.dtype == torch.float32
        assert float(derived.min()) >= 0.0 and float(derived.max()) <= 1.0
        assert info["checksum_sha256"] == sha256_bytes(derived.numpy().tobytes())
        assert info["implementation"] == C.IMPLEMENTATION
        assert info["parameters"] == C.PARAMETERS[(condition, severity)]
        assert info["derivative_id"] == f"p1#{C.condition_id(condition, severity)}"
        assert info["parent_image_id"] == "p1"


def test_brightness_multiplies_then_clips():
    x = torch.tensor([[[0.0, 0.5, 0.8, 1.0]]])
    dark, _ = C.corrupt(x, "brightness", 1, "p")
    bright, _ = C.corrupt(x, "brightness", 2, "p")
    assert torch.allclose(dark, torch.tensor([[[0.0, 0.3, 0.48, 0.6]]]))
    assert torch.allclose(bright, torch.tensor([[[0.0, 0.7, 1.0, 1.0]]]))


@pytest.mark.parametrize("severity,sigma,taps", [(1, 1.0, 7), (2, 2.0, 13)])
def test_blur_matches_an_independent_reflect_padded_oracle(severity, sigma, taps):
    kernel = C.gaussian_kernel1d(sigma)
    assert len(kernel) == taps == 2 * math.ceil(3 * sigma) + 1
    assert abs(float(kernel.sum()) - 1.0) < 1e-6
    x = torch.rand(3, 31, 37, generator=torch.Generator().manual_seed(5))
    blurred, _ = C.corrupt(x, "gaussian_blur", severity, "p")
    # torch "reflect" padding mirrors about the edge pixel without repeating it: scipy "mirror"
    oracle = np.stack(
        [
            ndimage.gaussian_filter(ch.double().numpy(), sigma, mode="mirror", truncate=3.0)
            for ch in x
        ]
    )
    assert float(np.max(np.abs(blurred.numpy() - oracle))) < 1e-5
    flat, _ = C.corrupt(torch.full((1, 20, 20), 0.25), "gaussian_blur", severity, "p")
    assert torch.allclose(flat, torch.full((1, 20, 20), 0.25), atol=1e-6)


def test_noise_is_seeded_per_parent_and_grayscale_stays_gray(tmp_path):
    gray = C.decode01(_image(tmp_path / "g.bmp", "L"))
    assert gray.shape[0] == 1
    a, info = C.corrupt(gray, "gaussian_noise", 2, "img_0001")
    b, _ = C.corrupt(gray, "gaussian_noise", 2, "img_0001")
    c, _ = C.corrupt(gray, "gaussian_noise", 2, "img_0002")
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert info["seed"] == derive_seed("img_0001", "gaussian_noise", 2, 703)
    assert C.noise_seed("img_0001", "brightness", 1) is None
    model_input = C.to_model_input(a, build_transform("neu_cls", "eval"))
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    pixels = model_input * std + mean
    assert torch.allclose(pixels[0], pixels[1], atol=1e-5)
    assert torch.allclose(pixels[1], pixels[2], atol=1e-5)


@pytest.mark.parametrize("severity,sd", [(1, 0.03), (2, 0.08)])
def test_noise_has_the_sealed_standard_deviation(severity, sd):
    x = torch.full((1, 200, 200), 0.5)
    noisy, _ = C.corrupt(x, "gaussian_noise", severity, "mid-gray")
    delta = (noisy - x).double()
    assert abs(float(delta.std()) - sd) < 0.03 * sd
    assert abs(float(delta.mean())) < 0.002


def test_unknown_conditions_and_training_transforms_are_rejected():
    with pytest.raises(ValueError):
        C.corrupt(torch.zeros(1, 8, 8), "fog", 1, "p")
    with pytest.raises(ValueError):
        C.corrupt(torch.zeros(1, 8, 8), "brightness", 3, "p")
    with pytest.raises(ValueError):
        C.to_model_input(torch.zeros(1, 8, 8), build_transform("neu_cls", "train"))


def test_protocol_record_lists_the_sealed_conditions():
    record = C.protocol()
    assert [(c["condition"], c["severity"]) for c in record["conditions"]] == list(C.CONDITIONS)
    assert [C.condition_id(c, s) for c, s in C.CONDITIONS] == [
        "clean",
        "brightness-1",
        "brightness-2",
        "gaussian_blur-1",
        "gaussian_blur-2",
        "gaussian_noise-1",
        "gaussian_noise-2",
    ]
    assert record["seed"]["labels"][-1] == C.CORRUPTION_SEED == 703
    assert sha256_json(record) == sha256_json(json.loads(json.dumps(record)))
