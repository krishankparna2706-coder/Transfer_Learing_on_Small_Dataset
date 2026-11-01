"""P2: preprocessing contracts are pinned, deterministic in evaluation, seeded in training."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from fewshotlab.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_transform,
    pil_to_tensor01,
    rrc_params,
    spec_for,
)
from fewshotlab.seeds import numpy_rng


def _image(size=(300, 400), mode="RGB", seed=0):
    rng = np.random.default_rng(seed)
    shape = (size[1], size[0], 3) if mode == "RGB" else (size[1], size[0])
    return Image.fromarray(rng.integers(0, 256, size=shape, dtype=np.uint8))


@pytest.mark.parametrize("dataset", ["flowers102", "neu_cls"])
def test_eval_transform_is_deterministic_and_shaped(dataset):
    transform = build_transform(dataset, "eval")
    image = _image((500, 375)) if dataset == "flowers102" else _image((200, 200), mode="L")
    first, second = transform(image), transform(image)
    assert first.shape == (3, 224, 224) and first.dtype == torch.float32
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


def test_flowers_eval_equals_the_torchvision_reference_composition():
    from torchvision import transforms as T

    image = _image((500, 667))
    ours = build_transform("flowers102", "eval")(image)
    reference = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.CenterCrop(224),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )(pil_to_tensor01(image))
    assert torch.equal(ours, reference)


def test_grayscale_is_replicated_and_normalized_numerically():
    gray = Image.fromarray(np.full((200, 200), 128, dtype=np.uint8))
    assert gray.mode == "L"
    x = build_transform("neu_cls", "eval")(gray)
    for channel in range(3):
        expected = (128 / 255 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
        assert torch.allclose(x[channel], torch.full_like(x[channel], expected), atol=1e-6)


def test_training_transform_depends_only_on_its_generator():
    transform = build_transform("flowers102", "train")
    image = _image((500, 667))
    first = transform(image, numpy_rng("augment", 11, 0, "img-1"))
    torch.manual_seed(123)
    np.random.seed(123)  # global random state must not matter
    again = transform(image, numpy_rng("augment", 11, 0, "img-1"))
    other_epoch = transform(image, numpy_rng("augment", 11, 1, "img-1"))
    assert first.shape == (3, 224, 224)
    assert torch.equal(first, again)
    assert not torch.equal(first, other_epoch)


def test_generator_usage_is_enforced():
    with pytest.raises(ValueError, match="explicit random generator"):
        build_transform("neu_cls", "train")(_image((200, 200), mode="L"))
    with pytest.raises(ValueError, match="deterministic"):
        build_transform("neu_cls", "eval")(_image((200, 200), mode="L"), numpy_rng("x"))


def test_random_resized_crop_parameters():
    rng = np.random.default_rng(0)
    for _ in range(300):
        top, left, h, w = rrc_params(375, 500, (0.7, 1.0), (3 / 4, 4 / 3), rng)
        assert top >= 0 and left >= 0 and top + h <= 375 and left + w <= 500
        assert h * w >= 0.7 * 375 * 500 * 0.98
    # an extreme aspect ratio falls back to torchvision's central crop rule
    assert rrc_params(10, 1000, (0.7, 1.0), (3 / 4, 4 / 3), np.random.default_rng(0)) == (
        0,
        493,
        10,
        13,
    )


def test_neu_training_flips_only_permute_pixels():
    image = _image((200, 200), mode="L")
    augmented = build_transform("neu_cls", "train")(image, numpy_rng("augment", 1, 0, "a"))
    plain = build_transform("neu_cls", "eval")(image)
    assert augmented.shape == plain.shape
    assert torch.equal(torch.sort(augmented.flatten()).values, torch.sort(plain.flatten()).values)


def test_contracts_are_pinned_and_identified():
    ids = {
        (d, s): spec_for(d, s).transform_id
        for d in ("flowers102", "neu_cls")
        for s in ("train", "eval")
    }
    assert len(set(ids.values())) == 4
    flowers_train = spec_for("flowers102", "train")
    assert flowers_train.random_resized_crop == (0.7, 1.0, 3 / 4, 4 / 3)
    assert flowers_train.hflip_p == 0.5 and flowers_train.vflip_p == 0.0
    neu_train = spec_for("neu_cls", "train")
    assert neu_train.full_frame and neu_train.hflip_p == 0.5 and neu_train.vflip_p == 0.5
    signature = spec_for("neu_cls", "eval").signature()
    assert signature["interpolation"] == "bilinear" and signature["antialias"] is True
    assert "torchvision" in signature and "pillow" in signature
    with pytest.raises(ValueError):
        spec_for("cifar10", "eval")
