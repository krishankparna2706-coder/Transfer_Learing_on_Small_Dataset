"""P0 SOFTWARE FIXTURE: random tensors through ResNet18.

These tests check shapes, parameter arithmetic and seeding. They are not benchmark results and
say nothing about Flowers102 or NEU-CLS accuracy.
"""

from __future__ import annotations

import pytest
import torch

from fewshotlab.errors import WeightsUnavailableError
from fewshotlab.models.backbone import EMBEDDING_DIM, build_resnet18

ENCODER_PARAMS = 11_176_512  # blueprint arithmetic: ResNet18 without its original classifier


@pytest.mark.parametrize("num_classes, total", [(102, 11_228_838), (6, 11_179_590)])
def test_forward_shapes_and_parameter_counts(num_classes, total):
    model, weights = build_resnet18(num_classes, init="random", init_seed=0)
    assert weights.weights_id == "none"
    model.eval()
    x = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    with torch.inference_mode():
        features = model.features(x)
        logits = model(x)
    assert features.shape == (2, EMBEDDING_DIM)
    assert logits.shape == (2, num_classes)
    assert torch.isfinite(logits).all()
    assert sum(p.numel() for p in model.encoder.parameters()) == ENCODER_PARAMS
    assert sum(p.numel() for p in model.head.parameters()) == 512 * num_classes + num_classes
    assert sum(p.numel() for p in model.parameters()) == total


def test_random_init_is_seeded_and_isolated():
    a, _ = build_resnet18(6, init="random", init_seed=11)
    torch.manual_seed(999)  # global RNG state must not leak into initialization
    b, _ = build_resnet18(6, init="random", init_seed=11)
    c, _ = build_resnet18(6, init="random", init_seed=22)
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key]), key
    assert not torch.equal(a.head.weight, c.head.weight)


def test_rejects_wrong_input_shape():
    model, _ = build_resnet18(6, init="random", init_seed=0)
    with pytest.raises(ValueError, match="batch, 3, H, W"):
        model(torch.zeros(2, 1, 224, 224))


def test_pretrained_without_cache_or_permission_fails_without_network(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("a network download was attempted")

    monkeypatch.setattr(torch.hub, "download_url_to_file", forbidden)
    with pytest.raises(WeightsUnavailableError) as exc:
        build_resnet18(6, init="imagenet", init_seed=0, weights_dir=tmp_path / "weights")
    assert "--allow-download" in str(exc.value)
    assert not (tmp_path / "weights").exists()
