"""P2: checks with the real ResNet18 IMAGENET1K_V1 weights (skipped when they are not cached)."""

from __future__ import annotations

import pytest
import torch

from fewshotlab.data.manifest import ClassMap
from fewshotlab.data.transforms import spec_for
from fewshotlab.models.backbone import WEIGHTS_FILENAME, build_resnet18
from fewshotlab.models.methods import build_model
from fewshotlab.models.spec import (
    IncompatibleModelError,
    ModelSpec,
    load_prototypes,
    save_prototypes,
)
from fewshotlab.paths import get_paths
from test_p2_policies import assert_policy_after_one_step

WEIGHTS_DIR = get_paths().weights_dir
pytestmark = pytest.mark.skipif(
    not (WEIGHTS_DIR / WEIGHTS_FILENAME).is_file(),
    reason="ResNet18 IMAGENET1K_V1 not cached; run `python -m fewshotlab weights fetch`",
)


def test_imagenet_weights_identity_and_loading():
    model, info = build_resnet18(102, init="imagenet", init_seed=0, weights_dir=WEIGHTS_DIR)
    assert info.weights_id == "IMAGENET1K_V1"
    assert info.sha256.startswith("f37072fd") and len(info.sha256) == 64
    random_model, _ = build_resnet18(102, init="random", init_seed=0)
    assert not torch.equal(model.encoder.conv1.weight, random_model.encoder.conv1.weight)
    assert not torch.allclose(model.encoder.bn1.running_var, torch.ones(64))
    assert torch.equal(model.head.weight, random_model.head.weight)  # same seeded new head


@pytest.mark.parametrize("method", ["frozen", "partial"])
def test_policies_hold_with_pretrained_weights(method):
    model, _ = build_model(method, 6, init_seed=0, weights_dir=WEIGHTS_DIR)
    batch = (
        torch.randn(4, 3, 64, 64, generator=torch.Generator().manual_seed(0)),
        torch.tensor([0, 1, 2, 3]),
    )
    assert_policy_after_one_step(model, method, batch)


def test_prototypes_reload_with_the_recorded_imagenet_encoder(tmp_path):
    _, info = build_resnet18(6, init="imagenet", init_seed=0, weights_dir=WEIGHTS_DIR)
    names = tuple(f"c{i}" for i in range(6))
    transform = spec_for("neu_cls", "eval")
    spec = ModelSpec(
        method="prototypes",
        dataset="neu_cls",
        class_names=names,
        class_map_sha256=ClassMap("neu_cls", names).sha256(),
        eval_transform_id=transform.transform_id,
        eval_transform=transform.signature(),
        encoder={
            "architecture": "resnet18",
            "weights_id": info.weights_id,
            "url": info.url,
            "filename": info.filename,
            "sha256": info.sha256,
            "init_seed": 0,
        },
        method_params={"epsilon": 1e-12},
    )
    save_prototypes(tmp_path / "p.npz", torch.randn(6, 512), spec)
    encoder, prototypes, _ = load_prototypes(
        tmp_path / "p.npz", weights_dir=WEIGHTS_DIR, dataset="neu_cls"
    )
    assert prototypes.shape == (6, 512) and not encoder.training
    tampered = ModelSpec.from_json(
        {**spec.to_json(), "encoder": {**spec.encoder, "sha256": "0" * 64}}
    )
    save_prototypes(tmp_path / "t.npz", torch.randn(6, 512), tampered)
    with pytest.raises(IncompatibleModelError, match="weights"):
        load_prototypes(tmp_path / "t.npz", weights_dir=WEIGHTS_DIR)
