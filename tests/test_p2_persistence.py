"""P2: models are rebuilt from their saved spec; mismatches and pickled objects are refused."""

from __future__ import annotations

import json
import pickle

import pytest
import torch

from fewshotlab.data.manifest import ClassMap
from fewshotlab.data.transforms import spec_for
from fewshotlab.models.backbone import build_resnet18
from fewshotlab.models.methods import build_model
from fewshotlab.models.spec import (
    LINEAR_FORMAT,
    IncompatibleModelError,
    ModelSpec,
    load_linear,
    load_prototypes,
    save_linear,
    save_prototypes,
)

NAMES = ("a", "b", "c")


class NotAllowed:  # an arbitrary object that must never be unpickled from a checkpoint
    pass


def make_spec(method, names=NAMES, dataset="fixture-x", init_seed=5):
    transform = spec_for(dataset, "eval")
    return ModelSpec(
        method=method,
        dataset=dataset,
        class_names=names,
        class_map_sha256=ClassMap(dataset, names).sha256(),
        eval_transform_id=transform.transform_id,
        eval_transform=transform.signature(),
        encoder={"architecture": "resnet18", "weights_id": "none", "init_seed": init_seed},
        method_params={},
        is_fixture=True,
    )


@pytest.fixture
def x():
    return torch.randn(2, 3, 64, 64, generator=torch.Generator().manual_seed(1))


def test_linear_round_trip_reproduces_predictions(tmp_path, x):
    model, _ = build_model("scratch", 3, init_seed=5, init_override="random")
    model.eval()
    with torch.inference_mode():
        before = model(x)
    spec = make_spec("scratch")
    save_linear(tmp_path / "best.ckpt", model, spec)
    loaded, loaded_spec = load_linear(
        tmp_path / "best.ckpt",
        dataset="fixture-x",
        class_map_sha256=spec.class_map_sha256,
        eval_transform_id=spec.eval_transform_id,
    )
    with torch.inference_mode():
        after = loaded(x)
    assert torch.allclose(before, after, atol=1e-6, rtol=0)  # declared CPU tolerance
    assert loaded_spec.to_json() == spec.to_json()
    assert not loaded.training


@pytest.mark.parametrize(
    "expected",
    [
        {"dataset": "neu_cls"},
        {"class_map_sha256": "0" * 64},
        {"eval_transform_id": "0000000000000000"},
        {"class_names": ("a", "c", "b")},
        {"method": "partial"},
    ],
)
def test_incompatible_contexts_are_refused(tmp_path, expected):
    model, _ = build_model("scratch", 3, init_seed=5, init_override="random")
    save_linear(tmp_path / "best.ckpt", model, make_spec("scratch"))
    with pytest.raises(IncompatibleModelError):
        load_linear(tmp_path / "best.ckpt", **expected)


def test_a_neu_model_cannot_load_with_a_flowers_class_map(tmp_path):
    neu = tuple(f"neu_{i}" for i in range(6))
    model, _ = build_model("frozen", 6, init_seed=5, init_override="random")
    save_linear(tmp_path / "best.ckpt", model, make_spec("frozen", names=neu))
    flowers = ClassMap("fixture-x", tuple(f"label_{i:03d}" for i in range(1, 103)))
    with pytest.raises(IncompatibleModelError, match="class map differs"):
        load_linear(tmp_path / "best.ckpt", class_map_sha256=flowers.sha256())


def test_checkpoints_never_unpickle_arbitrary_objects(tmp_path):
    model, _ = build_model("scratch", 3, init_seed=5, init_override="random")
    payload = {
        "format": LINEAR_FORMAT,
        "spec_json": json.dumps(make_spec("scratch").to_json()),
        "state_dict": model.state_dict(),
        "extra": NotAllowed(),
    }
    torch.save(payload, tmp_path / "evil.ckpt")
    with pytest.raises(pickle.UnpicklingError):
        load_linear(tmp_path / "evil.ckpt")


def test_head_size_must_match_the_spec(tmp_path):
    model, _ = build_model("scratch", 4, init_seed=5, init_override="random")
    with pytest.raises(ValueError, match="class count"):
        save_linear(tmp_path / "best.ckpt", model, make_spec("scratch"))


def test_prototype_round_trip_rebuilds_the_recorded_encoder(tmp_path):
    prototypes = torch.randn(3, 512, generator=torch.Generator().manual_seed(2))
    spec = make_spec("prototypes", init_seed=7)
    save_prototypes(tmp_path / "prototypes.npz", prototypes, spec)
    encoder, loaded, loaded_spec = load_prototypes(tmp_path / "prototypes.npz", dataset="fixture-x")
    assert torch.equal(loaded, prototypes)
    assert loaded_spec.to_json() == spec.to_json()
    reference, _ = build_resnet18(3, init="random", init_seed=7)
    for key, value in reference.encoder.state_dict().items():
        assert torch.equal(value, encoder.encoder.state_dict()[key]), key
    with pytest.raises(IncompatibleModelError):
        load_prototypes(tmp_path / "prototypes.npz", class_names=("c", "b", "a"))
