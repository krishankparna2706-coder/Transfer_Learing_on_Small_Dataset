"""P2: method registry, output dimensions, prediction paths and cache-key isolation."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data_fixtures import make_dataset
from fewshotlab.data.datasets import ManifestImageDataset
from fewshotlab.data.transforms import build_transform
from fewshotlab.errors import ConfigError
from fewshotlab.models import methods
from fewshotlab.models.cache import EmbeddingCache, cache_key, cache_metadata
from fewshotlab.models.prototypes import fit_prototypes


def test_registry():
    assert set(methods.REGISTRY) == {"scratch", "frozen", "partial", "prototypes"}
    assert methods.get_method("scratch").init == "random"
    assert all(
        methods.get_method(m).init == "imagenet" for m in ("frozen", "partial", "prototypes")
    )
    with pytest.raises(ConfigError):
        methods.get_method("maml")


@pytest.mark.parametrize("classes", [102, 6])
def test_output_dimensions(classes):
    model, _ = methods.build_model("scratch", classes, init_seed=0, init_override="random")
    model.eval()
    with torch.inference_mode():
        assert model(torch.zeros(2, 3, 64, 64)).shape == (2, classes)


def test_learned_and_prototype_prediction_paths(tmp_path):
    records, _ = make_dataset(tmp_path / "raw", per_split={"train": 3, "val": 2, "test": 1})
    root = tmp_path / "raw"
    support = [r for r in records if r.split == "train"]
    val = [r for r in records if r.split == "val"]
    transform = build_transform("fixture", "eval")
    support_loader = DataLoader(ManifestImageDataset(support, root, transform), batch_size=4)
    val_loader = DataLoader(ManifestImageDataset(val, root, transform), batch_size=4)
    model, _ = methods.build_model("scratch", 3, init_seed=0, init_override="random")
    out = methods.collect_logits(model, val_loader, "cpu")
    assert out.raw.shape == (6, 3) and np.allclose(out.scores.sum(1), 1.0, atol=1e-5)
    assert np.array_equal(out.pred, out.raw.argmax(1))
    assert sorted(out.indices.tolist()) == list(range(6))
    encoder, _ = methods.build_model("prototypes", 3, init_seed=0, init_override="random")
    embeddings, labels, _ = methods.collect_embeddings(encoder, support_loader, "cpu")
    assert embeddings.shape == (9, 512)
    prototypes = fit_prototypes(embeddings, labels, 3)
    query, qlabels, qindex = methods.collect_embeddings(encoder, val_loader, "cpu")
    proto_out = methods.prototype_outputs(prototypes, query, qlabels, qindex)
    assert proto_out.scores.shape == (6, 3)
    assert np.array_equal(proto_out.pred, proto_out.raw.argmax(1))  # raw = -distance


BASE = dict(
    encoder_sha256="a" * 64,
    transform_id="t1",
    manifest_sha256="m" * 64,
    class_map_sha256="c" * 64,
    image_ids=["x", "y"],
)


@pytest.mark.parametrize(
    "field, value",
    [
        ("encoder_sha256", "b" * 64),
        ("transform_id", "t2"),
        ("manifest_sha256", "n" * 64),
        ("class_map_sha256", "d" * 64),
        ("image_ids", ["y", "x"]),
        ("image_ids", ["x"]),
    ],
)
def test_cache_key_changes_with_every_component(field, value):
    assert cache_key(cache_metadata(**{**BASE, field: value})) != cache_key(cache_metadata(**BASE))


def test_cache_mismatch_is_a_miss(tmp_path):
    cache = EmbeddingCache(tmp_path)
    metadata = cache_metadata(**BASE)
    cache.save(metadata, np.ones((2, 4), dtype=np.float32))
    assert np.array_equal(cache.load(metadata), np.ones((2, 4), dtype=np.float32))
    assert cache.load(cache_metadata(**{**BASE, "transform_id": "t9"})) is None
    with pytest.raises(ValueError):
        cache.save(metadata, np.ones((3, 4), dtype=np.float32))
