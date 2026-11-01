"""Load a stored run for inference and score a loader of images with it.

Validation re-evaluation and the sealed test evaluation both use this module, so the two splits
are scored by the same code: the same checkpoint loading, compatibility checks, batch size and
score definitions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader, Dataset

from fewshotlab.config import RunConfig, load_run_config
from fewshotlab.data.manifest import ClassMap
from fewshotlab.data.transforms import spec_for
from fewshotlab.engine.trainer import EVAL_BATCH_SIZE
from fewshotlab.io_utils import read_json
from fewshotlab.models.methods import (
    Outputs,
    collect_embeddings,
    collect_logits,
    get_method,
    prototype_outputs,
)
from fewshotlab.models.spec import IncompatibleModelError, ModelSpec, load_linear, load_prototypes
from fewshotlab.paths import ProjectPaths


@dataclass
class StoredModel:
    cfg: RunConfig
    spec: ModelSpec
    device: str
    score: Callable[[DataLoader], Outputs]


def eval_loader(dataset: Dataset, num_workers: int) -> DataLoader:
    """Deterministic order, fixed batch size; worker count does not change any value."""
    return DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
    )


def load_stored_model(
    attempt_dir: Path, paths: ProjectPaths, class_map: ClassMap, device: str
) -> StoredModel:
    cfg = load_run_config(attempt_dir / "config.resolved.yaml")
    spec = ModelSpec.from_json(read_json(attempt_dir / "model_spec.json"))
    if spec_for(cfg.dataset, "eval").transform_id != spec.eval_transform_id:
        raise IncompatibleModelError(
            "this code version's evaluation transform differs from the "
            "one the model was trained with"
        )
    expected = {
        "dataset": cfg.dataset,
        "class_map_sha256": class_map.sha256(),
        "eval_transform_id": spec.eval_transform_id,
        "method": cfg.method,
    }
    if get_method(cfg.method).learned:
        model, _ = load_linear(attempt_dir / "best.ckpt", **expected)
        model.to(device)

        def score(loader: DataLoader) -> Outputs:
            return collect_logits(model, loader, device)

        return StoredModel(cfg, spec, device, score)

    if spec.encoder.get("weights_id") == "none":
        from fewshotlab.models.backbone import build_resnet18

        encoder, _ = build_resnet18(
            spec.num_classes, init="random", init_seed=int(spec.encoder["init_seed"])
        )
        prototypes = load_prototypes(attempt_dir / "prototypes.npz", **expected)[1]
    else:
        encoder, prototypes, _ = load_prototypes(
            attempt_dir / "prototypes.npz", weights_dir=paths.weights_dir, **expected
        )
    encoder.to(device)

    def score(loader: DataLoader) -> Outputs:
        query, labels, index = collect_embeddings(encoder, loader, device)
        return prototype_outputs(prototypes, query, labels, index)

    return StoredModel(cfg, spec, device, score)
