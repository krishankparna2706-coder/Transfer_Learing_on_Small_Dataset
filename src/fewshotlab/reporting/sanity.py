"""Sanity checks on finished runs, from stored artifacts only.

The checks answer one question: did each method do what its policy says?
* the stored true labels match the manifest;
* every class is scored;
* the new head moved;
* the frozen encoder is byte-identical to its starting weights;
* partial changed only layer4 and no BatchNorm running statistic;
* scratch updated its BatchNorm statistics;
* prototypes are finite with consistent norms (and not re-normalized);
* augmented training views are finite and not blank.

These are software checks, not performance claims.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from fewshotlab.config import load_run_config
from fewshotlab.data.imageio import open_image
from fewshotlab.data.manifest import read_manifest
from fewshotlab.data.registry import manifest_path
from fewshotlab.data.support import load_support
from fewshotlab.data.transforms import IMAGENET_MEAN, IMAGENET_STD, build_transform
from fewshotlab.engine import artifacts
from fewshotlab.io_utils import atomic_write_json, read_json
from fewshotlab.models.backbone import build_resnet18
from fewshotlab.models.spec import ModelSpec, load_linear
from fewshotlab.paths import ProjectPaths
from fewshotlab.seeds import numpy_rng

BN_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")


def _starting_model(spec: ModelSpec, paths: ProjectPaths):
    init = spec.encoder.get("init") or (
        "random" if spec.encoder.get("weights_id") == "none" else "imagenet"
    )
    model, _ = build_resnet18(
        spec.num_classes,
        init=init,
        init_seed=int(spec.encoder["init_seed"]),
        weights_dir=paths.weights_dir,
    )
    return model


def check_run(paths: ProjectPaths, row: dict[str, Any]) -> dict[str, Any]:
    attempt_dir: Path = row["_attempt_dir"]
    cfg = load_run_config(attempt_dir / "config.resolved.yaml")
    spec = ModelSpec.from_json(read_json(attempt_dir / "model_spec.json"))
    stored = artifacts.read_predictions(attempt_dir / "predictions_val.csv")
    truth = {
        r.image_id: r.class_id
        for r in read_manifest(manifest_path(paths, cfg.dataset), splits=("val",))
    }
    checks: dict[str, bool] = {
        "labels_match_manifest": len(stored["image_ids"]) == len(truth)
        and all(
            truth.get(i) == int(t) for i, t in zip(stored["image_ids"], stored["true"], strict=True)
        ),
        "every_class_scored": int(stored["scores"].shape[1]) == spec.num_classes,
    }
    if cfg.method != "prototypes":
        trained, _ = load_linear(attempt_dir / "best.ckpt")
        start = _starting_model(spec, paths)
        checks["head_updated"] = not torch.equal(trained.head.weight, start.head.weight)
        after, before = trained.encoder.state_dict(), start.encoder.state_dict()
        changed = {k for k in before if not torch.equal(after[k], before[k])}
        buffers = {k for k in before if k.endswith(BN_BUFFERS)}
        if cfg.method == "frozen":
            checks["encoder_unchanged"] = not changed
        elif cfg.method == "partial":
            checks["only_layer4_changed"] = bool(changed) and all(
                k.startswith("layer4.") for k in changed
            )
            checks["no_batchnorm_drift"] = not (changed & buffers)
        else:
            checks["batchnorm_statistics_updated"] = bool(changed & buffers)
    else:
        with np.load(attempt_dir / "prototypes.npz", allow_pickle=False) as data:
            prototypes = np.array(data["prototypes"], dtype=np.float64)
        norms = np.linalg.norm(prototypes, axis=1)
        checks["prototypes_finite"] = bool(np.isfinite(prototypes).all())
        if cfg.shots == 1:  # one unit vector per class: norm 1
            checks["prototype_norms_consistent"] = bool(np.allclose(norms, 1.0, atol=1e-5))
        else:  # mean of unit vectors, not re-normalized: norm below 1
            checks["prototype_norms_consistent"] = bool((norms < 1.0 - 1e-6).all())
    return {
        "run_id": row["run_id"],
        "method": row["method"],
        "k": row["k"],
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def augmentation_check(paths: ProjectPaths, rows: list[dict[str, Any]], out_dir: Path):
    learned = next((r for r in rows if r.get("_attempt_dir") and r["method"] != "prototypes"), None)
    if learned is None:
        return None
    cfg = load_run_config(learned["_attempt_dir"] / "config.resolved.yaml")
    support = load_support(paths, cfg.dataset, cfg.shots, cfg.support_seed)
    ids = [support["selected"][key][0] for key in sorted(support["selected"], key=int)][:4]
    records = {
        r.image_id: r for r in read_manifest(manifest_path(paths, cfg.dataset), splits=("train",))
    }
    transform = build_transform(cfg.dataset, "train")
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    views, ok = [], True
    for image_id in ids:
        image = open_image(paths.data_root / records[image_id].relative_path)
        for epoch in (1, 2, 3, 4):
            x = transform(image, numpy_rng("augment", cfg.train_seed, epoch, image_id))
            size = transform.spec.output_size
            ok &= bool(torch.isfinite(x).all()) and tuple(x.shape) == (3, size, size)
            pixels = (x * std + mean).clamp(0, 1)
            ok &= float(pixels.std()) > 0.01
            views.append(pixels.permute(1, 2, 0).numpy())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(ids), 4, figsize=(8, 2 * len(ids)), constrained_layout=True)
    for axis, view in zip(np.atleast_1d(axes).ravel(), views, strict=True):
        axis.imshow(view)
        axis.axis("off")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "augmentation_grid.png", dpi=80)
    plt.close(fig)
    return {
        "status": "PASS" if ok else "FAIL",
        "detail": f"{len(views)} augmented views of {len(ids)} support images from "
        f"{learned['run_id']} are finite, non-blank and of the contract size; grid saved "
        "locally "
        "(contains dataset images; not committed)",
    }


def run_sanity_checks(paths: ProjectPaths, rows: list[dict[str, Any]], out_dir: Path):
    runs = [check_run(paths, row) for row in rows if row.get("_attempt_dir")]
    result = {"runs": runs, "augmentation": augmentation_check(paths, rows, out_dir)}
    atomic_write_json(Path(out_dir) / "sanity.json", result)
    return result
