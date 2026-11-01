"""One run, end to end: verify inputs, create an attempt, fit, write evidence, record status.

Every artifact is written atomically. A run is COMPLETED only after its predictions are written,
re-read and re-scored. Any exception, including an interruption, marks the attempt FAILED with
the failure details. A COMPLETED run is immutable. Resuming continues an interrupted run as a new
attempt from its saved model, optimizer and epoch state.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from fewshotlab.config import RunConfig, config_hash, load_run_config, resolved_yaml
from fewshotlab.data.datasets import ManifestImageDataset
from fewshotlab.data.manifest import load_class_map, read_manifest
from fewshotlab.data.registry import class_map_path, manifest_path, require_prepared
from fewshotlab.data.support import load_support, support_path
from fewshotlab.data.transforms import build_transform, spec_for
from fewshotlab.devices import resolve_device
from fewshotlab.engine import artifacts
from fewshotlab.engine.environment import configure_determinism, environment_report
from fewshotlab.engine.gate import require_sealed_plan
from fewshotlab.engine.ledger import append_event
from fewshotlab.engine.provenance import git_state, source_digest, write_source_snapshot
from fewshotlab.engine.trainer import (
    EVAL_BATCH_SIZE,
    RESUME_FORMAT,
    SupportSet,
    ValidationSet,
    fit_learned,
    fit_prototype_method,
    query_outputs,
    validate,
)
from fewshotlab.errors import ConfigError, DataValidationError, ProtocolRefusal
from fewshotlab.hashing import sha256_file
from fewshotlab.io_utils import atomic_write_json, atomic_write_text, read_json, utc_now
from fewshotlab.metrics.core import check_prediction_ids, classification_metrics
from fewshotlab.models.methods import build_model, get_method
from fewshotlab.models.prototypes import EPSILON
from fewshotlab.models.spec import (
    IncompatibleModelError,
    ModelSpec,
    load_linear,
    load_prototypes,
    save_linear,
    save_prototypes,
)
from fewshotlab.paths import ProjectPaths, get_paths, portable_path
from fewshotlab.reporting.plots import confusion_plot, learning_curves
from fewshotlab.seeds import SEED_DERIVATION, derive_seed
from fewshotlab.status import RunStatus

HISTORY_COLUMNS = (
    "epoch",
    "train_loss",
    "train_examples",
    "optimizer_steps",
    "val_loss",
    "val_macro_f1",
    "val_top1",
    "val_balanced_accuracy",
    "is_best",
    "patience_reference",
    "epochs_without_improvement",
    "fit_seconds",
    "val_seconds",
)


@dataclass
class RunResult:
    run_id: str
    attempt_id: str
    attempt_dir: Path
    status: str
    metrics: dict[str, Any]
    selected_epoch: int | None
    stop_reason: str
    timing: dict[str, Any]


def init_seed_for(train_seed: int) -> int:
    """Initialization stream: depends only on train_seed, so at a given seed every method starts
    from the same new-head initialization (and scratch from the same random encoder)."""
    return derive_seed("init", train_seed) & ((1 << 63) - 1)


def _topk(cfg: RunConfig) -> tuple[int, ...]:
    return (5,) if cfg.dataset == "flowers102" else ()


def _in_project(paths: ProjectPaths, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else paths.root / path).resolve()


def resolve_inputs(cfg: RunConfig, paths: ProjectPaths) -> dict[str, Any]:
    """Verify every input hash and build the support and validation sets (never test)."""
    record = require_prepared(paths, cfg.dataset)
    mpath = manifest_path(paths, cfg.dataset)
    if _in_project(paths, cfg.split_manifest) != mpath.resolve():
        raise ConfigError(f"split_manifest must point to {portable_path(mpath)}")
    manifest_sha = sha256_file(mpath)
    if manifest_sha != record["manifest"]["sha256"]:
        raise DataValidationError(f"{mpath.name} changed after preparation")
    spath = support_path(paths, cfg.dataset, cfg.shots, cfg.support_seed)
    if _in_project(paths, cfg.support_manifest) != spath.resolve():
        raise ConfigError(f"support_manifest must point to {portable_path(spath)}")
    doc = load_support(paths, cfg.dataset, cfg.shots, cfg.support_seed)
    if (doc["k"], doc["seed"], doc["dataset"]) != (cfg.shots, cfg.support_seed, cfg.dataset):
        raise ConfigError("the support manifest does not match the config's dataset, K and seed")
    cmap = load_class_map(class_map_path(paths, cfg.dataset))
    train = {r.image_id: r for r in read_manifest(mpath, splits=("train",))}
    selected = [i for key in sorted(doc["selected"], key=int) for i in doc["selected"][key]]
    validation = sorted(read_manifest(mpath, splits=("val",)), key=lambda r: r.image_id)
    return {
        "manifest_path": mpath,
        "manifest_sha256": manifest_sha,
        "support_doc": doc,
        "support_path": spath,
        "class_map": cmap,
        "support": SupportSet(tuple(train[i] for i in selected)),
        "validation": ValidationSet(tuple(validation)),
    }


def _method_params(cfg: RunConfig) -> dict[str, Any]:
    if get_method(cfg.method).learned:
        return {
            "loss": "cross_entropy",
            "precision": cfg.precision,
            "optimizer": asdict(cfg.optimizer),
            "train": asdict(cfg.train),
            "selection": asdict(cfg.selection),
        }
    return {
        "epsilon": EPSILON,
        "normalization": "per-embedding L2 (rejects norm <= epsilon)",
        "prototype": "mean of normalized support embeddings (not renormalized)",
        "distance": "squared Euclidean",
        "score": "softmax(-squared distance), scale 1",
        "ties": "lowest class ID",
    }


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=HISTORY_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(history)
    atomic_write_text(path, buffer.getvalue())


def run_training(
    cfg: RunConfig,
    *,
    paths: ProjectPaths | None = None,
    device_override: str | None = None,
    allow_download: bool = False,
    resume: bool = False,
    init_override: str | None = None,
    on_epoch_end: Callable[[int], None] | None = None,
    command: list[str] | None = None,
) -> RunResult:
    paths = paths or get_paths()
    if init_override is not None and not cfg.is_fixture:
        raise ConfigError("init_override is reserved for software fixtures")
    method = get_method(cfg.method)
    cfg_sha = config_hash(cfg)
    run_id = artifacts.make_run_id(cfg, cfg_sha)
    inputs = resolve_inputs(cfg, paths)
    existing = [(d, artifacts.read_status(d)) for d in artifacts.attempt_dirs(paths, run_id)]
    if any(s and s["status"] == RunStatus.COMPLETED for _, s in existing):
        raise ProtocolRefusal(
            f"{run_id} already has a COMPLETED attempt; finished results are "
            "immutable (invalidate it explicitly before re-running)"
        )
    resume_state, resumed_from = None, None
    if resume:
        if not method.learned:
            raise ConfigError("prototypes have no epochs to resume")
        candidates = [
            (d, s)
            for d, s in existing
            if s and s["status"] in ("FAILED", "RUNNING") and (d / "last.ckpt").is_file()
        ]
        if not candidates:
            raise ConfigError(f"nothing to resume for {run_id}")
        previous_dir = candidates[-1][0]
        resume_state = torch.load(previous_dir / "last.ckpt", map_location="cpu", weights_only=True)
        if (
            resume_state.get("format") != RESUME_FORMAT
            or resume_state.get("config_sha256") != cfg_sha
            or resume_state.get("support_sha256") != inputs["support_doc"]["content_sha256"]
        ):
            raise ConfigError(
                "the saved state does not match this config and support set; "
                "start a fresh run instead"
            )
        resumed_from = {"attempt_id": previous_dir.name, "epoch": int(resume_state["epoch"])}
    device = resolve_device(device_override or cfg.device)
    determinism = configure_determinism(device)
    attempt_id, adir = artifacts.new_attempt(paths, run_id)
    if resume_state is not None:  # carry the best checkpoint selected before the interruption
        best = json.loads(resume_state["best_json"])
        source = artifacts.run_root(paths, run_id) / resumed_from["attempt_id"] / "best.ckpt"
        if best is not None:
            if not source.is_file() or sha256_file(source) != best["checkpoint_sha256"]:
                raise DataValidationError(
                    "the interrupted attempt's best.ckpt is missing or altered; resume refused"
                )
            shutil.copy2(source, adir / "best.ckpt")
    cmap = inputs["class_map"]
    num_classes = cmap.num_classes
    atomic_write_text(adir / "config.resolved.yaml", resolved_yaml(cfg))
    atomic_write_json(adir / "environment.json", environment_report(device, determinism))
    code = git_state(paths.root)
    digest, n_files = source_digest(paths.root)
    code.update(source_sha256=digest, source_files=n_files)
    if code["dirty"] or code["commit"] is None:
        code["source_snapshot"] = {
            "file": "source_snapshot.tar.gz",
            "sha256": write_source_snapshot(paths.root, adir / "source_snapshot.tar.gz"),
        }
    locks = {
        name: (sha256_file(paths.root / name) if (paths.root / name).is_file() else None)
        for name in ("uv.lock", "requirements.lock")
    }
    provenance: dict[str, Any] = {
        "protocol_id": cfg.protocol_id,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "command": command or sys.argv,
        "started_utc": utc_now(),
        "code": code,
        "locks": locks,
        "data": {
            "dataset": cfg.dataset,
            "split_manifest": {
                "path": portable_path(inputs["manifest_path"]),
                "sha256": inputs["manifest_sha256"],
            },
            "class_map_sha256": cmap.sha256(),
            "support": {
                "path": portable_path(inputs["support_path"]),
                "k": cfg.shots,
                "seed": cfg.support_seed,
                "content_sha256": inputs["support_doc"]["content_sha256"],
                "n_images": len(inputs["support"].records),
            },
            "validation_images": len(inputs["validation"].records),
        },
        "config": {"file": "config.resolved.yaml", "sha256": cfg_sha},
        "seeds": {
            "support_seed": cfg.support_seed,
            "train_seed": cfg.train_seed,
            "init_seed": init_seed_for(cfg.train_seed),
            "derivation": SEED_DERIVATION,
            "streams": {
                "init": ["init", cfg.train_seed],
                "augmentation": ["augment", cfg.train_seed, "<epoch>", "<image_id>"],
                "loader_order": ["loader", cfg.train_seed, "<epoch>"],
            },
        },
        "device": device,
        "resumed_from": resumed_from,
        "status": RunStatus.RUNNING.value,
    }
    atomic_write_json(adir / "provenance.json", provenance)
    event = {
        "event": "status",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "protocol_id": cfg.protocol_id,
        "dataset": cfg.dataset,
        "method": cfg.method,
        "k": cfg.shots,
        "seed": cfg.support_seed,
        "category": cfg.category,
        "is_fixture": cfg.is_fixture,
        "config_sha256": cfg_sha,
        "run_dir": artifacts.logical_run_dir(paths, run_id, attempt_id),
    }
    artifacts.write_status(
        adir,
        RunStatus.RUNNING,
        run_id=run_id,
        attempt_id=attempt_id,
        category=cfg.category,
        is_fixture=cfg.is_fixture,
        started_utc=provenance["started_utc"],
        resumed_from=resumed_from,
    )
    append_event(paths, **event, status=RunStatus.RUNNING.value, resumed_from=resumed_from)
    try:
        run_started = time.perf_counter()
        init_seed = init_seed_for(cfg.train_seed)
        model, weights = build_model(
            cfg.method,
            num_classes,
            init_seed=init_seed,
            weights_dir=paths.weights_dir,
            allow_download=allow_download,
            init_override=init_override,
        )
        parameters = {
            "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "total": int(sum(p.numel() for p in model.parameters())),
        }
        eval_spec = spec_for(cfg.dataset, "eval")
        spec = ModelSpec(
            method=cfg.method,
            dataset=cfg.dataset,
            class_names=cmap.names,
            class_map_sha256=cmap.sha256(),
            eval_transform_id=eval_spec.transform_id,
            eval_transform=eval_spec.signature(),
            encoder={
                "architecture": "resnet18",
                **asdict(weights),
                "init_seed": init_seed,
                "init": init_override or method.init,
            },
            method_params=_method_params(cfg),
            support_sha256=inputs["support_doc"]["content_sha256"],
            run_id=run_id,
            is_fixture=cfg.is_fixture,
        )
        atomic_write_json(adir / "model_spec.json", spec.to_json())
        expected = {
            "dataset": cfg.dataset,
            "class_map_sha256": spec.class_map_sha256,
            "eval_transform_id": spec.eval_transform_id,
            "method": cfg.method,
        }
        if method.learned:
            result = fit_learned(
                model,
                cfg,
                inputs["support"],
                inputs["validation"],
                data_root=paths.data_root,
                device=device,
                num_classes=num_classes,
                save_best=lambda m: save_linear(adir / "best.ckpt", m, spec),
                reload_best=lambda: load_linear(adir / "best.ckpt", **expected)[0],
                last_path=adir / "last.ckpt",
                write_history=lambda h: _write_history(adir / "train_history.csv", h),
                identity={"config_sha256": cfg_sha, "support_sha256": spec.support_sha256},
                resume_state=resume_state,
                on_epoch_end=on_epoch_end,
            )
            artifact = {"file": "best.ckpt", "sha256": sha256_file(adir / "best.ckpt")}
        else:
            prototypes, result = fit_prototype_method(
                model,
                cfg,
                inputs["support"],
                inputs["validation"],
                data_root=paths.data_root,
                device=device,
                num_classes=num_classes,
            )
            artifact = {
                "file": "prototypes.npz",
                "sha256": save_prototypes(adir / "prototypes.npz", prototypes, spec),
            }
            if init_override is None:
                encoder, loaded, _ = load_prototypes(
                    adir / "prototypes.npz", weights_dir=paths.weights_dir, **expected
                )
            else:  # fixture encoders are rebuilt from their recorded random-init seed
                from fewshotlab.models.backbone import build_resnet18

                encoder, _ = build_resnet18(num_classes, init="random", init_seed=init_seed)
                loaded = load_prototypes(adir / "prototypes.npz", **expected)[1]
            again = query_outputs(
                encoder, loaded, inputs["validation"], cfg, data_root=paths.data_root, device=device
            )
            result.reload_check = {
                "identical_predictions": bool(np.array_equal(again.pred, result.outputs.pred)),
                "max_abs_score_difference": float(
                    np.max(np.abs(again.scores - result.outputs.scores))
                ),
            }
            if not result.reload_check["identical_predictions"]:
                raise RuntimeError("reloaded prototypes do not reproduce their predictions")
        validation = inputs["validation"].records
        ids = [validation[int(i)].image_id for i in result.outputs.indices]
        predictions_sha = artifacts.write_predictions(
            adir / "predictions_val.csv",
            image_ids=ids,
            split="val",
            true=result.outputs.labels,
            pred=result.outputs.pred,
            scores=result.outputs.scores,
            run_id=run_id,
            attempt_id=attempt_id,
            score_kind=method.score_kind,
        )
        stored = artifacts.read_predictions(adir / "predictions_val.csv")
        check_prediction_ids(stored["image_ids"], [r.image_id for r in validation])
        metrics = classification_metrics(
            stored["true"], stored["pred"], num_classes, scores=stored["scores"], topk=_topk(cfg)
        )
        val_doc = {
            "schema_version": 1,
            "status": RunStatus.COMPLETED.value,
            "split": "val",
            "note": "validation-only development result; not a held-out test benchmark",
            "run_id": run_id,
            "attempt_id": attempt_id,
            "protocol_id": cfg.protocol_id,
            "dataset": cfg.dataset,
            "method": cfg.method,
            "k": cfg.shots,
            "seed": cfg.support_seed,
            "category": cfg.category,
            "is_fixture": cfg.is_fixture,
            "class_map_sha256": cmap.sha256(),
            "label_budget": {
                "fit_labels": len(inputs["support"].records),
                "validation_labels": len(validation),
            },
            "selected_epoch": result.selected_epoch,
            "epochs_run": result.epochs_run,
            "stop_reason": result.stop_reason,
            "optimizer_steps": result.optimizer_steps,
            "val_loss": result.val_loss,
            "metrics": metrics,
            "predictions": {"file": "predictions_val.csv", "sha256": predictions_sha},
            "artifact": artifact,
            "reload_check": result.reload_check,
            "parameters": parameters,
            "timing": {
                **result.timing,
                "total_wall_seconds": time.perf_counter() - run_started,
                "val_pass_ms_per_image": 1000.0
                * (
                    result.timing["val_seconds_per_epoch"][-1]
                    if "val_seconds_per_epoch" in result.timing
                    else result.timing["val_seconds"]
                )
                / len(validation),
                "timing_note": "wall-clock on the recorded device; validation passes use batch "
                "size 64 and include image decoding; not a latency benchmark",
            },
            "memory": result.memory,
            "device": device,
            "computed_from": "predictions_val.csv via fewshotlab.metrics.core."
            "classification_metrics",
            "created_utc": utc_now(),
        }
        atomic_write_json(adir / "val_metrics.json", val_doc)
        title = (
            f"{cfg.dataset} {cfg.method} K={cfg.shots} seed={cfg.support_seed} "
            f"[{cfg.category}{', FIXTURE' if cfg.is_fixture else ''}] validation only"
        )
        if result.history:
            learning_curves(result.history, adir / "plots" / "learning_curves.png", title=title)
        confusion_plot(
            metrics["confusion_row_normalized"], adir / "plots" / "confusion_val.png", title=title
        )
        resume_state_file = adir / "last.ckpt"
        if resume_state_file.exists():  # only needed to resume an unfinished run
            resume_state_file.unlink()
        provenance.update(
            resume_state="last.ckpt deleted after completion (it is only needed to resume)",
            ended_utc=utc_now(),
            status=RunStatus.COMPLETED.value,
            weights=asdict(weights),
            artifacts={
                "model": artifact,
                "predictions_val": predictions_sha,
                "val_metrics": sha256_file(adir / "val_metrics.json"),
                "model_spec": sha256_file(adir / "model_spec.json"),
            },
        )
        atomic_write_json(adir / "provenance.json", provenance)
        artifacts.write_status(adir, RunStatus.COMPLETED, ended_utc=provenance["ended_utc"])
        append_event(
            paths,
            **event,
            status=RunStatus.COMPLETED.value,
            val_macro_f1=metrics["macro_f1"],
            selected_epoch=result.selected_epoch,
        )
        return RunResult(
            run_id,
            attempt_id,
            adir,
            RunStatus.COMPLETED.value,
            metrics,
            result.selected_epoch,
            result.stop_reason,
            result.timing,
        )
    except BaseException as exc:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc)[:2000],
            "traceback": traceback.format_exc()[-8000:],
        }
        artifacts.write_status(adir, RunStatus.FAILED, failure=failure, ended_utc=utc_now())
        provenance.update(status=RunStatus.FAILED.value, failure=failure, ended_utc=utc_now())
        atomic_write_json(adir / "provenance.json", provenance)
        append_event(
            paths,
            **event,
            status=RunStatus.FAILED.value,
            reason=f"{failure['type']}: {failure['message'][:300]}",
        )
        raise


def latest_completed_attempt(paths: ProjectPaths, run_id: str) -> Path:
    completed = [
        d
        for d in artifacts.attempt_dirs(paths, run_id)
        if (artifacts.read_status(d) or {}).get("status") == RunStatus.COMPLETED
    ]
    if not completed:
        raise ConfigError(f"no COMPLETED attempt found for run {run_id!r}")
    return completed[-1]


def evaluate_run(
    run_id: str,
    split: str,
    *,
    paths: ProjectPaths | None = None,
    device_override: str | None = None,
) -> dict[str, Any]:
    """Re-evaluate a stored run on validation (a verification, never a new result).

    The test split is refused until the P5 gate seals an evaluation plan.
    """
    paths = paths or get_paths()
    if split == "test":
        require_sealed_plan(paths)
    if split != "val":
        raise ConfigError("split must be 'val' or 'test'")
    adir = latest_completed_attempt(paths, run_id)
    cfg = load_run_config(adir / "config.resolved.yaml")
    spec = ModelSpec.from_json(read_json(adir / "model_spec.json"))
    if spec_for(cfg.dataset, "eval").transform_id != spec.eval_transform_id:
        raise IncompatibleModelError(
            "this code version's evaluation transform differs from the "
            "one the model was trained with"
        )
    inputs = resolve_inputs(cfg, paths)
    device = resolve_device(device_override or cfg.device)
    expected = {
        "dataset": cfg.dataset,
        "class_map_sha256": inputs["class_map"].sha256(),
        "eval_transform_id": spec.eval_transform_id,
        "method": cfg.method,
    }
    if get_method(cfg.method).learned:
        model, _ = load_linear(adir / "best.ckpt", **expected)
        loader = DataLoader(
            ManifestImageDataset(
                inputs["validation"].records, paths.data_root, build_transform(cfg.dataset, "eval")
            ),
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
        )
        outputs, _, _ = validate(
            model.to(device), loader, device, inputs["class_map"].num_classes, cfg.dataset
        )
    else:
        if spec.encoder.get("weights_id") == "none":
            from fewshotlab.models.backbone import build_resnet18

            encoder, _ = build_resnet18(
                spec.num_classes, init="random", init_seed=int(spec.encoder["init_seed"])
            )
            prototypes = load_prototypes(adir / "prototypes.npz", **expected)[1]
        else:
            encoder, prototypes, _ = load_prototypes(
                adir / "prototypes.npz", weights_dir=paths.weights_dir, **expected
            )
        outputs = query_outputs(
            encoder, prototypes, inputs["validation"], cfg, data_root=paths.data_root, device=device
        )
    stored = artifacts.read_predictions(adir / "predictions_val.csv")
    ids = [inputs["validation"].records[int(i)].image_id for i in outputs.indices]
    order = {image_id: i for i, image_id in enumerate(ids)}
    pred = np.array([outputs.pred[order[i]] for i in stored["image_ids"]])
    scores = np.stack([outputs.scores[order[i]] for i in stored["image_ids"]])
    recomputed = classification_metrics(stored["true"], pred, spec.num_classes)
    stored_doc = read_json(adir / "val_metrics.json")
    verification = {
        "kind": "verification: re-evaluation of an unchanged artifact (not a new result)",
        "run_id": run_id,
        "attempt_id": adir.name,
        "split": "val",
        "device": device,
        "time_utc": utc_now(),
        "identical_predictions": bool(np.array_equal(pred, stored["pred"])),
        "max_abs_score_difference": float(np.max(np.abs(scores - stored["scores"]))),
        "stored_macro_f1": stored_doc["metrics"]["macro_f1"],
        "recomputed_macro_f1": recomputed["macro_f1"],
    }
    folder = adir / "verifications"
    folder.mkdir(exist_ok=True)
    path = folder / f"val-{len(list(folder.glob('val-*.json'))) + 1:03d}.json"
    atomic_write_json(path, verification)
    append_event(
        paths,
        event="verification",
        run_id=run_id,
        attempt_id=adir.name,
        split="val",
        identical_predictions=verification["identical_predictions"],
        file=portable_path(path),
    )
    return verification
