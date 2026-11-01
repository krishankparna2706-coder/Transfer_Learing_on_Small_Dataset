"""Fitting and validation for the four methods (protocol v1 recipe).

The fit functions accept only a :class:`SupportSet` (training-split images) and a
:class:`ValidationSet` (validation-split images). There is no parameter or code path through which
held-out images could enter fitting or model selection.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fewshotlab.config import RunConfig
from fewshotlab.data.datasets import ManifestImageDataset
from fewshotlab.data.manifest import ImageRecord
from fewshotlab.data.transforms import build_transform
from fewshotlab.engine.batching import NoSingletonBatchSampler
from fewshotlab.errors import FewShotLabError, ProtocolRefusal
from fewshotlab.metrics.core import classification_metrics
from fewshotlab.models.backbone import ResNet18Classifier
from fewshotlab.models.methods import (
    Outputs,
    collect_embeddings,
    collect_logits,
    prototype_outputs,
)
from fewshotlab.models.policies import check_train_mode, enter_train_mode, optimizer_groups
from fewshotlab.models.prototypes import fit_prototypes

EVAL_BATCH_SIZE = 64
RESUME_FORMAT = "fewshotlab-resume-v1"


class NonFiniteError(FewShotLabError):
    """A loss, gradient step or parameter became non-finite; the run is marked FAILED."""


@dataclass(frozen=True)
class SupportSet:
    records: tuple[ImageRecord, ...]

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("empty support set")
        foreign = [r.image_id for r in self.records if r.split != "train"]
        if foreign:
            raise ProtocolRefusal(f"support images must come from the train split: {foreign[:3]}")


@dataclass(frozen=True)
class ValidationSet:
    records: tuple[ImageRecord, ...]

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("empty validation set")
        foreign = [r.image_id for r in self.records if r.split != "val"]
        if foreign:
            raise ProtocolRefusal(f"validation images must come from the val split: {foreign[:3]}")


@dataclass
class FitResult:
    outputs: Outputs  # validation outputs of the selected model, after reloading it
    metrics: dict[str, Any]
    val_loss: float | None
    selected_epoch: int | None
    epochs_run: int
    stop_reason: str
    optimizer_steps: int
    history: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    reload_check: dict[str, Any] = field(default_factory=dict)


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def memory_snapshot(device: str) -> dict[str, Any]:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    info: dict[str, Any] = {
        "process_peak_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024)
    }
    if device == "mps":
        info["mps_current_allocated_bytes"] = int(torch.mps.current_allocated_memory())
        info["mps_driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
    elif device == "cuda":
        info["cuda_max_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    return info


def _loader(dataset: ManifestImageDataset, workers: int, **kwargs: Any) -> DataLoader:
    return DataLoader(dataset, num_workers=workers, persistent_workers=False, **kwargs)


def _topk(dataset: str, num_classes: int) -> tuple[int, ...]:
    return (5,) if dataset == "flowers102" and num_classes >= 5 else ()


def validate(
    model: ResNet18Classifier, loader: DataLoader, device: str, num_classes: int, dataset: str
) -> tuple[Outputs, float, dict[str, Any]]:
    """Validation pass without gradients; sample-weighted cross-entropy in float64."""
    outputs = collect_logits(model, loader, device)
    logits = torch.from_numpy(outputs.raw).double()
    labels = torch.from_numpy(outputs.labels).long()
    loss = float(F.cross_entropy(logits, labels, reduction="sum") / len(labels))
    metrics = classification_metrics(
        outputs.labels,
        outputs.pred,
        num_classes,
        scores=outputs.scores,
        topk=_topk(dataset, num_classes),
    )
    return outputs, loss, metrics


def fit_learned(
    model: ResNet18Classifier,
    cfg: RunConfig,
    support: SupportSet,
    validation: ValidationSet,
    *,
    data_root: Path,
    device: str,
    num_classes: int,
    save_best: Callable[[ResNet18Classifier], str],
    reload_best: Callable[[], ResNet18Classifier],
    last_path: Path,
    write_history: Callable[[list[dict[str, Any]]], None],
    identity: dict[str, str],
    resume_state: dict[str, Any] | None = None,
    on_epoch_end: Callable[[int], None] | None = None,
) -> FitResult:
    method, seed = cfg.method, cfg.train_seed
    train_ds = ManifestImageDataset(
        support.records,
        data_root,
        build_transform(cfg.dataset, "train"),
        augment=True,
        augmentation_labels=("augment", seed),
    )
    sampler = NoSingletonBatchSampler(
        len(train_ds), cfg.train.batch_size_cap, seed_labels=("loader", seed)
    )
    train_loader = _loader(train_ds, cfg.train.num_workers, batch_sampler=sampler)
    val_ds = ManifestImageDataset(
        validation.records, data_root, build_transform(cfg.dataset, "eval")
    )
    val_loader = _loader(val_ds, cfg.train.num_workers, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    model.to(device)
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, method, cfg.optimizer),
        weight_decay=cfg.optimizer.weight_decay,
        betas=tuple(cfg.optimizer.betas),
        eps=cfg.optimizer.eps,
    )
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    reference, stale, start_epoch, steps = float("-inf"), 0, 1, 0
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        history = json.loads(resume_state["history_json"])
        best = json.loads(resume_state["best_json"])
        reference = float(resume_state["patience_reference"])
        stale = int(resume_state["patience_counter"])
        steps = int(resume_state["optimizer_steps"])
        start_epoch = int(resume_state["epoch"]) + 1
    best_outputs: Outputs | None = None
    fit_seconds: list[float] = []
    val_seconds: list[float] = []
    stop_reason = "max_epochs"
    epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg.train.max_epochs + 1):
        train_ds.set_epoch(epoch)
        sampler.set_epoch(epoch)
        enter_train_mode(model, method)
        check_train_mode(model, method)
        synchronize(device)
        started = time.perf_counter()
        loss_sum, seen = 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            if not bool(torch.isfinite(loss)):
                raise NonFiniteError(f"non-finite training loss at epoch {epoch}, step {steps + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
            loss_sum += float(loss.detach()) * len(y)
            seen += len(y)
        synchronize(device)
        fit_time = time.perf_counter() - started
        if seen != len(train_ds):
            raise RuntimeError(f"epoch {epoch} used {seen} of {len(train_ds)} support images")
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise NonFiniteError(f"non-finite parameters after epoch {epoch}")
        started = time.perf_counter()
        outputs, val_loss, metrics = validate(model, val_loader, device, num_classes, cfg.dataset)
        synchronize(device)
        val_time = time.perf_counter() - started
        f1 = metrics["macro_f1"]
        is_best = (
            best is None
            or f1 > best["macro_f1"]
            or (f1 == best["macro_f1"] and val_loss < best["val_loss"])
        )
        if is_best:
            best = {
                "epoch": epoch,
                "macro_f1": f1,
                "val_loss": val_loss,
                "checkpoint_sha256": save_best(model),
            }
            best_outputs = outputs
        if f1 >= reference + cfg.selection.patience_min_delta:
            reference, stale = f1, 0
        else:
            stale += 1
        fit_seconds.append(fit_time)
        val_seconds.append(val_time)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / seen,
                "train_examples": seen,
                "optimizer_steps": steps,
                "val_loss": val_loss,
                "val_macro_f1": f1,
                "val_top1": metrics["top1"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "is_best": is_best,
                "patience_reference": reference,
                "epochs_without_improvement": stale,
                "fit_seconds": fit_time,
                "val_seconds": val_time,
            }
        )
        write_history(history)
        torch.save(
            {
                "format": RESUME_FORMAT,
                "epoch": epoch,
                **identity,
                "model_state": {k: v.detach().to("cpu") for k, v in model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(),
                "history_json": json.dumps(history),
                "best_json": json.dumps(best),
                "patience_reference": reference,
                "patience_counter": stale,
                "optimizer_steps": steps,
            },
            last_path,
        )
        if on_epoch_end is not None:
            on_epoch_end(epoch)
        if stale >= cfg.train.patience:
            stop_reason = "patience"
            break
    assert best is not None
    selected = reload_best().to(device)
    reloaded, reloaded_loss, reloaded_metrics = validate(
        selected, val_loader, device, num_classes, cfg.dataset
    )
    reload_check: dict[str, Any] = {
        "selected_epoch": best["epoch"],
        "selection_macro_f1": best["macro_f1"],
        "reloaded_macro_f1": reloaded_metrics["macro_f1"],
    }
    if best_outputs is not None:  # in-process selection outputs are available (not resumed)
        reload_check["identical_predictions"] = bool(
            np.array_equal(best_outputs.pred, reloaded.pred)
        )
        reload_check["max_abs_score_difference"] = float(
            np.max(np.abs(best_outputs.scores - reloaded.scores))
        )
    if reloaded_metrics["macro_f1"] != best["macro_f1"]:
        raise RuntimeError(
            f"the reloaded checkpoint does not reproduce its selection metric: {reload_check}"
        )
    return FitResult(
        outputs=reloaded,
        metrics=reloaded_metrics,
        val_loss=reloaded_loss,
        selected_epoch=best["epoch"],
        epochs_run=epoch,
        stop_reason=stop_reason,
        optimizer_steps=steps,
        history=history,
        timing={
            "fit_seconds_per_epoch": fit_seconds,
            "val_seconds_per_epoch": val_seconds,
            "resumed_from_epoch": start_epoch - 1 if resume_state else None,
        },
        memory=memory_snapshot(device),
        reload_check=reload_check,
    )


def fit_prototype_method(
    encoder: ResNet18Classifier,
    cfg: RunConfig,
    support: SupportSet,
    validation: ValidationSet,
    *,
    data_root: Path,
    device: str,
    num_classes: int,
) -> tuple[torch.Tensor, FitResult]:
    """Class means of normalized support embeddings (no optimizer, no epochs)."""
    transform = build_transform(cfg.dataset, "eval")  # deterministic for support and queries
    support_loader = _loader(
        ManifestImageDataset(support.records, data_root, transform),
        cfg.train.num_workers,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
    )
    val_loader = _loader(
        ManifestImageDataset(validation.records, data_root, transform),
        cfg.train.num_workers,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
    )
    encoder.to(device)
    synchronize(device)
    started = time.perf_counter()
    embeddings, labels, indices = collect_embeddings(encoder, support_loader, device)
    ids = [support.records[int(i)].image_id for i in indices]
    prototypes = fit_prototypes(embeddings, labels, num_classes, ids)
    synchronize(device)
    fit_time = time.perf_counter() - started
    started = time.perf_counter()
    query, qlabels, qindex = collect_embeddings(encoder, val_loader, device)
    outputs = prototype_outputs(
        prototypes, query, qlabels, qindex, [validation.records[int(i)].image_id for i in qindex]
    )
    metrics = classification_metrics(
        outputs.labels,
        outputs.pred,
        num_classes,
        scores=outputs.scores,
        topk=_topk(cfg.dataset, num_classes),
    )
    synchronize(device)
    val_time = time.perf_counter() - started
    return prototypes, FitResult(
        outputs=outputs,
        metrics=metrics,
        val_loss=None,
        selected_epoch=None,
        epochs_run=0,
        stop_reason="not applicable (class means; no epochs)",
        optimizer_steps=0,
        timing={"embedding_fit_seconds": fit_time, "val_seconds": val_time},
        memory=memory_snapshot(device),
    )


def query_outputs(
    encoder: ResNet18Classifier,
    prototypes: torch.Tensor,
    validation: ValidationSet,
    cfg: RunConfig,
    *,
    data_root: Path,
    device: str,
) -> Outputs:
    loader = _loader(
        ManifestImageDataset(validation.records, data_root, build_transform(cfg.dataset, "eval")),
        cfg.train.num_workers,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
    )
    encoder.to(device)
    query, labels, index = collect_embeddings(encoder, loader, device)
    return prototype_outputs(prototypes, query, labels, index)
