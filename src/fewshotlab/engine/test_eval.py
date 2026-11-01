"""Sealed held-out test evaluation (P5). Every entry point first verifies the seal.

* Each sealed run is scored once on its dataset's full test split (clean images). Scoring uses
  the same inference code as validation. ``predictions_test.csv`` and ``test_metrics.json`` go
  into the sealed attempt folder, and one event goes into the study's evaluation ledger and the
  run ledger. A run with a recorded evaluation is never evaluated again.
* The robustness study scores the sealed demo checkpoint on the same test images under the seven
  sealed conditions (clean, plus three corruptions at two levels). Each derivative is paired with
  its clean parent, and the corruption metrics are kept apart from the primary benchmark.

The trainer and runner never import this module. Test images enter a model only through
:class:`GatedTestDataset`, which requires a :class:`~fewshotlab.engine.seal.VerifiedSeal`.
"""

from __future__ import annotations

import csv
import io
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from fewshotlab.data.imageio import open_image
from fewshotlab.data.manifest import ImageRecord, load_class_map, read_manifest
from fewshotlab.data.registry import class_map_path, manifest_path
from fewshotlab.data.transforms import ImageTransform, build_transform
from fewshotlab.devices import resolve_device
from fewshotlab.engine import artifacts
from fewshotlab.engine.inference import eval_loader, load_stored_model
from fewshotlab.engine.ledger import append_event
from fewshotlab.engine.seal import (
    ROBUSTNESS_DIR,
    VerifiedSeal,
    append_evaluation_event,
    read_evaluation_ledger,
    verify_seal,
)
from fewshotlab.engine.trainer import memory_snapshot, synchronize
from fewshotlab.errors import FewShotLabError, ProtocolRefusal
from fewshotlab.hashing import sha256_bytes, sha256_file, sha256_json
from fewshotlab.io_utils import atomic_write_bytes, atomic_write_json, utc_now
from fewshotlab.metrics.core import check_prediction_ids, classification_metrics
from fewshotlab.models.methods import get_method
from fewshotlab.paths import ProjectPaths, portable_path
from fewshotlab.simulation import corruptions, review

NOTE = (
    "held-out TEST result under a sealed plan; checkpoint, threshold and every other decision "
    "were fixed on validation before test access"
)


class GatedTestDataset(Dataset):
    """Held-out test images. Constructible only with a verified seal.

    With ``corruption=(condition, severity)`` each image is decoded to [0, 1], corrupted at
    native resolution and passed through the same evaluation contract. The derivative records
    (seed and pixel checksum) are kept in ``derivatives`` (use ``num_workers=0`` to read them).
    """

    def __init__(
        self,
        records: list[ImageRecord],
        data_root: str | Path,
        transform: ImageTransform,
        *,
        seal: VerifiedSeal,
        corruption: tuple[str, int] | None = None,
    ) -> None:
        if not isinstance(seal, VerifiedSeal):
            raise ProtocolRefusal("test images can only be loaded with a verified seal")
        if any(r.split != "test" for r in records):
            raise ValueError("GatedTestDataset holds held-out test records only")
        if transform.spec.stage != "eval":
            raise ValueError("test images use the deterministic evaluation contract")
        self.records = list(records)
        self.data_root = Path(data_root)
        self.transform = transform
        self.corruption = corruption
        self.derivatives: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        record = self.records[index]
        path = self.data_root / record.relative_path
        if self.corruption is None:
            return self.transform(open_image(path)), record.class_id, index
        condition, severity = self.corruption
        derived, info = corruptions.corrupt(
            corruptions.decode01(path), condition, severity, record.image_id
        )
        self.derivatives[index] = info
        return corruptions.to_model_input(derived, self.transform), record.class_id, index


def _device(seal: VerifiedSeal, override: str | None) -> str:
    sealed = seal.plan["content"]["environment"]["device"]
    device = resolve_device(override) if override else sealed
    if device != sealed:
        raise ProtocolRefusal(f"the plan was sealed for device {sealed!r}, not {device!r}")
    return device


def _test_records(seal: VerifiedSeal, paths: ProjectPaths, dataset: str) -> list[ImageRecord]:
    records = sorted(
        read_manifest(manifest_path(paths, dataset), splits=("test",)), key=lambda r: r.image_id
    )
    expected = seal.plan["content"]["data"][dataset]["expected_test_ids"]
    if (
        len(records) != expected["count"]
        or sha256_json([r.image_id for r in records]) != (expected["sha256"])
    ):
        raise ProtocolRefusal(f"{dataset} test IDs differ from the sealed list")
    return records


def _score(model, dataset: GatedTestDataset, workers: int, device: str):
    synchronize(device)
    start = time.perf_counter()
    outputs = model.score(eval_loader(dataset, workers))
    synchronize(device)
    seconds = time.perf_counter() - start
    ids = [dataset.records[int(i)].image_id for i in outputs.indices]
    check_prediction_ids(ids, [r.image_id for r in dataset.records])
    return outputs, ids, seconds


def _topk(dataset: str) -> tuple[int, ...]:
    return (5,) if dataset == "flowers102" else ()


def _recorded(seal: VerifiedSeal, event: str) -> set[tuple[str, str]]:
    return {
        (e["run_id"], e["attempt_id"])
        for e in read_evaluation_ledger(seal.directory)
        if e.get("event") == event and e.get("audit_token") == seal.audit_token
    }


def evaluate_sealed_run(
    seal: VerifiedSeal, run: dict[str, Any], paths: ProjectPaths, device: str
) -> dict[str, Any]:
    adir = artifacts.run_root(paths, run["run_id"]) / run["attempt_id"]
    for name in ("predictions_test.csv", "test_metrics.json"):
        if (adir / name).exists():
            raise ProtocolRefusal(
                f"{portable_path(adir / name)} exists without a recorded evaluation event; "
                "refusing to overwrite it (possible unrecorded test exposure)"
            )
    dataset = run["dataset"]
    records = _test_records(seal, paths, dataset)
    cmap = load_class_map(class_map_path(paths, dataset))
    model = load_stored_model(adir, paths, cmap, device)
    data = GatedTestDataset(records, paths.data_root, build_transform(dataset, "eval"), seal=seal)
    outputs, ids, seconds = _score(model, data, model.cfg.train.num_workers, device)
    predictions_sha = artifacts.write_predictions(
        adir / "predictions_test.csv",
        image_ids=ids,
        split="test",
        true=outputs.labels,
        pred=outputs.pred,
        scores=outputs.scores,
        run_id=run["run_id"],
        attempt_id=run["attempt_id"],
        score_kind=get_method(run["method"]).score_kind,
    )
    stored = artifacts.read_predictions(adir / "predictions_test.csv")
    truth = {r.image_id: r.class_id for r in records}
    if [truth[i] for i in stored["image_ids"]] != stored["true"].tolist():
        raise FewShotLabError("stored test labels do not match the split manifest")
    metrics = classification_metrics(
        stored["true"],
        stored["pred"],
        cmap.num_classes,
        scores=stored["scores"],
        topk=_topk(dataset),
    )
    doc = {
        "schema_version": 1,
        "split": "test",
        "note": NOTE,
        "study_id": seal.study_id,
        "audit_token": seal.audit_token,
        "plan_sha256": seal.plan_sha256,
        "run_id": run["run_id"],
        "attempt_id": run["attempt_id"],
        "dataset": dataset,
        "method": run["method"],
        "k": run["k"],
        "seed": run["seed"],
        "category": run["category"],
        "is_fixture": run["is_fixture"],
        "device": device,
        "created_utc": utc_now(),
        "computed_from": "predictions_test.csv via fewshotlab.metrics.core.classification_metrics",
        "predictions": {"file": "predictions_test.csv", "sha256": predictions_sha},
        "metrics": metrics,
        "timing": {
            "seconds": seconds,
            "ms_per_image": 1000.0 * seconds / len(ids),
            "note": "wall clock for decode, preprocessing and inference (batch 64), "
            "accelerator synchronized; not a latency benchmark",
        },
        "memory": memory_snapshot(device),
    }
    atomic_write_json(adir / "test_metrics.json", doc)
    files = {
        "predictions_test.csv": predictions_sha,
        "test_metrics.json": sha256_file(adir / "test_metrics.json"),
    }
    common = {"run_id": run["run_id"], "attempt_id": run["attempt_id"], "files": files}
    append_evaluation_event(seal, event="test_evaluation", n_images=len(ids), **common)
    append_event(
        paths,
        event="test_evaluation",
        study_id=seal.study_id,
        audit_token=seal.audit_token,
        split="test",
        **common,
    )
    return {"run_id": run["run_id"], "outcome": "evaluated", "macro_f1": metrics["macro_f1"]}


def _checksum_csv(rows: list[list[Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["parent_image_id", "condition", "seed", "derivative_sha256"])
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def evaluate_robustness(seal: VerifiedSeal, paths: ProjectPaths, device: str) -> dict[str, Any]:
    content = seal.plan["content"]
    demo = content["demo"]["selected"]
    threshold = content["demo"]["threshold"]["selected"]["threshold"]
    dataset = content["demo"]["target"]["dataset"]
    adir = artifacts.run_root(paths, demo["run_id"]) / demo["attempt_id"]
    out = adir / ROBUSTNESS_DIR
    if out.exists() and any(out.iterdir()):
        raise ProtocolRefusal(f"{portable_path(out)} already holds unrecorded outputs")
    records = _test_records(seal, paths, dataset)
    cmap = load_class_map(class_map_path(paths, dataset))
    model = load_stored_model(adir, paths, cmap, device)
    transform = build_transform(dataset, "eval")
    truth = {r.image_id: r.class_id for r in records}
    conditions, predictions, files, checksums = [], {}, {}, []
    for condition, severity in corruptions.CONDITIONS:
        cid = corruptions.condition_id(condition, severity)
        data = GatedTestDataset(
            records, paths.data_root, transform, seal=seal, corruption=(condition, severity)
        )
        outputs, ids, seconds = _score(model, data, 0, device)
        name = f"predictions_test__{cid}.csv"
        files[f"{ROBUSTNESS_DIR}/{name}"] = artifacts.write_predictions(
            out / name,
            image_ids=ids,
            split="test",
            true=outputs.labels,
            pred=outputs.pred,
            scores=outputs.scores,
            run_id=demo["run_id"],
            attempt_id=demo["attempt_id"],
            score_kind=get_method(demo["method"]).score_kind,
            condition=cid,
        )
        stored = artifacts.read_predictions(out / name)
        metrics = classification_metrics(stored["true"], stored["pred"], cmap.num_classes)
        routing = review.review_metrics(
            stored["true"], stored["pred"], stored["scores"].max(axis=1), threshold
        )
        predictions[cid] = dict(zip(stored["image_ids"], stored["pred"].tolist(), strict=True))
        checksums += [
            [
                records[i].image_id,
                cid,
                "" if data.derivatives[i]["seed"] is None else data.derivatives[i]["seed"],
                data.derivatives[i]["checksum_sha256"],
            ]
            for i in range(len(records))
        ]
        conditions.append(
            {
                "condition_id": cid,
                "condition": condition,
                "severity": severity,
                "parameters": corruptions.PARAMETERS[(condition, severity)],
                "n_images": len(ids),
                "macro_f1": metrics["macro_f1"],
                "top1": metrics["top1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "per_class": metrics["per_class"],
                "confusion": metrics["confusion"],
                "review": routing,
                "seconds": seconds,
            }
        )
    clean_row = conditions[0]
    clean = predictions["clean"]
    paired = []
    for row in conditions[1:]:
        now = predictions[row["condition_id"]]
        paired.append(
            {
                "condition_id": row["condition_id"],
                "n_parents": len(clean),
                "delta_macro_f1": row["macro_f1"] - clean_row["macro_f1"],
                "delta_top1": row["top1"] - clean_row["top1"],
                "delta_coverage": row["review"]["coverage"] - clean_row["review"]["coverage"],
                "changed_predictions": sum(now[i] != clean[i] for i in clean),
                "clean_correct_now_wrong": sum(
                    clean[i] == truth[i] and now[i] != truth[i] for i in clean
                ),
                "clean_wrong_now_correct": sum(
                    clean[i] != truth[i] and now[i] == truth[i] for i in clean
                ),
            }
        )
    primary = adir / "predictions_test.csv"
    consistency = None
    if primary.is_file():
        base = artifacts.read_predictions(primary)
        consistency = {
            "compared_with": "predictions_test.csv (primary clean evaluation of this run)",
            "identical_predictions": dict(
                zip(base["image_ids"], base["pred"].tolist(), strict=True)
            )
            == clean,
        }
    checksum_bytes = _checksum_csv(checksums)
    atomic_write_bytes(out / "corruption_checksums.csv", checksum_bytes)
    files[f"{ROBUSTNESS_DIR}/corruption_checksums.csv"] = sha256_bytes(checksum_bytes)
    doc = {
        "schema_version": 1,
        "split": "test",
        "note": "ROBUSTNESS STUDY on the sealed demo checkpoint: controlled synthetic corruptions "
        "of held-out test images; reported apart from the primary benchmark. Derivatives are "
        "paired with their clean parent and are not new test images.",
        "study_id": seal.study_id,
        "audit_token": seal.audit_token,
        "plan_sha256": seal.plan_sha256,
        "run_id": demo["run_id"],
        "attempt_id": demo["attempt_id"],
        "method": demo["method"],
        "dataset": dataset,
        "threshold": threshold,
        "device": device,
        "created_utc": utc_now(),
        "corruption_implementation": corruptions.IMPLEMENTATION,
        "conditions": conditions,
        "paired_vs_clean": paired,
        "clean_path_consistency": consistency,
        "files": dict(sorted(files.items())),
    }
    atomic_write_json(out / "robustness_test.json", doc)
    files[f"{ROBUSTNESS_DIR}/robustness_test.json"] = sha256_file(out / "robustness_test.json")
    common = {"run_id": demo["run_id"], "attempt_id": demo["attempt_id"], "files": files}
    append_evaluation_event(seal, event="robustness_evaluation", **common)
    append_event(
        paths,
        event="robustness_evaluation",
        study_id=seal.study_id,
        audit_token=seal.audit_token,
        split="test",
        **common,
    )
    return {"run_id": demo["run_id"], "outcome": "evaluated", "conditions": len(conditions)}


def run_test_evaluation(
    study: dict[str, Any],
    paths: ProjectPaths,
    *,
    device_override: str | None = None,
    only_runs: set[str] | None = None,
    robustness: bool = True,
) -> dict[str, Any]:
    """``evaluate --split test --study PATH``: verify the seal, then evaluate what is pending."""
    seal = verify_seal(study, paths)
    device = _device(seal, device_override)
    content = seal.plan["content"]
    sealed_ids = {r["run_id"] for r in content["runs"]}
    if only_runs and only_runs - sealed_ids:
        raise ProtocolRefusal(f"not in the sealed plan: {sorted(only_runs - sealed_ids)}")
    done = _recorded(seal, "test_evaluation")
    results = []
    for run in content["runs"]:
        if only_runs and run["run_id"] not in only_runs:
            continue
        if (run["run_id"], run["attempt_id"]) in done:
            results.append({"run_id": run["run_id"], "outcome": "already evaluated"})
            continue
        results.append(evaluate_sealed_run(seal, run, paths, device))
    stress = None
    if robustness and not only_runs and content["demo"]:
        demo = content["demo"]["selected"]
        if (demo["run_id"], demo["attempt_id"]) in _recorded(seal, "robustness_evaluation"):
            stress = {"run_id": demo["run_id"], "outcome": "already evaluated"}
        else:
            stress = evaluate_robustness(seal, paths, device)
    return {
        "study_id": seal.study_id,
        "audit_token": seal.audit_token,
        "device": device,
        "runs": results,
        "robustness": stress,
    }
