"""The run registry the app reads: verified runs, regenerated metrics and sealed identities.

- **Numbers.** Benchmark numbers come only from ``benchmark_report.slot_values``. It regenerates
  every metric from the stored prediction files and checks them against the run records
  (validation) or the sealed evaluation ledger (test). A missing value stays ``None``, shown as
  NOT RUN, never as zero.
- **Models.** A model is identified through ``engine.predict.model_ref`` before anything loads.
- **Caching.** Everything returned is plain data (dicts, lists, strings), so a UI can cache it
  under ``fingerprint``. The fingerprint changes whenever a ledger, the sealed plan or the study
  changes.
"""

from __future__ import annotations

from typing import Any

from fewshotlab.data.manifest import load_class_map, per_class_counts, read_manifest, split_counts
from fewshotlab.data.registry import class_map_path, manifest_path
from fewshotlab.engine.predict import ModelRef, ModelUnavailable, model_ref
from fewshotlab.engine.seal import EVALUATION_LEDGER, SEALED_PLAN, _sealed_studies, seal_dir
from fewshotlab.hashing import sha256_file, sha256_json
from fewshotlab.io_utils import read_json, utc_now
from fewshotlab.paths import ProjectPaths, portable_path
from fewshotlab.reporting.benchmark_report import label_budgets, slot_values

SPLIT_LABELS = {
    "val": "VALIDATION (development; checkpoints were selected on these images)",
    "test": "HELD-OUT TEST (sealed plan; each run evaluated once)",
}


def fingerprint(study: dict[str, Any], paths: ProjectPaths) -> str:
    """Changes whenever anything the registry reads from the run or seal records changes."""
    directory = seal_dir(paths, study["study_id"])
    parts: dict[str, Any] = {"study": study}
    for path in (
        paths.runs / "ledger.jsonl",
        directory / EVALUATION_LEDGER,
        directory / SEALED_PLAN,
    ):
        parts[path.name] = sha256_file(path) if path.is_file() else None
    return sha256_json(parts)


def seal_summary(study: dict[str, Any], paths: ProjectPaths) -> dict[str, Any] | None:
    """The study's intact sealed plan in brief, or ``None`` when it is not sealed."""
    sealed = _sealed_studies(paths).get(study["study_id"])
    if sealed is None:
        return None
    plan, events = sealed["plan"], sealed["events"]
    content, token = plan["content"], plan["audit_token"]
    mine = [e for e in events if e.get("audit_token") == token]
    tests = [e for e in mine if e.get("event") == "test_evaluation"]
    robustness = [e for e in mine if e.get("event") == "robustness_evaluation"]
    reproduction = plan["attestations"].get("validation_reproduction", {})
    return {
        "study_id": study["study_id"],
        "audit_token": token,
        "plan_sha256": plan["plan_sha256"],
        "sealed_utc": plan["sealed_utc"],
        "partial": bool(plan["partial"]),
        "plan_file": portable_path(seal_dir(paths, study["study_id"]) / SEALED_PLAN),
        "git": plan["attestations"].get("git", {}),
        "validation_reproduction": {k: v for k, v in reproduction.items() if k != "results"},
        "code": {k: content["code"][k] for k in ("scope", "source_sha256", "source_files")},
        "protocol_modules": content["code"]["protocol_modules"],
        "environment": content["environment"],
        "data": content["data"],
        "weights": content["weights"],
        "metrics": content["metrics"],
        "calibration": content["calibration"],
        "demo": content.get("demo"),
        "replay": {k: v for k, v in (content.get("replay") or {}).items() if k != "csv"},
        "corruptions": content.get("corruptions"),
        "runs": len(content["runs"]),
        "duplicates": len(content["duplicates"]),
        "missing": len(content["missing"]),
        "artifacts": {r["run_id"]: r["artifact"]["sha256"] for r in content["runs"]},
        "test_evaluations": len(tests),
        "test_window_utc": (
            [min(e["time_utc"] for e in tests), max(e["time_utc"] for e in tests)]
            if tests
            else None
        ),
        "robustness_utc": robustness[0]["time_utc"] if robustness else None,
    }


def dataset_facts(paths: ProjectPaths, dataset: str) -> dict[str, Any]:
    """Split counts, class map, source, split rule and audit status for one prepared dataset."""
    out: dict[str, Any] = {"dataset": dataset}
    try:
        cmap = load_class_map(class_map_path(paths, dataset))
        mpath = manifest_path(paths, dataset)
        records = read_manifest(mpath, splits=("train", "val", "test"))
    except (OSError, ValueError) as exc:
        return {**out, "status": "NOT PREPARED", "problem": str(exc)}
    per_class = {}
    for split in ("train", "val", "test"):
        counts = list(per_class_counts(records, split).values())
        per_class[split] = [min(counts), max(counts)] if counts else None
    prepared_path = paths.manifests / f"{dataset}.prepare.json"
    prepared = read_json(prepared_path) if prepared_path.is_file() else {}
    audit_path = paths.reports / "data_audit" / dataset / "data_audit.json"
    audit = read_json(audit_path) if audit_path.is_file() else {}
    return {
        **out,
        "status": "PREPARED",
        "classes": cmap.num_classes,
        "class_names": list(cmap.names),
        # Readable names for display only (Flowers: a secondary source); identity stays the ID.
        "display_names": [
            str(cmap.extra.get(name, {}).get("display_name") or name) for name in cmap.names
        ],
        "class_map_sha256": cmap.sha256(),
        "manifest": {"file": mpath.name, "sha256": sha256_file(mpath)},
        "counts": split_counts(records),
        "per_class_range": per_class,
        "source": prepared.get("source"),
        "split": prepared.get("split"),
        "terms": prepared.get("terms"),
        "prepared_utc": prepared.get("prepared_utc"),
        "audit": {
            "status": audit.get("status", "NOT RUN"),
            "generated_utc": audit.get("generated_utc"),
            "file": portable_path(audit_path) if audit_path.is_file() else None,
        },
    }


def load_registry(study: dict[str, Any], paths: ProjectPaths) -> dict[str, Any]:
    """Every number and identity the app shows for one study, regenerated from stored records."""
    return {
        "study": study,
        "fingerprint": fingerprint(study, paths),
        "generated_utc": utc_now(),
        "budgets": label_budgets(paths, study),
        "rows": {split: slot_values(study, paths, split) for split in ("val", "test")},
        "seal": seal_summary(study, paths),
        "datasets": {d: dataset_facts(paths, d) for d in study["datasets"]},
    }


def default_cell(registry: dict[str, Any], dataset: str) -> tuple[str, str, int, int] | None:
    """The cell the app opens first: the largest K, its first seed, then the method with the
    highest validation macro-F1 (exact ties to the lexically first name, as for the demo)."""
    rows = [
        r
        for r in registry["rows"]["val"]
        if r["dataset"] == dataset and r["status"] == "COMPLETED" and r["macro_f1"] is not None
    ]
    if not rows:
        return None
    k = max(r["k"] for r in rows)
    seed = min(r["seed"] for r in rows if r["k"] == k)
    best = min(
        (r for r in rows if r["k"] == k and r["seed"] == seed),
        key=lambda r: (-r["macro_f1"], r["method"]),
    )
    return dataset, best["method"], k, seed


def cell(registry: dict[str, Any], dataset: str, method: str, k: int, seed: int) -> dict | None:
    """The validation row of one planned slot (None when the study does not plan it)."""
    for row in registry["rows"]["val"]:
        if (row["dataset"], row["method"], row["k"], row["seed"]) == (dataset, method, k, seed):
            return row
    return None


def cell_model(
    paths: ProjectPaths, registry: dict[str, Any], dataset: str, method: str, k: int, seed: int
) -> ModelRef:
    """Identify the stored model behind one planned slot, or explain why it cannot be used.

    A deterministic duplicate slot resolves to the run it duplicates. In a sealed study, the
    checkpoint must also match the hash in the sealed plan.
    """
    row = cell(registry, dataset, method, k, seed)
    if row is None:
        study_id = registry["study"]["study_id"]
        raise ModelUnavailable(f"{dataset} {method} K={k} seed {seed} is not part of {study_id}")
    if row["status"] != "COMPLETED":
        raise ModelUnavailable(
            f"{dataset} {method} K={k} seed {seed} is {row['status']}: no verified model exists "
            "for this cell (train it with `python -m fewshotlab benchmark --config CONFIG`)"
        )
    run_id = row["source_run_id"]
    seal = registry.get("seal")
    sealed_sha = seal["artifacts"].get(run_id) if seal else None
    return model_ref(paths, run_id, sealed_artifact_sha256=sealed_sha)
