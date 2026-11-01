"""The P5 test lock: seal every pre-test decision, then re-verify it before any test access.

``lock-test`` derives a plan from stored files only (no test prediction exists yet) and writes it
to ``reports/<study_id>/sealed_plan.json``.

* Content is re-derived and compared on every test access. Any difference refuses access.
  - code: a digest of src/, configs/ and the lock files, plus per-file protocol-module hashes
  - environment: lock hashes, Python and library versions, the evaluation device
  - data: split-manifest and class-map hashes, split counts, expected test IDs (count and hash)
  - supports: each support manifest's content hash
  - runs: every unique COMPLETED slot with its attempt, checkpoint or prototype hash, model-spec
    hash, validation prediction and metric hashes and validation macro-F1. Deterministic
    duplicates point at their original and are never evaluated twice.
  - pretrained weights, metric definitions, the review scorer, calibration (none), the
    corruption protocol, the demo checkpoint and review threshold chosen on validation, the
    replay manifest hash, the test-evaluation plan and the report rules
* Attestations are recorded once, when sealing:
  - the git state and the commits that trained the sealed runs
  - a validation reproduction: the current code must re-score every sealed run's validation set
    with exactly the stored predictions

The plan's SHA-256 and a random audit token are written to the plan file, the study's evaluation
ledger and the run ledger. A study can be sealed once. Changing a sealed decision needs a new,
separately named protocol.
"""

from __future__ import annotations

import json
import os
import platform
import secrets
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fewshotlab import PROTOCOL_ID, __version__
from fewshotlab.data.manifest import load_class_map, read_manifest
from fewshotlab.data.registry import class_map_path, manifest_path, require_prepared
from fewshotlab.devices import resolve_device
from fewshotlab.engine import artifacts
from fewshotlab.engine.ledger import append_event
from fewshotlab.engine.provenance import code_files, git_state
from fewshotlab.engine.study import plan_study, slot_status
from fewshotlab.errors import FewShotLabError, ProtocolRefusal
from fewshotlab.hashing import sha256_file, sha256_json
from fewshotlab.io_utils import (
    append_jsonl,
    atomic_write_bytes,
    atomic_write_json,
    read_json,
    utc_now,
)
from fewshotlab.models.methods import REGISTRY
from fewshotlab.paths import ProjectPaths, portable_path
from fewshotlab.reporting.aggregate import verify_evidence
from fewshotlab.simulation import corruptions, review
from fewshotlab.simulation.demo import demo_target_for, select_demo
from fewshotlab.simulation.replay import REPLAY_FILE, replay_manifest, sealed_summary

SCHEMA_VERSION = 1
SEALED_PLAN = "sealed_plan.json"
SEAL_PREVIEW = "seal_preview.json"
EVALUATION_LEDGER = "evaluation_ledger.jsonl"
TEST_REPORT_DIR = "test"
ROBUSTNESS_DIR = "robustness"
SEALED_SOURCE_ROOTS = ("src", "configs")
SEALED_SOURCE_FILES = ("pyproject.toml", "uv.lock", "requirements.lock", ".python-version")
PROTOCOL_MODULES = (
    "src/fewshotlab/data/imageio.py",
    "src/fewshotlab/data/transforms.py",
    "src/fewshotlab/metrics/core.py",
    "src/fewshotlab/models/backbone.py",
    "src/fewshotlab/models/methods.py",
    "src/fewshotlab/models/prototypes.py",
    "src/fewshotlab/models/spec.py",
    "src/fewshotlab/engine/inference.py",
    "src/fewshotlab/engine/seal.py",
    "src/fewshotlab/engine/test_eval.py",
    "src/fewshotlab/simulation/corruptions.py",
    "src/fewshotlab/simulation/demo.py",
    "src/fewshotlab/simulation/replay.py",
    "src/fewshotlab/simulation/review.py",
)
TEST_OUTPUT_PREFIXES = (
    "predictions_test",
    "test_metrics",
    "robustness_test",
    "corruption_checksums",
)

METRICS = {
    "implementation": "fewshotlab.metrics.core.classification_metrics on stored predictions",
    "primary": "macro_f1",
    "class_set": "all declared classes of the dataset's class map",
    "zero_division": 0,
    "definitions": {
        "top1": "correct / N",
        "macro_f1": "mean over all classes of F1_c = 2 TP_c / (2 TP_c + FP_c + FN_c)",
        "balanced_accuracy": "mean over all classes of recall_c",
        "top5": "Flowers102 only; classes ranked by score, ties to the lower class ID",
        "per_class": "precision, recall, F1 and support for every declared class",
        "confusion": "raw C x C counts (rows = true class) and row-normalized (empty rows null)",
    },
    "prediction": "argmax of the score vector; ties go to the lowest class ID",
}
CALIBRATION = {
    "method": "none",
    "note": "scores are uncalibrated model scores; no temperature, Platt or isotonic fit",
}
REPORT_RULES = {
    "primary_metric": "macro_f1 (top-1, balanced accuracy, per-class and confusion also reported)",
    "aggregation": "per dataset x K x method: mean and sample SD (ddof=1) over distinct runs; "
    "a deterministic duplicate counts once",
    "paired": "per dataset x K: method differences matched on support seed",
    "intervals": "no population confidence interval is inferred from five seeds",
    "robustness": "corruption metrics are reported apart from the primary benchmark; each "
    "derivative is paired with its clean parent and never counted as a new test image",
    "one_result_per_run": "each sealed run is evaluated on test exactly once",
}


@dataclass(frozen=True)
class VerifiedSeal:
    """Proof that the sealed plan matched the current files when it was checked."""

    study_id: str
    directory: Path
    plan: dict[str, Any]
    audit_token: str
    plan_sha256: str
    verified_utc: str


def seal_dir(paths: ProjectPaths, study_id: str) -> Path:
    return paths.reports / study_id


def _normalize(obj: Any) -> Any:
    return json.loads(json.dumps(obj, sort_keys=True, allow_nan=False))


def _plan_hash(plan: dict[str, Any]) -> str:
    return sha256_json({"content": plan["content"], "attestations": plan["attestations"]})


def _sealed_scope(relative: str) -> bool:
    return relative.split("/", 1)[0] in SEALED_SOURCE_ROOTS or relative in SEALED_SOURCE_FILES


def code_section(root: Path) -> dict[str, Any]:
    entries = [
        [p.relative_to(root).as_posix(), sha256_file(p)]
        for p in code_files(root)
        if _sealed_scope(p.relative_to(root).as_posix())
    ]
    return {
        "scope": "src/, configs/, " + ", ".join(SEALED_SOURCE_FILES) + " (tests/, app/ excluded)",
        "source_sha256": sha256_json(entries),
        "source_files": len(entries),
        "protocol_modules": {
            rel: (sha256_file(root / rel) if (root / rel).is_file() else None)
            for rel in PROTOCOL_MODULES
        },
    }


def _environment_section(root: Path, study: dict[str, Any]) -> dict[str, Any]:
    import numpy
    import PIL
    import torch
    import torchvision

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "locks": {
            name: (sha256_file(root / name) if (root / name).is_file() else None)
            for name in ("uv.lock", "requirements.lock")
        },
        "study_device": study.get("device", "auto"),
        "device": resolve_device(study.get("device", "auto")),
    }


def _data_section(paths: ProjectPaths, datasets: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset in datasets:
        record = require_prepared(paths, dataset)
        mpath = manifest_path(paths, dataset)
        manifest_sha = sha256_file(mpath)
        if manifest_sha != record["manifest"]["sha256"]:
            raise ProtocolRefusal(f"{mpath.name} changed after preparation")
        cmap = load_class_map(class_map_path(paths, dataset))
        counts = {s: len(read_manifest(mpath, splits=(s,))) for s in ("train", "val", "test")}
        test_ids = sorted(r.image_id for r in read_manifest(mpath, splits=("test",)))
        out[dataset] = {
            "split_manifest": mpath.name,
            "split_manifest_sha256": manifest_sha,
            "class_map_sha256": cmap.sha256(),
            "num_classes": cmap.num_classes,
            "counts": counts,
            "expected_test_ids": {"count": len(test_ids), "sha256": sha256_json(test_ids)},
        }
    return out


def _run_entry(study: dict[str, Any], slot, attempt_id: str, attempt_dir: Path) -> dict[str, Any]:
    doc = read_json(attempt_dir / "val_metrics.json")
    provenance = read_json(attempt_dir / "provenance.json")
    spec = read_json(attempt_dir / "model_spec.json")
    artifact_file = doc["artifact"]["file"]
    artifact_sha = sha256_file(attempt_dir / artifact_file)
    if artifact_sha != doc["artifact"]["sha256"]:
        raise FewShotLabError(f"{artifact_file} hash differs from its run record")
    if doc["category"] != study["category"] or bool(doc["is_fixture"]) != (
        study["category"] == "fixture"
    ):
        raise FewShotLabError("run category or fixture flag does not match the study")
    code = provenance.get("code", {})
    return {
        "run_id": slot.run_id,
        "attempt_id": attempt_id,
        "dataset": slot.dataset,
        "method": slot.method,
        "k": slot.k,
        "seed": slot.seed,
        "category": doc["category"],
        "is_fixture": bool(doc["is_fixture"]),
        "config_sha256": slot.config_sha256,
        "support_sha256": slot.support_sha256,
        "artifact": {"file": artifact_file, "sha256": artifact_sha},
        "model_spec_sha256": sha256_file(attempt_dir / "model_spec.json"),
        "predictions_val_sha256": sha256_file(attempt_dir / "predictions_val.csv"),
        "val_metrics_sha256": sha256_file(attempt_dir / "val_metrics.json"),
        "val_macro_f1": doc["metrics"]["macro_f1"],
        "selected_epoch": doc.get("selected_epoch"),
        "encoder": {
            key: spec["encoder"].get(key) for key in ("init", "weights_id", "filename", "sha256")
        },
        "training_code": {
            "commit": code.get("commit"),
            "source_sha256": code.get("source_sha256"),
            "dirty": code.get("dirty"),
        },
    }


def _runs_section(study: dict[str, Any], paths: ProjectPaths, slots: list) -> dict[str, Any]:
    runs, duplicates, missing = [], [], []
    for slot in slots:
        where = {"dataset": slot.dataset, "method": slot.method, "k": slot.k, "seed": slot.seed}
        if slot.duplicate_of:
            duplicates.append({**where, "run_id": slot.run_id, "duplicate_of": slot.duplicate_of})
            continue
        state = slot_status(paths, slot)
        if state["status"] != "COMPLETED":
            missing.append({**where, "run_id": slot.run_id, "status": state["status"]})
            continue
        attempt_dir = artifacts.run_root(paths, slot.run_id) / state["attempt_id"]
        problem = verify_evidence(attempt_dir)
        if problem is None:
            try:
                runs.append(_run_entry(study, slot, state["attempt_id"], attempt_dir))
                continue
            except FewShotLabError as exc:
                problem = str(exc)
        missing.append(
            {**where, "run_id": slot.run_id, "status": "EVIDENCE_FAILED", "reason": problem}
        )
    return {"runs": runs, "duplicates": duplicates, "missing": missing}


def _weights_section(paths: ProjectPaths, runs: list[dict[str, Any]]) -> dict[str, Any]:
    weights: dict[str, Any] = {}
    for run in runs:
        encoder = run["encoder"]
        if encoder.get("weights_id") in (None, "none"):
            continue
        cached = paths.weights_dir / str(encoder["filename"])
        weights[encoder["weights_id"]] = {
            "filename": encoder["filename"],
            "sha256": encoder["sha256"],
            "cached_file_sha256": sha256_file(cached) if cached.is_file() else None,
        }
    return weights


def _scorer_section() -> dict[str, Any]:
    return {
        "implementation": "fewshotlab.simulation.review",
        "score": "maximum class score: softmax(logits) for learned methods, softmax(-squared "
        "distance) for prototypes; an uncalibrated model score, not a probability of being right",
        "score_kinds": {name: info.score_kind for name, info in sorted(REGISTRY.items())},
        "threshold_grid": list(review.THRESHOLD_GRID),
        "target_validation_coverage": review.TARGET_COVERAGE,
        "selection_rule": "highest grid threshold with validation coverage >= 0.80",
        "review_rule": "review when the maximum score is strictly below the threshold",
        "accepted_error_when_nothing_accepted": None,
    }


def build_seal_plan(study: dict[str, Any], paths: ProjectPaths) -> dict[str, Any]:
    """The plan content from current files, plus problems that block sealing."""
    blocking: list[str] = []
    demo_target = demo_target_for(study)
    slots = plan_study(study, paths)
    runs = _runs_section(study, paths, slots)
    try:
        demo: dict[str, Any] | None = select_demo(study, paths)
    except FewShotLabError as exc:
        demo = None
        blocking.append(f"demo selection is not possible yet: {exc}")
    weights = _weights_section(paths, runs["runs"])
    for weights_id, entry in weights.items():
        if entry["cached_file_sha256"] != entry["sha256"]:
            blocking.append(f"pretrained weights {weights_id} are missing or differ in the cache")
    replay = replay_manifest(paths, demo_target["dataset"])
    content = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": study.get("protocol_id", PROTOCOL_ID),
        "fewshotlab_version": __version__,
        "study": {"config_sha256": sha256_json(study), **study},
        "code": code_section(paths.root),
        "environment": _environment_section(paths.root, study),
        "data": _data_section(paths, list(study["datasets"])),
        "supports": {f"{s.dataset}-k{s.k}-s{s.seed}": s.support_sha256 for s in slots},
        "logical_slots": len(slots),
        **runs,
        "weights": weights,
        "metrics": METRICS,
        "scorer": _scorer_section(),
        "calibration": CALIBRATION,
        "corruptions": corruptions.protocol(),
        "demo": demo,
        "replay": sealed_summary(replay),
        "test_plan": {
            "evaluate": "every sealed run once, on the full held-out test split of its dataset "
            "(clean images), with the same inference code as validation",
            "outputs": "predictions_test.csv and test_metrics.json in each sealed attempt folder",
            "robustness": {
                "run": "the sealed demo checkpoint only",
                "dataset": demo_target["dataset"],
                "conditions": replay["streams"],
                "outputs": f"{ROBUSTNESS_DIR}/ in the demo attempt folder",
            },
            "repeat_policy": "one result per sealed run; a recorded evaluation is never re-run",
            "invalidation": "a changed recipe, checkpoint, support, scorer, threshold, corruption "
            "protocol or test selection invalidates the seal",
        },
        "report_rules": REPORT_RULES,
    }
    return {"content": _normalize(content), "blocking": blocking, "replay": replay}


def _git_lines(root: Path, *args: str) -> list[str] | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=60
        )
    except OSError:
        return None
    return out.stdout.splitlines() if out.returncode == 0 else None


def training_code(paths: ProjectPaths, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The commits that trained the sealed runs and the sealed code files changed since."""
    commits = sorted(
        {(r["training_code"]["commit"], r["training_code"]["dirty"]) for r in runs},
        key=lambda item: (str(item[0]), str(item[1])),
    )
    out = []
    for commit, dirty in commits:
        changed = (
            _git_lines(
                paths.root,
                "diff",
                "--name-only",
                str(commit),
                "HEAD",
                "--",
                *SEALED_SOURCE_ROOTS,
                *SEALED_SOURCE_FILES,
            )
            if commit
            else None
        )
        out.append(
            {
                "commit": commit,
                "dirty": dirty,
                "runs": sum(1 for r in runs if r["training_code"]["commit"] == commit),
                "sealed_files_changed_since": changed,
            }
        )
    return out


def reproduce_validation(
    paths: ProjectPaths, runs: list[dict[str, Any]], *, device: str | None = None
) -> dict[str, Any]:
    """Re-score every sealed run's validation set with the current code."""
    from fewshotlab.engine.runner import evaluate_run

    results, failures = [], []
    for run in runs:
        check = evaluate_run(run["run_id"], "val", paths=paths, device_override=device)
        row = {
            "run_id": run["run_id"],
            "attempt_id": check["attempt_id"],
            "device": check["device"],
            "identical_predictions": check["identical_predictions"],
            "macro_f1_equal": check["recomputed_macro_f1"] == check["stored_macro_f1"],
            "max_abs_score_difference": check["max_abs_score_difference"],
        }
        if not (
            row["identical_predictions"]
            and row["macro_f1_equal"]
            and row["attempt_id"] == run["attempt_id"]
        ):
            failures.append(run["run_id"])
        results.append(row)
    return {
        "runs": len(results),
        "all_identical": not failures,
        "failures": failures,
        "devices": sorted({r["device"] for r in results}),
        "max_abs_score_difference": max(
            (r["max_abs_score_difference"] for r in results), default=None
        ),
        "results": results,
    }


def read_evaluation_ledger(directory: Path) -> list[dict[str, Any]]:
    path = directory / EVALUATION_LEDGER
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_evaluation_event(seal: VerifiedSeal, **fields: Any) -> dict[str, Any]:
    event = {
        "time_utc": utc_now(),
        "study_id": seal.study_id,
        "audit_token": seal.audit_token,
        "plan_sha256": seal.plan_sha256,
        **fields,
    }
    append_jsonl(seal.directory / EVALUATION_LEDGER, event)
    return event


def is_test_output(path: Path) -> bool:
    return path.name.startswith(TEST_OUTPUT_PREFIXES)


def find_test_outputs(paths: ProjectPaths) -> list[Path]:
    """Every per-run test output under runs/ (symlinks followed) and every test report file."""
    hits: list[Path] = []
    for base in (paths.runs, paths.demo_assets):
        for folder, _dirs, files in os.walk(base, followlinks=True):
            hits += [Path(folder) / name for name in files if is_test_output(Path(name))]
    if paths.reports.is_dir():
        for study_dir in sorted(p for p in paths.reports.iterdir() if p.is_dir()):
            for folder, _dirs, files in os.walk(study_dir):
                inside = Path(folder).relative_to(study_dir).parts[:1] == (TEST_REPORT_DIR,)
                hits += [
                    Path(folder) / n for n in files if inside or is_test_output(Path(folder) / n)
                ]
    return sorted(set(hits))


def _sealed_studies(paths: ProjectPaths) -> dict[str, dict[str, Any]]:
    """Sealed plans whose own hash and ledger record are intact (a light check, no rebuild)."""
    out: dict[str, dict[str, Any]] = {}
    if not paths.reports.is_dir():
        return out
    for plan_path in sorted(paths.reports.glob(f"*/{SEALED_PLAN}")):
        plan = read_json(plan_path)
        events = read_evaluation_ledger(plan_path.parent)
        sealed = any(
            e.get("event") == "sealed"
            and e.get("audit_token") == plan.get("audit_token")
            and e.get("plan_sha256") == plan.get("plan_sha256")
            for e in events
        )
        if plan.get("is_seal") and plan.get("plan_sha256") == _plan_hash(plan) and sealed:
            out[plan_path.parent.name] = {"plan": plan, "events": events}
    return out


def audit_test_outputs(paths: ProjectPaths) -> list[str]:
    """Problems with test-split outputs: each must be recorded under an intact seal."""
    hits = find_test_outputs(paths)
    if not hits:
        return []
    sealed = _sealed_studies(paths)
    recorded: dict[tuple[str, str, str], str] = {}
    for study in sealed.values():
        token = study["plan"]["audit_token"]
        for event in study["events"]:
            if event.get("audit_token") != token or "files" not in event:
                continue
            for name, digest in event["files"].items():
                recorded[(event["run_id"], event["attempt_id"], name)] = digest
    problems = []
    artifacts_root = paths.runs / artifacts.ARTIFACTS_DIRNAME
    for path in hits:
        try:
            run_id, attempt_id, *rest = path.relative_to(artifacts_root).parts
        except ValueError:
            try:
                study_id = path.relative_to(paths.reports).parts[0]
            except ValueError:
                problems.append(f"{portable_path(path)}: test output outside any run or report")
                continue
            if study_id not in sealed:
                problems.append(f"{portable_path(path)}: study {study_id} has no intact seal")
            continue
        key = (run_id, attempt_id, "/".join(rest))
        if recorded.get(key) != sha256_file(path):
            problems.append(f"{portable_path(path)}: not recorded by a sealed evaluation event")
    return problems


def preview(
    study: dict[str, Any],
    paths: ProjectPaths,
    *,
    reproduce: bool = True,
    device: str | None = None,
) -> dict[str, Any]:
    """``lock-test --dry-run``: everything the seal would record. Test access stays locked."""
    built = build_seal_plan(study, paths)
    content = built["content"]
    git = git_state(paths.root)
    blocking = list(built["blocking"])
    if git["dirty"] and study["category"] != "fixture":
        blocking.append("code, config or lock files differ from the commit")
    if content["missing"]:
        blocking.append(f"{len(content['missing'])} planned run(s) are not COMPLETED and verified")
    exposure = [portable_path(p) for p in find_test_outputs(paths)]
    if exposure:
        blocking.append(f"{len(exposure)} test-split output(s) already exist")
    reproduction = (
        reproduce_validation(paths, content["runs"], device=device)
        if reproduce and content["runs"]
        else None
    )
    if reproduction and not reproduction["all_identical"]:
        blocking.append("validation reproduction failed for " + ", ".join(reproduction["failures"]))
    doc = {
        "is_seal": False,
        "note": "PREVIEW ONLY: nothing is sealed and the test split stays locked",
        "generated_utc": utc_now(),
        "study_id": study["study_id"],
        "ready_to_seal": not blocking,
        "blocking": blocking,
        "content_sha256": sha256_json(content),
        "git": {k: git[k] for k in ("commit", "branch", "dirty", "changed_code_paths")},
        "training_code": training_code(paths, content["runs"]),
        "validation_reproduction": reproduction,
        "content": content,
    }
    atomic_write_json(seal_dir(paths, study["study_id"]) / SEAL_PREVIEW, doc)
    return doc


def seal(
    study: dict[str, Any],
    paths: ProjectPaths,
    *,
    allow_partial: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    """``lock-test``: write the sealed plan and its audit token. Refuses on any open problem."""
    directory = seal_dir(paths, study["study_id"])
    if (directory / SEALED_PLAN).exists():
        raise ProtocolRefusal(
            f"study {study['study_id']} is already sealed; a changed decision needs a new, "
            "separately named protocol"
        )
    problems: list[str] = []
    exposure = find_test_outputs(paths)
    if exposure:
        problems.append(
            f"test-split outputs already exist ({len(exposure)}), e.g. {portable_path(exposure[0])}"
        )
    built = build_seal_plan(study, paths)
    content = built["content"]
    problems += built["blocking"]
    if content["missing"] and not allow_partial:
        problems.append(
            f"{len(content['missing'])} planned run(s) are not COMPLETED and verified (seal a "
            "clearly labelled partial benchmark only with --allow-partial)"
        )
    if not content["runs"]:
        problems.append("no completed runs to seal")
    git = git_state(paths.root)
    if git["dirty"] and study["category"] != "fixture":
        problems.append("code, config or lock files differ from the commit; commit them first")
    if problems:
        raise ProtocolRefusal("cannot seal:\n  - " + "\n  - ".join(problems))
    reproduction = reproduce_validation(paths, content["runs"], device=device)
    if not reproduction["all_identical"]:
        raise ProtocolRefusal(
            "cannot seal: the current code does not reproduce the stored validation predictions "
            f"of {reproduction['failures']}"
        )
    attestations = _normalize(
        {
            "sealed_by": f"fewshotlab {__version__} lock-test",
            "git": {k: git[k] for k in ("commit", "branch", "dirty", "changed_code_paths")},
            "training_code": training_code(paths, content["runs"]),
            "validation_reproduction": reproduction,
            "partial": bool(content["missing"]),
        }
    )
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(directory / REPLAY_FILE, built["replay"]["csv"])
    plan: dict[str, Any] = {"content": content, "attestations": attestations}
    plan_sha = _plan_hash(plan)
    sealed_utc = utc_now()
    token = (
        "seal-"
        + sha256_json(
            {"plan_sha256": plan_sha, "sealed_utc": sealed_utc, "nonce": secrets.token_hex(16)}
        )[:32]
    )
    plan = {
        "is_seal": True,
        "schema_version": SCHEMA_VERSION,
        "study_id": study["study_id"],
        "protocol_id": content["protocol_id"],
        "partial": bool(content["missing"]),
        "sealed_utc": sealed_utc,
        "audit_token": token,
        "plan_sha256": plan_sha,
        **plan,
    }
    atomic_write_json(directory / SEALED_PLAN, plan)
    append_jsonl(
        directory / EVALUATION_LEDGER,
        {
            "event": "sealed",
            "time_utc": sealed_utc,
            "study_id": study["study_id"],
            "audit_token": token,
            "plan_sha256": plan_sha,
            "runs": len(content["runs"]),
            "partial": plan["partial"],
        },
    )
    append_event(
        paths,
        event="seal",
        study_id=study["study_id"],
        audit_token=token,
        plan_sha256=plan_sha,
        runs=len(content["runs"]),
        file=portable_path(directory / SEALED_PLAN),
    )
    return plan


def _short(value: Any) -> str:
    text = json.dumps(value, sort_keys=True)
    return text if len(text) <= 80 else text[:77] + "..."


def _differences(sealed: Any, current: Any, where: str, out: list[str], limit: int = 25) -> None:
    if len(out) >= limit:
        return
    if isinstance(sealed, dict) and isinstance(current, dict):
        for key in sorted(set(sealed) | set(current)):
            if key not in current:
                out.append(f"{where}.{key}: removed")
            elif key not in sealed:
                out.append(f"{where}.{key}: added")
            else:
                _differences(sealed[key], current[key], f"{where}.{key}", out, limit)
    elif isinstance(sealed, list) and isinstance(current, list) and len(sealed) == len(current):
        for i, (a, b) in enumerate(zip(sealed, current, strict=True)):
            label = a.get("run_id", i) if isinstance(a, dict) else i
            _differences(a, b, f"{where}[{label}]", out, limit)
    elif sealed != current:
        out.append(f"{where}: sealed {_short(sealed)} != now {_short(current)}")


def verify_seal(
    study: dict[str, Any], paths: ProjectPaths, *, require_clean: bool = True
) -> VerifiedSeal:
    """Fail closed: refuse unless the sealed plan matches the current files exactly."""
    directory = seal_dir(paths, study["study_id"])
    path = directory / SEALED_PLAN
    if not path.is_file():
        raise ProtocolRefusal(
            f"test evaluation is locked: study {study['study_id']} has no sealed evaluation "
            "plan. Run `lock-test --plan PATH` first. Nothing was evaluated and no file was "
            "written."
        )
    plan = read_json(path)
    if not plan.get("is_seal") or plan.get("plan_sha256") != _plan_hash(plan):
        raise ProtocolRefusal(f"{portable_path(path)} was altered after sealing")
    token = plan["audit_token"]
    if not any(
        e.get("event") == "sealed"
        and e.get("audit_token") == token
        and e.get("plan_sha256") == plan["plan_sha256"]
        for e in read_evaluation_ledger(directory)
    ):
        raise ProtocolRefusal("the seal's audit token is not recorded in the evaluation ledger")
    replay_path = directory / REPLAY_FILE
    if not replay_path.is_file() or sha256_file(replay_path) != plan["content"]["replay"].get(
        "csv_sha256"
    ):
        raise ProtocolRefusal(f"{REPLAY_FILE} is missing or differs from the sealed hash")
    sealed_study = {k: v for k, v in plan["content"]["study"].items() if k != "config_sha256"}
    if _normalize(study) != sealed_study:
        raise ProtocolRefusal("the study configuration differs from the sealed one")
    try:
        current = build_seal_plan(study, paths)["content"]
    except FewShotLabError as exc:
        raise ProtocolRefusal(
            "the sealed inputs can no longer be re-derived; test access refused (fail closed): "
            f"{exc}"
        ) from exc
    differences: list[str] = []
    _differences(plan["content"], current, "content", differences)
    if differences:
        raise ProtocolRefusal(
            "the sealed plan no longer matches the current files; test access refused (fail "
            "closed):\n  - " + "\n  - ".join(differences)
        )
    if require_clean and study["category"] != "fixture" and git_state(paths.root)["dirty"]:
        raise ProtocolRefusal("code, config or lock files differ from the commit")
    return VerifiedSeal(study["study_id"], directory, plan, token, plan["plan_sha256"], utc_now())
