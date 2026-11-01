"""Study plans: a prespecified matrix of run slots, its integrity checks and its execution.

A study config lists datasets, shots, methods and support seeds. Each slot expands to the same run
config (and therefore the same run ID) as a standalone config file, so no job can be run twice
under two names. Deterministic prototype slots whose support membership is identical to an earlier
slot are marked DEDUPLICATED and counted once. They are never counted as independent evidence.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fewshotlab import PROTOCOL_ID
from fewshotlab.config import (
    DATASETS,
    METHODS,
    RunConfig,
    config_hash,
    load_yaml_mapping,
    parse_run_config,
)
from fewshotlab.data.registry import manifest_path
from fewshotlab.data.support import load_support, support_path
from fewshotlab.data.transforms import spec_for
from fewshotlab.engine import artifacts
from fewshotlab.engine.runner import init_seed_for, run_training
from fewshotlab.errors import ConfigError, FewShotLabError
from fewshotlab.hashing import sha256_json
from fewshotlab.io_utils import atomic_write_json, atomic_write_text, utc_now
from fewshotlab.models.methods import get_method
from fewshotlab.paths import ProjectPaths, get_paths
from fewshotlab.status import RunStatus

STUDY_KEYS = {
    "study_id",
    "protocol_id",
    "category",
    "description",
    "datasets",
    "shots",
    "methods",
    "support_seeds",
    "device",
}
STUDY_CATEGORIES = ("mvp", "core", "development", "fixture")


@dataclass(frozen=True)
class Slot:
    dataset: str
    method: str
    k: int
    seed: int
    config: RunConfig
    run_id: str
    config_sha256: str
    support_sha256: str
    support_membership_sha256: str
    duplicate_of: str | None = None


def load_study(path: str | Path) -> dict[str, Any]:
    data = load_yaml_mapping(path)
    unknown = sorted(set(data) - STUDY_KEYS)
    missing = sorted(
        {"study_id", "category", "datasets", "shots", "methods", "support_seeds"} - set(data)
    )
    if unknown or missing:
        raise ConfigError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    if data.get("protocol_id", PROTOCOL_ID) != PROTOCOL_ID:
        raise ConfigError(f"{path}: protocol_id must be {PROTOCOL_ID}")
    if data["category"] not in STUDY_CATEGORIES:
        raise ConfigError(f"{path}: category must be one of {list(STUDY_CATEGORIES)}")
    for key in ("datasets", "shots", "methods", "support_seeds"):
        if not isinstance(data[key], list) or not data[key]:
            raise ConfigError(f"{path}: {key} must be a non-empty list")
    bad_methods = sorted(set(data["methods"]) - set(METHODS))
    if bad_methods:
        raise ConfigError(f"{path}: unknown methods {bad_methods}")
    return data


def _config_path(path: Path, paths: ProjectPaths) -> str:
    try:
        return path.resolve().relative_to(paths.root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def plan_study(study: dict[str, Any], paths: ProjectPaths | None = None) -> list[Slot]:
    paths = paths or get_paths()
    fixture = study["category"] == "fixture"
    if not fixture and set(study["datasets"]) - set(DATASETS):
        raise ConfigError(f"unknown datasets {sorted(set(study['datasets']) - set(DATASETS))}")
    slots: list[Slot] = []
    first_by_membership: dict[tuple[str, int, str], str] = {}
    for dataset in study["datasets"]:
        for k in study["shots"]:
            for method in study["methods"]:
                for seed in study["support_seeds"]:
                    cfg = parse_run_config(
                        {
                            "dataset": dataset,
                            "method": method,
                            "shots": k,
                            "support_seed": seed,
                            "split_manifest": _config_path(manifest_path(paths, dataset), paths),
                            "support_manifest": _config_path(
                                support_path(paths, dataset, k, seed), paths
                            ),
                            "category": study["category"],
                            "is_fixture": fixture,
                            "device": study.get("device", "auto"),
                        }
                    )
                    digest = config_hash(cfg)
                    doc = load_support(paths, dataset, k, seed)
                    membership = sha256_json({c: sorted(ids) for c, ids in doc["selected"].items()})
                    duplicate = None
                    if not get_method(
                        method
                    ).learned:  # deterministic: identical inputs, identical run
                        key = (dataset, k, membership)
                        duplicate = first_by_membership.get(key)
                        if duplicate is None:
                            first_by_membership[key] = artifacts.make_run_id(cfg, digest)
                    slots.append(
                        Slot(
                            dataset,
                            method,
                            k,
                            seed,
                            cfg,
                            artifacts.make_run_id(cfg, digest),
                            digest,
                            doc["content_sha256"],
                            membership,
                            duplicate,
                        )
                    )
    return slots


def verify_plan(slots: list[Slot]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    ids = [s.run_id for s in slots]
    check("unique_run_ids", len(ids) == len(set(ids)), f"{len(set(ids))} unique of {len(ids)}")
    groups: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    for s in slots:
        groups[(s.dataset, s.k, s.seed)].add(s.support_sha256)
    shared = all(len(v) == 1 for v in groups.values())
    check(
        "shared_support_ids",
        shared,
        "every method at a dataset/K/seed reads the same support manifest"
        if shared
        else f"mismatched groups: {[k for k, v in groups.items() if len(v) > 1]}",
    )
    init_ok = all((s.config.weights == "none") == (s.method == "scratch") for s in slots)
    check(
        "initialization_policy",
        init_ok,
        "scratch starts from random init; frozen/partial/prototypes from IMAGENET1K_V1",
    )
    seeds_ok = all(init_seed_for(s.config.train_seed) == init_seed_for(s.seed) for s in slots)
    check("paired_init_seed", seeds_ok, "init seed depends only on train_seed (= support_seed)")
    train_ids = {
        (s.dataset, spec_for(s.dataset, "train").transform_id)
        for s in slots
        if get_method(s.method).learned
    }
    per_dataset = defaultdict(set)
    for dataset, transform in train_ids:
        per_dataset[dataset].add(transform)
    same = all(len(v) == 1 for v in per_dataset.values())
    check(
        "identical_training_transform_per_dataset",
        same,
        f"{ {d: sorted(v) for d, v in per_dataset.items()} }",
    )
    recipe = {
        (s.config.train, s.config.optimizer, s.config.selection)
        for s in slots
        if get_method(s.method).learned
    }
    check(
        "single_recipe", len(recipe) <= 1, f"{len(recipe)} distinct recipe(s) across learned slots"
    )
    return checks


def slot_status(paths: ProjectPaths, slot: Slot) -> dict[str, Any]:
    if slot.duplicate_of:
        return {"status": "DEDUPLICATED", "attempt_id": None}
    attempts = [(d, artifacts.read_status(d)) for d in artifacts.attempt_dirs(paths, slot.run_id)]
    if not attempts:
        return {"status": RunStatus.NOT_RUN.value, "attempt_id": None, "attempts": 0}
    completed = [(d, s) for d, s in attempts if s and s["status"] == RunStatus.COMPLETED]
    chosen_dir, chosen = completed[-1] if completed else attempts[-1]
    return {
        "status": (chosen or {}).get("status", "UNKNOWN"),
        "attempt_id": chosen_dir.name,
        "attempts": len(attempts),
    }


def write_plan(
    study: dict[str, Any], slots: list[Slot], checks: list[dict[str, str]], paths: ProjectPaths
) -> Path:
    out = paths.reports / study["study_id"]
    rows = [
        {
            "dataset": s.dataset,
            "k": s.k,
            "method": s.method,
            "seed": s.seed,
            "run_id": s.run_id,
            "config_sha256": s.config_sha256,
            "support_sha256": s.support_sha256,
            "support_membership_sha256": s.support_membership_sha256,
            "duplicate_of": s.duplicate_of,
            **slot_status(paths, s),
        }
        for s in slots
    ]
    unique = sum(1 for s in slots if not s.duplicate_of)
    atomic_write_json(
        out / "plan.json",
        {
            "study": study,
            "generated_utc": utc_now(),
            "logical_slots": len(slots),
            "unique_runs": unique,
            "checks": checks,
            "slots": rows,
        },
    )
    lines = [
        f"# Study plan: {study['study_id']}",
        "",
        f"{len(slots)} logical slots, {unique} unique runs "
        f"({len(slots) - unique} deterministic duplicates counted once). Category "
        f"`{study['category']}`. Generated {utc_now()}.",
        "",
        "## Pre-flight checks",
        "",
        *[f"- {c['status']} {c['check']}: {c['detail']}" for c in checks],
        "",
        "## Slots",
        "",
        "| dataset | K | method | seed | status | run ID |",
        "|---|---:|---|---:|---|---|",
    ]
    lines += [
        f"| {r['dataset']} | {r['k']} | {r['method']} | {r['seed']} | {r['status']}"
        f"{' (= ' + r['duplicate_of'] + ')' if r['duplicate_of'] else ''} | "
        f"`{r['run_id']}` |"
        for r in rows
    ]
    atomic_write_text(out / "plan.md", "\n".join(lines) + "\n")
    return out / "plan.json"


def run_study(
    study: dict[str, Any],
    *,
    paths: ProjectPaths | None = None,
    device_override: str | None = None,
    allow_download: bool = False,
    init_override: str | None = None,
    command: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run every unique, unfinished slot in order. A failed slot is recorded and the study goes on;
    an interruption (Ctrl-C) stops it."""
    paths = paths or get_paths()
    slots = plan_study(study, paths)
    checks = verify_plan(slots)
    write_plan(study, slots, checks, paths)
    failed_checks = [c for c in checks if c["status"] == "FAIL"]
    if failed_checks:
        raise ConfigError(f"pre-flight checks failed: {failed_checks}")
    outcomes = []
    for slot in slots:
        state = slot_status(paths, slot)["status"]
        if slot.duplicate_of or state == RunStatus.COMPLETED:
            outcomes.append({"run_id": slot.run_id, "outcome": "skipped", "status": state})
            continue
        try:
            result = run_training(
                slot.config,
                paths=paths,
                device_override=device_override,
                allow_download=allow_download,
                init_override=init_override,
                command=command,
            )
            outcomes.append(
                {
                    "run_id": slot.run_id,
                    "outcome": "ran",
                    "status": result.status,
                    "macro_f1_val": result.metrics["macro_f1"],
                }
            )
        except (FewShotLabError, RuntimeError, ValueError, OSError) as exc:
            outcomes.append(
                {
                    "run_id": slot.run_id,
                    "outcome": "failed",
                    "status": RunStatus.FAILED.value,
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                }
            )
    write_plan(study, slots, checks, paths)
    return outcomes
