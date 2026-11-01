"""Pre-test selections for the industrial demonstration, made on validation evidence only.

* Demo checkpoint: among the study's methods at the demo cell (NEU-CLS, K=10, support seed 11),
  the highest validation macro-F1 wins. Exact ties go to the lexically first method name. Every
  candidate must be COMPLETED with verified evidence, or no selection is made.
* Review threshold: chosen on that run's stored validation predictions with
  :func:`fewshotlab.simulation.review.select_threshold`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fewshotlab.engine import artifacts
from fewshotlab.engine.study import plan_study, slot_status
from fewshotlab.errors import FewShotLabError
from fewshotlab.io_utils import read_json
from fewshotlab.paths import ProjectPaths
from fewshotlab.reporting.aggregate import verify_evidence
from fewshotlab.simulation.review import select_threshold

DEMO_TARGET: dict[str, Any] = {"dataset": "neu_cls", "k": 10, "seed": 11}
RULE = (
    "highest validation macro-F1 among the study's methods at the demo cell; exact ties go to "
    "the lexically first method name"
)


def demo_target_for(study: dict[str, Any]) -> dict[str, Any]:
    """The study's demo cell. Protocol v1 fixes it to NEU-CLS K=10 seed 11 outside fixtures."""
    target = dict(study.get("demo") or DEMO_TARGET)
    if study["category"] != "fixture" and target != DEMO_TARGET:
        raise FewShotLabError(f"protocol v1 fixes the demo cell to {DEMO_TARGET}, not {target}")
    return target


def choose_demo(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise FewShotLabError("no demo candidates")
    return min(candidates, key=lambda c: (-c["val_macro_f1"], c["method"]))


def demo_candidates(
    study: dict[str, Any], paths: ProjectPaths, target: dict[str, Any] = DEMO_TARGET
) -> list[dict[str, Any]]:
    cell = (target["dataset"], target["k"], target["seed"])
    candidates = []
    for slot in plan_study(study, paths):
        if (slot.dataset, slot.k, slot.seed) != cell:
            continue
        if slot.duplicate_of:
            raise FewShotLabError(f"demo cell slot {slot.run_id} duplicates {slot.duplicate_of}")
        state = slot_status(paths, slot)
        if state["status"] != "COMPLETED":
            raise FewShotLabError(
                f"every demo candidate must be COMPLETED; {slot.run_id} is {state['status']}"
            )
        attempt_dir = artifacts.run_root(paths, slot.run_id) / state["attempt_id"]
        problem = verify_evidence(attempt_dir)
        if problem:
            raise FewShotLabError(f"{slot.run_id}: {problem}")
        doc = read_json(attempt_dir / "val_metrics.json")
        candidates.append(
            {
                "method": slot.method,
                "run_id": slot.run_id,
                "attempt_id": state["attempt_id"],
                "val_macro_f1": doc["metrics"]["macro_f1"],
                "val_top1": doc["metrics"]["top1"],
            }
        )
    if sorted(c["method"] for c in candidates) != sorted(study["methods"]):
        raise FewShotLabError(
            f"demo cell {cell}: expected one run per method {sorted(study['methods'])}, "
            f"found {sorted(c['method'] for c in candidates)}"
        )
    return sorted(candidates, key=lambda c: c["method"])


def select_demo(study: dict[str, Any], paths: ProjectPaths) -> dict[str, Any]:
    target = demo_target_for(study)
    candidates = demo_candidates(study, paths, target)
    chosen = choose_demo(candidates)
    attempt_dir = artifacts.run_root(paths, chosen["run_id"]) / chosen["attempt_id"]
    stored = artifacts.read_predictions(attempt_dir / "predictions_val.csv")
    max_scores = stored["scores"].max(axis=1)
    return {
        "rule": RULE,
        "target": dict(target),
        "candidates": candidates,
        "selected": chosen,
        "threshold": select_threshold(stored["true"], stored["pred"], max_scores),
        "validation_max_score": {
            "min": float(np.min(max_scores)),
            "median": float(np.median(max_scores)),
            "max": float(np.max(max_scores)),
        },
    }
