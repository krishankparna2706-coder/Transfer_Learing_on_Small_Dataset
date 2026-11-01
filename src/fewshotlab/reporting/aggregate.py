"""Select the run records that may appear in a report, with an explicit reason for each exclusion.

A record is accepted only if all of the following hold:
* its latest status is COMPLETED;
* it is not a software fixture;
* its protocol and category match the request;
* its evidence verifies: the predictions file hash matches, and the metrics recomputed from that
  file equal the stored ones.
"""

from __future__ import annotations

from typing import Any

from fewshotlab import PROTOCOL_ID
from fewshotlab.engine import artifacts
from fewshotlab.engine.ledger import latest_status
from fewshotlab.hashing import sha256_file
from fewshotlab.io_utils import read_json
from fewshotlab.metrics.core import classification_metrics
from fewshotlab.paths import ProjectPaths


def verify_evidence(attempt_dir) -> str | None:
    """None if the stored validation metrics regenerate from the stored predictions."""
    doc_path = attempt_dir / "val_metrics.json"
    predictions = attempt_dir / "predictions_val.csv"
    if not doc_path.is_file() or not predictions.is_file():
        return "missing val_metrics.json or predictions_val.csv"
    doc = read_json(doc_path)
    if sha256_file(predictions) != doc["predictions"]["sha256"]:
        return "predictions file hash does not match val_metrics.json"
    stored = artifacts.read_predictions(predictions)
    topk = (5,) if doc["dataset"] == "flowers102" else ()
    again = classification_metrics(
        stored["true"],
        stored["pred"],
        doc["metrics"]["num_classes"],
        scores=stored["scores"],
        topk=topk,
    )
    for key in ("macro_f1", "top1", "balanced_accuracy", *(f"top{k}" for k in topk)):
        if again[key] != doc["metrics"][key]:
            return f"{key} does not regenerate from the stored predictions"
    if again["confusion"] != doc["metrics"]["confusion"]:
        return "confusion matrix does not regenerate from the stored predictions"
    return None


def collect_runs(
    paths: ProjectPaths,
    *,
    protocol_id: str = PROTOCOL_ID,
    categories: tuple[str, ...] = ("core",),
) -> dict[str, list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for (run_id, attempt_id), event in sorted(latest_status(paths).items()):
        reason = None
        if event.get("status") != "COMPLETED":
            reason = f"status {event.get('status')}"
        elif event.get("is_fixture"):
            reason = "software fixture (is_fixture=true)"
        elif event.get("protocol_id") != protocol_id:
            reason = f"protocol {event.get('protocol_id')} != {protocol_id}"
        elif event.get("category") not in categories:
            reason = f"category {event.get('category')} not in {list(categories)}"
        else:
            reason = verify_evidence(artifacts.run_root(paths, run_id) / attempt_id)
        row = {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "status": event.get("status"),
            "category": event.get("category"),
            "is_fixture": event.get("is_fixture"),
        }
        if reason:
            rejected.append({**row, "reason": reason})
        else:
            doc = read_json(artifacts.run_root(paths, run_id) / attempt_id / "val_metrics.json")
            accepted.append(
                {
                    **row,
                    "dataset": doc["dataset"],
                    "method": doc["method"],
                    "k": doc["k"],
                    "seed": doc["seed"],
                    "metrics": doc["metrics"],
                }
            )
    return {"accepted": accepted, "rejected": rejected}
