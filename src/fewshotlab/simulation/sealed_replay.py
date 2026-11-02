"""A verified, read-only view of the sealed NEU replay (P6 shows it; P7 adds live inference).

After the sealed evaluation, the app may play back the replay: the sealed demo checkpoint's
stored predictions on the held-out test images and their sealed corruptions, in the sealed order,
under the sealed review threshold. Nothing is selected or tuned here, and no metric is reported
from it.

Opening the view checks identities instead of the whole code tree (D-052). These checks run
before any test image is read:
- the sealed plan is intact: its own hash and its "sealed" ledger record;
- the replay-relevant protocol modules still have their sealed hashes;
- the corruption protocol, the expected test IDs, the class map, the demo checkpoint and the
  replay manifest all match the plan;
- the robustness evaluation is recorded once under the seal, and every file it wrote still has
  its recorded hash.
Each displayed derivative is then regenerated from its parent image and must reproduce its sealed
pixel checksum.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fewshotlab.data.manifest import ImageRecord, load_class_map, read_manifest
from fewshotlab.data.registry import class_map_path, manifest_path
from fewshotlab.engine import artifacts
from fewshotlab.engine.seal import ROBUSTNESS_DIR, _normalize, _sealed_studies, seal_dir
from fewshotlab.errors import BlockedError
from fewshotlab.hashing import sha256_file, sha256_json
from fewshotlab.io_utils import read_json
from fewshotlab.paths import ProjectPaths
from fewshotlab.simulation import corruptions, review
from fewshotlab.simulation.replay import REPLAY_FILE

REPLAY_MODULES = (
    "src/fewshotlab/data/imageio.py",
    "src/fewshotlab/data/transforms.py",
    "src/fewshotlab/simulation/corruptions.py",
    "src/fewshotlab/simulation/replay.py",
    "src/fewshotlab/simulation/review.py",
)
_CONDITION_TEXT = {
    "brightness": ("brightness", "factor", "x{}"),
    "gaussian_blur": ("Gaussian blur", "sigma", "sigma {} px"),
    "gaussian_noise": ("Gaussian noise", "sd", "SD {}"),
}


class ReplayUnavailable(BlockedError):
    """The sealed replay cannot be shown; the message says what is missing or changed."""


@dataclass(frozen=True)
class Stream:
    stream_id: str
    condition: str
    severity: int
    parameters: dict[str, float]

    @property
    def label(self) -> str:
        if self.condition == "clean":
            return "clean (no corruption)"
        name, key, fmt = _CONDITION_TEXT.get(self.condition, (self.condition, "", "{}"))
        return f"stress: {name} {fmt.format(self.parameters.get(key, '?'))}"


@dataclass(frozen=True)
class ReplayItem:
    """One image of a stream: sealed order, stored sealed prediction, review flag and truth."""

    position: int
    stream_id: str
    parent_image_id: str
    derivative_id: str
    condition: str
    severity: int
    parameters: dict[str, float]
    seed: int | None
    true_class_id: int
    pred_class_id: int
    max_score: float
    review: bool

    @property
    def correct(self) -> bool:
        return self.pred_class_id == self.true_class_id


@dataclass(frozen=True)
class SealedReplay:
    identity: dict[str, Any]
    class_names: tuple[str, ...]
    threshold: float
    threshold_rule: str
    streams: tuple[Stream, ...]
    summaries: dict[str, dict[str, Any]]  # stored per-stream results of the sealed evaluation
    _items: dict[str, tuple[ReplayItem, ...]]
    _checksums: dict[tuple[str, str], str]
    _records: dict[str, ImageRecord]
    _data_root: Path

    def items(self, stream_id: str) -> tuple[ReplayItem, ...]:
        return self._items[stream_id]

    def tally(self, stream_id: str, shown: int | None = None) -> dict[str, Any]:
        """Counts over the first ``shown`` items of a stream, each with its denominator."""
        items = self._items[stream_id][: shown if shown is not None else None]
        n = len(items)
        reviewed = sum(i.review for i in items)
        accepted = n - reviewed
        accepted_errors = sum((not i.review) and (not i.correct) for i in items)
        return {
            "shown": n,
            "of": len(self._items[stream_id]),
            "reviewed": reviewed,
            "accepted": accepted,
            "accepted_errors": accepted_errors,
            "errors": sum(not i.correct for i in items),
            "coverage": accepted / n if n else None,
            "accepted_error_rate": accepted_errors / accepted if accepted else None,
        }

    def pixels(self, item: ReplayItem) -> np.ndarray:
        """The derivative's pixels (H, W, 3, uint8), regenerated and checked against the seal."""
        record = self._records[item.parent_image_id]
        x = corruptions.decode01(self._data_root / record.relative_path)
        y, made = corruptions.corrupt(x, item.condition, item.severity, item.parent_image_id)
        if made["checksum_sha256"] != self._checksums.get((item.parent_image_id, item.stream_id)):
            raise ReplayUnavailable(
                f"{item.derivative_id}: the regenerated pixels differ from the sealed checksum"
            )
        image = (y.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        if image.shape[0] == 1:
            image = image.expand(3, -1, -1)
        return image.permute(1, 2, 0).contiguous().numpy()


def _fail(study_id: str, problems: list[str]) -> ReplayUnavailable:
    return ReplayUnavailable(
        f"the sealed replay of {study_id} does not verify, so nothing is shown (fail closed):\n  - "
        + "\n  - ".join(problems)
    )


def _read_manifest_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["position"] = int(row["position"])
        row["severity"] = int(row["severity"])
        row["true_class_id"] = int(row["true_class_id"])
        row["parameters"] = json.loads(row["parameters"]) if row["parameters"] else {}
        row["seed"] = int(row["seed"]) if row["seed"] else None
    return rows


def load_sealed_replay(study: dict[str, Any], paths: ProjectPaths) -> SealedReplay:
    """Verify the sealed replay's identities, then return the read-only view."""
    study_id = study["study_id"]
    sealed = _sealed_studies(paths).get(study_id)
    if sealed is None:
        raise ReplayUnavailable(
            f"study {study_id} has no intact sealed plan; the replay exists only after "
            "`lock-test --plan PATH` and `evaluate --split test --study PATH`"
        )
    plan, events = sealed["plan"], sealed["events"]
    content, token = plan["content"], plan["audit_token"]
    demo = content.get("demo")
    if not demo:
        raise ReplayUnavailable(f"the sealed plan of {study_id} names no demo checkpoint")
    selected, dataset = demo["selected"], demo["target"]["dataset"]
    problems: list[str] = []

    sealed_modules = content["code"]["protocol_modules"]
    for rel in REPLAY_MODULES:
        path = paths.root / rel
        if (sha256_file(path) if path.is_file() else None) != sealed_modules.get(rel):
            problems.append(f"{rel} changed after sealing")
    if _normalize(corruptions.protocol()) != content["corruptions"]:
        problems.append("the corruption protocol differs from the sealed one")

    records = read_manifest(manifest_path(paths, dataset), splits=("test",))
    expected = content["data"][dataset]["expected_test_ids"]
    if sha256_json(sorted(r.image_id for r in records)) != expected["sha256"]:
        problems.append(f"the {dataset} test IDs differ from the sealed list")
    cmap = load_class_map(class_map_path(paths, dataset))
    if cmap.sha256() != content["data"][dataset]["class_map_sha256"]:
        problems.append(f"the {dataset} class map differs from the sealed one")

    manifest_file = seal_dir(paths, study_id) / REPLAY_FILE
    if not manifest_file.is_file() or sha256_file(manifest_file) != content["replay"]["csv_sha256"]:
        problems.append(f"{REPLAY_FILE} is missing or differs from the sealed hash")

    adir = artifacts.run_root(paths, selected["run_id"]) / selected["attempt_id"]
    sealed_run = next((r for r in content["runs"] if r["run_id"] == selected["run_id"]), None)
    if sealed_run is None:
        problems.append("the demo checkpoint is not one of the sealed runs")
    else:
        checkpoint = adir / sealed_run["artifact"]["file"]
        if not checkpoint.is_file() or sha256_file(checkpoint) != sealed_run["artifact"]["sha256"]:
            problems.append("the demo checkpoint is missing or differs from the sealed hash")

    recorded = [
        e
        for e in events
        if e.get("event") == "robustness_evaluation"
        and e.get("audit_token") == token
        and e.get("run_id") == selected["run_id"]
        and e.get("attempt_id") == selected["attempt_id"]
    ]
    if len(recorded) != 1:
        problems.append(
            f"expected one recorded robustness evaluation under the seal, found {len(recorded)}"
        )
    else:
        for name, digest in recorded[0]["files"].items():
            path = adir / name
            if not path.is_file() or sha256_file(path) != digest:
                problems.append(f"{ROBUSTNESS_DIR} file {Path(name).name} is missing or changed")
    if problems:
        raise _fail(study_id, problems)

    threshold = float(demo["threshold"]["selected"]["threshold"])
    out_dir = adir / ROBUSTNESS_DIR
    doc = read_json(out_dir / "robustness_test.json")
    if float(doc["threshold"]) != threshold or doc["audit_token"] != token:
        raise _fail(study_id, ["the stored robustness results name another threshold or seal"])
    with (out_dir / "corruption_checksums.csv").open(encoding="utf-8", newline="") as handle:
        checksums = {
            (row["parent_image_id"], row["condition"]): row["derivative_sha256"]
            for row in csv.DictReader(handle)
        }

    rows = _read_manifest_rows(manifest_file)
    stream_ids = list(dict.fromkeys(row["stream"] for row in rows))
    items: dict[str, tuple[ReplayItem, ...]] = {}
    streams = []
    for stream_id in stream_ids:
        mine = sorted((r for r in rows if r["stream"] == stream_id), key=lambda r: r["position"])
        stored = artifacts.read_predictions(out_dir / f"predictions_test__{stream_id}.csv")
        where = {image_id: n for n, image_id in enumerate(stored["image_ids"])}
        built = []
        for row in mine:
            n = where.get(row["parent_image_id"])
            if n is None or int(stored["true"][n]) != row["true_class_id"]:
                raise _fail(study_id, [f"{row['derivative_id']} has no matching stored prediction"])
            score = float(np.max(stored["scores"][n]))
            built.append(
                ReplayItem(
                    position=row["position"],
                    stream_id=stream_id,
                    parent_image_id=row["parent_image_id"],
                    derivative_id=row["derivative_id"],
                    condition=row["condition"],
                    severity=row["severity"],
                    parameters=row["parameters"],
                    seed=row["seed"],
                    true_class_id=row["true_class_id"],
                    pred_class_id=int(stored["pred"][n]),
                    max_score=score,
                    review=bool(review.review_flags([score], threshold)[0]),
                )
            )
        items[stream_id] = tuple(built)
        streams.append(
            Stream(stream_id, mine[0]["condition"], mine[0]["severity"], mine[0]["parameters"])
        )

    summaries = {
        c["condition_id"]: {
            "n_images": c["n_images"],
            "macro_f1": c["macro_f1"],
            "top1": c["top1"],
            "coverage": c["review"]["coverage"],
            "accepted_error_rate": c["review"]["accepted_error_rate"],
            "review_rate": c["review"]["review_rate"],
        }
        for c in doc["conditions"]
    }
    identity = {
        "study_id": study_id,
        "audit_token": token,
        "plan_sha256": plan["plan_sha256"],
        "sealed_utc": plan["sealed_utc"],
        "dataset": dataset,
        "run_id": selected["run_id"],
        "attempt_id": selected["attempt_id"],
        "method": selected["method"],
        "k": demo["target"]["k"],
        "seed": demo["target"]["seed"],
        "val_macro_f1": selected["val_macro_f1"],
        "checkpoint_sha256": sealed_run["artifact"]["sha256"],
        "selection_rule": demo["rule"],
        "replay_manifest_sha256": content["replay"]["csv_sha256"],
        "order_rule": content["replay"]["order_rule"],
        "robustness_utc": recorded[0]["time_utc"],
    }
    return SealedReplay(
        identity=identity,
        class_names=tuple(cmap.names),
        threshold=threshold,
        threshold_rule=demo["threshold"]["rule"],
        streams=tuple(streams),
        summaries=summaries,
        _items=items,
        _checksums=checksums,
        _records={r.image_id: r for r in records},
        _data_root=paths.data_root,
    )
