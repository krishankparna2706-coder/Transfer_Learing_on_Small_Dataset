"""Offline rehearsal of the industrial replay (P7), from a cold start to the last image.

``python -m fewshotlab demo --rehearse`` runs it without a browser:
1. **Start:** verifies the sealed replay and loads the sealed demo checkpoint, timed as the cold
   start.
2. **Forward latency:** measured with the protocol's 20 warm-up and 200 timed requests.
3. **Every image:** scores every image of every stream live, times each one end to end, and
   compares it with the stored sealed prediction.
4. **The recording:** writes the following under ``demo_assets/<study>/``:
   - ``rehearsal.json``, the summary;
   - ``rehearsal.md``, a readable report;
   - ``rehearsal_events.csv``, one row per image, kept local like other per-image outputs.
   A ``replay_rehearsal`` event in the run ledger records the files' hashes.

Every network connection attempted during the rehearsal is refused at once, so a hidden download
cannot pass unnoticed. This is a demonstration record, never a benchmark result.
"""

from __future__ import annotations

import csv
import io
import json
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fewshotlab.engine.ledger import append_event
from fewshotlab.errors import BlockedError
from fewshotlab.hashing import sha256_file
from fewshotlab.io_utils import atomic_write_bytes, atomic_write_json, atomic_write_text, utc_now
from fewshotlab.paths import ProjectPaths, portable_path
from fewshotlab.simulation.replay_live import (
    TIMED_REQUESTS,
    WARMUP_REQUESTS,
    LiveReplay,
    LiveResult,
    describe,
    environment,
)
from fewshotlab.simulation.sealed_replay import load_sealed_replay

SUMMARY_FILE = "rehearsal.json"
REPORT_FILE = "rehearsal.md"
EVENTS_FILE = "rehearsal_events.csv"
EVENT_COLUMNS = (
    "sequence_id",
    "position",
    "stream",
    "derivative_id",
    "parent_image_id",
    "condition",
    "severity",
    "seed",
    "time_utc",
    "run_id",
    "attempt_id",
    "predicted_class",
    "model_score",
    "review",
    "true_class",
    "stored_predicted_class",
    "stored_model_score",
    "matches_stored",
    "preprocess_ms",
    "forward_ms",
    "end_to_end_ms",
    "device",
)


class NetworkAttempt(BlockedError):
    """A network connection was attempted during the offline rehearsal."""


@contextmanager
def no_network() -> Iterator[None]:
    """Refuse every outgoing socket connection inside the block."""
    original = socket.socket.connect

    def refuse(_sock: socket.socket, address: Any) -> None:
        raise NetworkAttempt(f"network access attempted during the offline rehearsal: {address!r}")

    socket.socket.connect = refuse  # type: ignore[method-assign]
    try:
        yield
    finally:
        socket.socket.connect = original  # type: ignore[method-assign]


def output_dir(paths: ProjectPaths, study_id: str) -> Path:
    return paths.demo_assets / study_id


def _event(sequence: str, result: LiveResult, names: tuple[str, ...], identity: dict, device: str):
    item = result.item
    return {
        "sequence_id": sequence,
        "position": item.position,
        "stream": item.stream_id,
        "derivative_id": item.derivative_id,
        "parent_image_id": item.parent_image_id,
        "condition": item.condition,
        "severity": item.severity,
        "seed": "" if item.seed is None else item.seed,
        "time_utc": utc_now(),
        "run_id": identity["run_id"],
        "attempt_id": identity["attempt_id"],
        "predicted_class": names[result.pred_class_id],
        "model_score": f"{result.max_score:.6f}",
        "review": result.review,
        "true_class": names[item.true_class_id],
        "stored_predicted_class": names[item.pred_class_id],
        "stored_model_score": f"{item.max_score:.6f}",
        "matches_stored": result.matches_stored,
        "preprocess_ms": f"{result.preprocess_ms:.3f}",
        "forward_ms": f"{result.forward_ms:.3f}",
        "end_to_end_ms": f"{result.end_to_end_ms:.3f}",
        "device": device,
    }


def _stats(values: list[float]) -> dict[str, Any]:
    return {k: v for k, v in describe(values).items() if k != "samples_ms"}


def rehearse(
    study: dict[str, Any],
    paths: ProjectPaths,
    *,
    device: str = "cpu",
    warmup: int = WARMUP_REQUESTS,
    timed: int = TIMED_REQUESTS,
    record: bool = True,
) -> dict[str, Any]:
    """Run the whole replay offline and write its recording; returns the summary."""
    started = utc_now()
    sequence = "rehearsal-" + started.replace("-", "").replace(":", "")
    with no_network():
        view = load_sealed_replay(study, paths)
        live = LiveReplay.load(view, paths, device)
        forward = live.time_forward(
            view.items(view.streams[0].stream_id)[0], warmup=warmup, timed=timed
        )
        identity, names = view.identity, view.class_names
        events, results, streams = [], [], []
        for stream in view.streams:
            mine = [live.infer(item) for item in view.items(stream.stream_id)]
            results += mine
            events += [_event(sequence, r, names, identity, device) for r in mine]
            accepted = [r for r in mine if not r.review]
            wrong = sum(r.pred_class_id != r.item.true_class_id for r in accepted)
            streams.append(
                {
                    "stream": stream.stream_id,
                    "label": stream.label,
                    "images": len(mine),
                    "live_matches_stored": sum(r.matches_stored for r in mine),
                    "max_score_difference": max(r.score_difference for r in mine),
                    "reviewed": len(mine) - len(accepted),
                    "coverage": len(accepted) / len(mine),
                    "accepted_errors": wrong,
                    "accepted_error_rate": wrong / len(accepted) if accepted else None,
                    "stored": view.summaries[stream.stream_id],
                }
            )
    clean_errors = [
        {
            "image_id": r.item.parent_image_id,
            "true_class": names[r.item.true_class_id],
            "predicted_class": names[r.pred_class_id],
            "model_score": round(r.max_score, 6),
            "review": r.review,
        }
        for r in sorted(
            (r for r in results if r.item.stream_id == "clean"),
            key=lambda r: r.item.parent_image_id,
        )
        if r.pred_class_id != r.item.true_class_id
    ]
    summary = {
        "schema_version": 1,
        "kind": "replay rehearsal: a demonstration record, not a benchmark result",
        "study_id": study["study_id"],
        "sequence_id": sequence,
        "started_utc": started,
        "finished_utc": utc_now(),
        "seal": {k: identity[k] for k in ("audit_token", "plan_sha256", "sealed_utc")},
        "model": {
            k: identity[k]
            for k in ("run_id", "attempt_id", "method", "k", "seed", "checkpoint_sha256")
        },
        "threshold": view.threshold,
        "threshold_rule": view.threshold_rule,
        "network": "every outgoing connection was refused during the rehearsal; none was attempted",
        "environment": environment(device),
        "cold_start_ms": live.cold_start_ms,
        "forward_latency": forward,
        "per_image": {
            "images": len(results),
            "preprocess": _stats([r.preprocess_ms for r in results]),
            "forward": _stats([r.forward_ms for r in results]),
            "end_to_end": _stats([r.end_to_end_ms for r in results]),
            "end_to_end_excludes": "browser rendering (the rehearsal runs without a browser)",
        },
        "playback": "the app shows one image per second: a presentation pace, not a speed figure",
        "streams": streams,
        "all_live_match_stored": all(r.matches_stored for r in results),
        "clean_errors": clean_errors,
        "error_gallery_rule": "every clean-stream test image the sealed model gets wrong, sorted "
        "by image ID (fixed before looking at the errors)",
    }
    if record:
        out = output_dir(paths, study["study_id"])
        out.mkdir(parents=True, exist_ok=True)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=EVENT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(events)
        atomic_write_bytes(out / EVENTS_FILE, buffer.getvalue().encode("utf-8"))
        atomic_write_json(out / SUMMARY_FILE, summary)
        atomic_write_text(out / REPORT_FILE, report_markdown(summary))
        files = {name: sha256_file(out / name) for name in (SUMMARY_FILE, REPORT_FILE, EVENTS_FILE)}
        append_event(
            paths,
            event="replay_rehearsal",
            study_id=study["study_id"],
            audit_token=identity["audit_token"],
            run_id=identity["run_id"],
            attempt_id=identity["attempt_id"],
            device=device,
            sequence_id=sequence,
            all_live_match_stored=summary["all_live_match_stored"],
            folder=portable_path(out),
            files=files,
        )
        summary["files"] = files
    return summary


def _ms(stats: dict[str, Any]) -> str:
    return f"median {stats['median_ms']:.2f} ms, 95th percentile {stats['p95_ms']:.2f} ms"


def report_markdown(summary: dict[str, Any]) -> str:
    env, fwd, per = summary["environment"], summary["forward_latency"], summary["per_image"]
    model, seal = summary["model"], summary["seal"]
    lines = [
        f"# Replay rehearsal: {summary['study_id']}",
        "",
        "> **Demonstration record, not a benchmark result.** The sealed demo checkpoint scored the "
        "sealed NEU replay live, offline, and every result was compared with the stored sealed "
        "prediction.",
        "",
        f"Sequence `{summary['sequence_id']}`, {summary['started_utc']} to "
        f"{summary['finished_utc']}.",
        "",
        "| Item | Observed |",
        "|---|---|",
        f"| Seal | `{seal['audit_token']}` (sealed {seal['sealed_utc']}) |",
        f"| Model | `{model['run_id']}` / `{model['attempt_id']}` ({model['method']}, "
        f"K={model['k']}, seed {model['seed']}); checkpoint SHA-256 "
        f"`{model['checkpoint_sha256'][:16]}...` |",
        f"| Review threshold | {summary['threshold']:g} (sealed on validation) |",
        f"| Device | {env['device']}; {env['cpu_threads']} CPU threads; {env['precision']}; "
        f"synchronization: {env['synchronization']}; {env['machine']} |",
        f"| Network | {summary['network']} |",
        f"| Cold start (identify and load the checkpoint) | {summary['cold_start_ms']:.0f} ms |",
        f"| Forward latency (batch 1, {fwd['warmup_requests']} warm-up, {fwd['timed_requests']} "
        f"timed, input {fwd['input_shape']}) | {_ms(fwd)} |",
        f"| Per image, end to end ({per['images']:,} images; browser rendering excluded) | "
        f"{_ms(per['end_to_end'])} |",
        f"| Per image, preprocessing (decode, corrupt, transform) | {_ms(per['preprocess'])} |",
        f"| Per image, scoring call | {_ms(per['forward'])} |",
        f"| Playback | {summary['playback']} |",
        "",
        "## Live versus stored sealed predictions",
        "",
        "| Stream | Images | Live = stored | Largest score difference | Coverage (live) | "
        "Errors among accepted (live) | Coverage (stored) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary["streams"]:
        rate = s["accepted_error_rate"]
        lines.append(
            f"| {s['label']} | {s['images']} | {s['live_matches_stored']} | "
            f"{s['max_score_difference']:.1e} | {s['coverage']:.1%} | "
            f"{s['accepted_errors']} ({'n/a' if rate is None else f'{rate:.1%}'}) | "
            f"{s['stored']['coverage']:.1%} |"
        )
    verdict = "all match" if summary["all_live_match_stored"] else "MISMATCHES FOUND"
    lines += [
        "",
        f"**Result:** {verdict}. Coverage and the error among accepted predictions are read "
        "together; NEU-CLS has no defect-free class, so nothing here measures false accepts.",
        "",
        f"## Fixed error gallery: clean stream ({len(summary['clean_errors'])} errors)",
        "",
        f"Rule: {summary['error_gallery_rule']}.",
        "",
        "| Image | True class | Predicted | Model score | Decision |",
        "|---|---|---|---:|---|",
    ]
    for e in summary["clean_errors"]:
        decision = "human review" if e["review"] else "ACCEPTED (wrong)"
        lines.append(
            f"| `{e['image_id']}` | {e['true_class']} | {e['predicted_class']} | "
            f"{e['model_score']:.4f} | {decision} |"
        )
    return "\n".join(lines) + "\n"


def load_summary(paths: ProjectPaths, study_id: str) -> dict[str, Any] | None:
    path = output_dir(paths, study_id) / SUMMARY_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
