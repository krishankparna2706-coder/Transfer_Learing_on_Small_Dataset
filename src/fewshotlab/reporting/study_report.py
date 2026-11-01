"""Validation-only study reports, regenerated entirely from stored run records.

The study plan defines the slots. A slot shows metrics only if its latest COMPLETED attempt passes
the evidence check (the stored metrics regenerate from ``predictions_val.csv``, whose hash
matches). Missing, failed, deduplicated or unverifiable slots stay visible with null metrics.
No number in a table or chart is typed in by hand.
"""

from __future__ import annotations

import csv
import io
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fewshotlab.engine import artifacts  # noqa: E402
from fewshotlab.engine.ledger import latest_status  # noqa: E402
from fewshotlab.engine.study import plan_study, slot_status  # noqa: E402
from fewshotlab.io_utils import atomic_write_json, atomic_write_text, read_json, utc_now  # noqa: E402
from fewshotlab.paths import ProjectPaths  # noqa: E402
from fewshotlab.reporting.aggregate import collect_runs, verify_evidence  # noqa: E402

COLUMNS = (
    "dataset",
    "k",
    "method",
    "seed",
    "status",
    "macro_f1",
    "top1",
    "balanced_accuracy",
    "top5",
    "selected_epoch",
    "epochs_run",
    "stop_reason",
    "optimizer_steps",
    "trainable_parameters",
    "fit_labels",
    "validation_labels",
    "train_wall_seconds",
    "mean_fit_seconds_per_epoch",
    "val_pass_ms_per_image",
    "device",
    "run_id",
    "attempt_id",
    "predictions_sha256",
    "artifact_sha256",
)
BANNER = (
    "VALIDATION ONLY: exploratory development results, n = 1 run per cell. Each learned model's "
    "checkpoint was selected on these same validation images, so its numbers are optimistic. "
    "None of them is a held-out test result, and they are not comparable with published "
    "Flowers102 test benchmarks."
)


def slot_row(paths: ProjectPaths, slot) -> dict[str, Any]:
    state = slot_status(paths, slot)
    row: dict[str, Any] = {column: None for column in COLUMNS}
    row.update(
        dataset=slot.dataset,
        k=slot.k,
        method=slot.method,
        seed=slot.seed,
        status=state["status"],
        run_id=slot.run_id,
        attempt_id=state.get("attempt_id"),
    )
    if slot.duplicate_of:
        row["status"] = f"DEDUPLICATED (= {slot.duplicate_of})"
        return row
    if state["status"] != "COMPLETED":
        return row
    attempt_dir = artifacts.run_root(paths, slot.run_id) / state["attempt_id"]
    problem = verify_evidence(attempt_dir)
    if problem:
        row.update(status="EVIDENCE_FAILED", problem=problem)
        return row
    doc = read_json(attempt_dir / "val_metrics.json")
    metrics, timing = doc["metrics"], doc["timing"]
    fits = timing.get("fit_seconds_per_epoch")
    row.update(
        macro_f1=metrics["macro_f1"],
        top1=metrics["top1"],
        balanced_accuracy=metrics["balanced_accuracy"],
        top5=metrics.get("top5"),
        selected_epoch=doc["selected_epoch"],
        epochs_run=doc["epochs_run"],
        stop_reason=doc["stop_reason"],
        optimizer_steps=doc["optimizer_steps"],
        trainable_parameters=doc.get("parameters", {}).get("trainable"),
        fit_labels=doc["label_budget"]["fit_labels"],
        validation_labels=doc["label_budget"]["validation_labels"],
        train_wall_seconds=timing.get("total_wall_seconds"),
        mean_fit_seconds_per_epoch=(sum(fits) / len(fits))
        if fits
        else timing.get("embedding_fit_seconds"),
        val_pass_ms_per_image=timing.get("val_pass_ms_per_image"),
        device=doc["device"],
        predictions_sha256=doc["predictions"]["sha256"],
        artifact_sha256=doc["artifact"]["sha256"],
    )
    row["_attempt_dir"] = attempt_dir
    row["_per_class"] = metrics["per_class"]
    return row


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _plot(rows: list[dict[str, Any]], path: Path, title: str) -> Path | None:
    done = [r for r in rows if r["macro_f1"] is not None]
    if not done:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    methods = sorted(
        {r["method"] for r in done}, key=["scratch", "frozen", "partial", "prototypes"].index
    )
    for metric, axis in zip(("macro_f1", "top1"), axes, strict=True):
        for method in methods:
            points = sorted((r["k"], r[metric]) for r in done if r["method"] == method)
            axis.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=method)
            for k, value in points:
                axis.annotate(
                    f"{value:.2f}",
                    (k, value),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                )
        axis.set_xscale("log")
        ks = sorted({r["k"] for r in done})
        axis.set_xticks(ks, [str(k) for k in ks])
        axis.set_xlabel("K training images per class")
        axis.set_ylabel(f"validation {metric.replace('_', '-')}")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def build_study_report(
    study: dict[str, Any],
    paths: ProjectPaths,
    out_root: Path,
    *,
    sanity: bool = True,
    errors: bool = True,
) -> dict[str, Any]:
    slots = plan_study(study, paths)
    rows = [slot_row(paths, slot) for slot in slots]
    out = Path(out_root) / study["study_id"]
    out.mkdir(parents=True, exist_ok=True)
    public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=list(COLUMNS), extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(public)
    atomic_write_text(out / "comparison.csv", buffer.getvalue())
    per_class = io.StringIO(newline="")
    pc_writer = csv.writer(per_class, lineterminator="\n")
    pc_writer.writerow(
        ["run_id", "method", "k", "class_id", "precision", "recall", "f1", "support", "predicted"]
    )
    for row in rows:
        for entry in row.get("_per_class") or []:
            pc_writer.writerow(
                [
                    row["run_id"],
                    row["method"],
                    row["k"],
                    entry["class_id"],
                    entry["precision"],
                    entry["recall"],
                    entry["f1"],
                    entry["support"],
                    entry["predicted"],
                ]
            )
    atomic_write_text(out / "per_class.csv", per_class.getvalue())
    fixture = study["category"] == "fixture"
    title = (
        f"{study['study_id']}: VALIDATION ONLY{' SOFTWARE FIXTURE' if fixture else ''}, "
        f"n=1 run per point (seeds {study['support_seeds']}); not a test result"
    )
    plot = _plot(rows, out / "plots" / "val_metrics_by_k.png", title)
    for row in rows:  # copy per-run plots (no dataset images) next to the report
        if row.get("_attempt_dir"):
            for name in ("learning_curves.png", "confusion_val.png"):
                source = row["_attempt_dir"] / "plots" / name
                if source.is_file():
                    target = out / "runs" / row["run_id"] / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
    sanity_results = None
    if sanity:
        from fewshotlab.reporting.sanity import run_sanity_checks

        sanity_results = run_sanity_checks(paths, rows, out / "sanity")
    gallery = None
    if errors:
        from fewshotlab.reporting.errors import error_galleries

        gallery = error_galleries(paths, rows, out / "errors")
    summary = {
        "study": study,
        "generated_utc": utc_now(),
        "banner": BANNER,
        "rows": public,
        "plot": str(plot.relative_to(out)) if plot else None,
        "sanity": sanity_results,
        "error_galleries": gallery,
    }
    atomic_write_json(out / "comparison.json", summary)
    atomic_write_text(out / "comparison.md", _markdown(study, rows, summary, fixture))
    return {**summary, "out": out, "rows_full": rows}


def _markdown(study, rows, summary, fixture) -> str:
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    lines = [
        f"# {study['study_id']}: validation comparison",
        "",
        f"> **{BANNER}**" + (" **SOFTWARE FIXTURE: not real data.**" if fixture else ""),
        "",
        f"Generated {summary['generated_utc']} from the run ledger and the stored prediction "
        f"files. Slot statuses: {statuses}. Every metric value regenerates from its run's "
        "`predictions_val.csv`.",
        "",
        "| Dataset | K | Method | Status | Macro-F1 | Top-1 | Balanced acc. | Top-5 | "
        "Selected epoch | Epochs run | Optimizer steps | Trainable params | Fit / val labels "
        "| Wall time (s) |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for r in rows:
        epoch = (
            "class means (no epochs)"
            if r["method"] == "prototypes" and r["status"] == "COMPLETED"
            else _fmt(r["selected_epoch"])
        )
        lines.append(
            f"| {r['dataset']} | {r['k']} | {r['method']} | {r['status']} | {_fmt(r['macro_f1'])} "
            f"| {_fmt(r['top1'])} | {_fmt(r['balanced_accuracy'])} | {_fmt(r['top5'])} | {epoch} "
            f"| {_fmt(r['epochs_run'])} | {_fmt(r['optimizer_steps'])} | "
            f"{_fmt(r['trainable_parameters'])} | {_fmt(r['fit_labels'])} / "
            f"{_fmt(r['validation_labels'])} | {_fmt(r['train_wall_seconds'], 1)} |"
        )
    lines += ["", "Values are fractions in [0, 1]. `null` means not measured.", ""]
    if summary["plot"]:
        lines += [f"![validation metrics by K]({summary['plot']})", ""]
    lines += ["## Run evidence", ""]
    for r in rows:
        lines.append(
            f"- `{r['run_id']}` / `{r['attempt_id']}`: {r['status']}; predictions "
            f"sha256 `{r['predictions_sha256']}`; artifact sha256 `{r['artifact_sha256']}`"
            + (f"; problem: {r['problem']}" if r.get("problem") else "")
        )
    if summary["sanity"] is not None:
        lines += ["", "## Sanity checks", ""]
        for item in summary["sanity"]["runs"]:
            lines.append(
                f"- {item['status']} `{item['run_id']}`: "
                + ", ".join(f"{k}={v}" for k, v in item["checks"].items())
            )
        aug = summary["sanity"].get("augmentation")
        if aug:
            lines.append(f"- {aug['status']} augmentation: {aug['detail']}")
    if summary["error_galleries"] is not None:
        lines += [
            "",
            "## Error galleries (local only; they contain dataset images)",
            "",
            f"Rule, fixed in advance: {summary['error_galleries']['rule']}",
            "",
        ]
        for item in summary["error_galleries"]["runs"]:
            lines.append(
                f"- `{item['run_id']}`: {item['n_errors']} misclassified of "
                f"{item['n_images']} validation images; shown: {item['shown']}"
            )
    return "\n".join(lines) + "\n"


def ledger_summary(paths: ProjectPaths, out_root: Path) -> dict[str, Any]:
    """Report without a study: statuses by category, plus the verified core and MVP runs."""
    counts: dict[str, dict[str, int]] = {}
    for event in latest_status(paths).values():
        bucket = counts.setdefault(event.get("category", "unknown"), {})
        bucket[event["status"]] = bucket.get(event["status"], 0) + 1
    verified = collect_runs(paths, categories=("core", "mvp"))
    summary = {
        "generated_utc": utc_now(),
        "status_counts_by_category": counts,
        "accepted": verified["accepted"],
        "rejected": verified["rejected"],
    }
    out = Path(out_root)
    atomic_write_json(out / "ledger_summary.json", summary)
    lines = [
        "# Ledger summary",
        "",
        f"Generated {summary['generated_utc']}.",
        "",
        "| Category | Status counts |",
        "|---|---|",
    ]
    lines += [f"| {c} | {v} |" for c, v in sorted(counts.items())]
    lines += [
        "",
        f"Verified MVP/core runs: {len(verified['accepted'])}; excluded: "
        f"{len(verified['rejected'])}.",
        "",
    ]
    atomic_write_text(out / "ledger_summary.md", "\n".join(lines))
    return summary
