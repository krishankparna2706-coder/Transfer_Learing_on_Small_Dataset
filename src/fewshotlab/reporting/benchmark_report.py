"""Seed-aggregated benchmark tables and plots, regenerated from stored predictions.

* Validation aggregates are development evidence. Each learned checkpoint was selected on the
  same validation images, so the values are optimistic.
* Test aggregates use only runs evaluated under an intact seal. Every value is recomputed from
  the run's ``predictions_test.csv``, whose hash must match the study's evaluation ledger.

Per dataset x K x method:
* logical slots and distinct runs (a deterministic duplicate counts once)
* mean and sample SD (ddof=1) over distinct runs (SD is null below two distinct runs)
* min and max

Paired differences are computed per dataset x K and method pair: the per-seed difference A - B
over seeds where both slots have a value, with mean, sample SD, min, max and the number of seeds
where A > B. Five seeds describe this benchmark's support and initialization variability. No
population confidence interval is inferred from them.
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fewshotlab import PROTOCOL_ID  # noqa: E402
from fewshotlab.data.manifest import load_class_map, read_manifest  # noqa: E402
from fewshotlab.data.registry import class_map_path, manifest_path  # noqa: E402
from fewshotlab.engine import artifacts  # noqa: E402
from fewshotlab.engine.study import plan_study, slot_status  # noqa: E402
from fewshotlab.hashing import sha256_file  # noqa: E402
from fewshotlab.io_utils import atomic_write_json, atomic_write_text, read_json, utc_now  # noqa: E402
from fewshotlab.metrics.core import classification_metrics  # noqa: E402
from fewshotlab.paths import ProjectPaths  # noqa: E402
from fewshotlab.reporting.aggregate import verify_evidence  # noqa: E402

METHOD_ORDER = ("scratch", "frozen", "partial", "prototypes")
PAIRS = (
    ("frozen", "scratch"),
    ("partial", "scratch"),
    ("prototypes", "scratch"),
    ("partial", "frozen"),
    ("prototypes", "frozen"),
    ("partial", "prototypes"),
)
METRICS = ("macro_f1", "top1", "balanced_accuracy")
BANNERS = {
    "val": "VALIDATION ONLY: development results. Each learned checkpoint was selected on these "
    "same validation images, so the numbers are optimistic. They are not held-out test results.",
    "test": "HELD-OUT TEST under a sealed plan: every model, threshold and rule was fixed on "
    "validation before test access, and each run was evaluated on test once.",
}


def _method_key(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


def _val_metrics(paths: ProjectPaths, slot) -> tuple[str, dict[str, Any] | None]:
    state = slot_status(paths, slot)
    if state["status"] != "COMPLETED":
        return state["status"], None
    attempt_dir = artifacts.run_root(paths, slot.run_id) / state["attempt_id"]
    if verify_evidence(attempt_dir):
        return "EVIDENCE_FAILED", None
    return "COMPLETED", read_json(attempt_dir / "val_metrics.json")["metrics"]


def recorded_test_files(paths: ProjectPaths, study_id: str) -> dict[tuple[str, str], dict]:
    """(run_id, attempt_id) -> recorded files, from the study's intact seal (empty otherwise)."""
    from fewshotlab.engine.seal import _sealed_studies

    sealed = _sealed_studies(paths).get(study_id)
    if not sealed:
        return {}
    token = sealed["plan"]["audit_token"]
    return {
        (e["run_id"], e["attempt_id"]): e["files"]
        for e in sealed["events"]
        if e.get("event") == "test_evaluation" and e.get("audit_token") == token
    }


def _test_metrics(
    paths: ProjectPaths, slot, recorded: dict[tuple[str, str], dict]
) -> tuple[str, dict[str, Any] | None]:
    state = slot_status(paths, slot)
    if state["status"] != "COMPLETED":
        return state["status"], None
    files = recorded.get((slot.run_id, state["attempt_id"]))
    if not files:
        return "NOT_EVALUATED", None
    attempt_dir = artifacts.run_root(paths, slot.run_id) / state["attempt_id"]
    for name in ("predictions_test.csv", "test_metrics.json"):
        path = attempt_dir / name
        if not path.is_file() or sha256_file(path) != files.get(name):
            return "EVIDENCE_FAILED", None
    doc = read_json(attempt_dir / "test_metrics.json")
    stored = artifacts.read_predictions(attempt_dir / "predictions_test.csv")
    topk = (5,) if slot.dataset == "flowers102" else ()
    again = classification_metrics(
        stored["true"],
        stored["pred"],
        doc["metrics"]["num_classes"],
        scores=stored["scores"],
        topk=topk,
    )
    keys = (*METRICS, *(f"top{k}" for k in topk), "confusion")
    if any(again[key] != doc["metrics"][key] for key in keys):
        return "EVIDENCE_FAILED", None
    return "EVALUATED", doc["metrics"]


def slot_values(study: dict[str, Any], paths: ProjectPaths, split: str) -> list[dict[str, Any]]:
    slots = plan_study(study, paths)
    by_run = {s.run_id: s for s in slots}
    recorded = recorded_test_files(paths, study["study_id"]) if split == "test" else {}
    cache: dict[str, tuple[str, dict[str, Any] | None]] = {}
    rows = []
    for slot in slots:
        source = by_run.get(slot.duplicate_of, slot) if slot.duplicate_of else slot
        if source.run_id not in cache:
            cache[source.run_id] = (
                _val_metrics(paths, source)
                if split == "val"
                else _test_metrics(paths, source, recorded)
            )
        status, metrics = cache[source.run_id]
        rows.append(
            {
                "dataset": slot.dataset,
                "k": slot.k,
                "method": slot.method,
                "seed": slot.seed,
                "run_id": slot.run_id,
                "source_run_id": source.run_id,
                "duplicate": bool(slot.duplicate_of),
                "status": status,
                **{m: (metrics[m] if metrics else None) for m in METRICS},
                "top5": metrics.get("top5") if metrics else None,
            }
        )
    return rows


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "sd": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) >= 2 else None,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def aggregate(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["k"], row["method"])].append(row)
    out = []
    for (dataset, k, method), items in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1], _method_key(kv[0][2]))
    ):
        distinct = {r["source_run_id"]: r[metric] for r in items if r[metric] is not None}
        out.append(
            {
                "dataset": dataset,
                "k": k,
                "method": method,
                "metric": metric,
                "n_slots": len(items),
                "n_with_value": sum(r[metric] is not None for r in items),
                "n_distinct": len(distinct),
                "duplicates": sum(r["duplicate"] for r in items),
                **_stats(list(distinct.values())),
                "seeds": sorted(r["seed"] for r in items if r[metric] is not None),
            }
        )
    return out


def paired(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    cells = {(r["dataset"], r["k"], r["method"], r["seed"]): r for r in rows}
    datasets = sorted({r["dataset"] for r in rows})
    shots = sorted({r["k"] for r in rows})
    methods = {r["method"] for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    out = []
    for dataset in datasets:
        for k in shots:
            for a, b in PAIRS:
                if a not in methods or b not in methods:
                    continue
                diffs, used = [], []
                for seed in seeds:
                    ra, rb = cells.get((dataset, k, a, seed)), cells.get((dataset, k, b, seed))
                    if ra and rb and ra[metric] is not None and rb[metric] is not None:
                        diffs.append(ra[metric] - rb[metric])
                        used.append(seed)
                out.append(
                    {
                        "dataset": dataset,
                        "k": k,
                        "a": a,
                        "b": b,
                        "metric": metric,
                        "n_pairs": len(diffs),
                        "distinct_a": len(
                            {cells[(dataset, k, a, s)]["source_run_id"] for s in used}
                        ),
                        "distinct_b": len(
                            {cells[(dataset, k, b, s)]["source_run_id"] for s in used}
                        ),
                        **{f"{key}_diff": value for key, value in _stats(diffs).items()},
                        "a_better": sum(d > 0 for d in diffs),
                        "ties": sum(d == 0 for d in diffs),
                        "seeds": used,
                    }
                )
    return out


def label_budgets(paths: ProjectPaths, study: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for dataset in study["datasets"]:
        classes = load_class_map(class_map_path(paths, dataset)).num_classes
        validation = len(read_manifest(manifest_path(paths, dataset), splits=("val",)))
        out[dataset] = {
            "classes": classes,
            "validation_labels": validation,
            "support_labels": {str(k): k * classes for k in study["shots"]},
        }
    return out


def _csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {k: (" ".join(map(str, v)) if isinstance(v, list) else v) for k, v in row.items()}
        )
    return buffer.getvalue()


def _fmt(value: float | None, digits: int = 3) -> str:
    return "null" if value is None else f"{value:.{digits}f}"


def _plot(study, aggregates, rows, budgets, split: str, path: Path) -> Path | None:
    means = [a for a in aggregates if a["metric"] == "macro_f1" and a["mean"] is not None]
    if not means:
        return None
    datasets = [d for d in study["datasets"] if any(a["dataset"] == d for a in means)]
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(5.2 * len(datasets), 4.2), constrained_layout=True, squeeze=False
    )
    for axis, dataset in zip(axes[0], datasets, strict=True):
        for offset, method in enumerate(sorted({a["method"] for a in means}, key=_method_key)):
            cells = sorted(
                (a for a in means if a["dataset"] == dataset and a["method"] == method),
                key=lambda a: a["k"],
            )
            ks = [a["k"] * (1 + 0.04 * (offset - 1.5)) for a in cells]
            axis.errorbar(
                ks,
                [a["mean"] for a in cells],
                yerr=[a["sd"] or 0.0 for a in cells],
                marker="o",
                capsize=3,
                label=method,
            )
            for a, x in zip(cells, ks, strict=True):
                values = {
                    r["source_run_id"]: r["macro_f1"]
                    for r in rows
                    if (r["dataset"], r["k"], r["method"]) == (dataset, a["k"], method)
                    and r["macro_f1"] is not None
                }
                axis.scatter([x] * len(values), list(values.values()), s=8, alpha=0.35)
        shots = sorted(study["shots"])
        axis.set_xscale("log")
        axis.set_xticks(shots, [str(k) for k in shots])
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.3)
        budget = budgets[dataset]
        axis.set_xlabel(
            f"K labelled support images per class (support = K x {budget['classes']} labels)"
        )
        axis.set_ylabel(f"{'validation' if split == 'val' else 'test'} macro-F1")
        axis.set_title(
            f"{dataset}: + {budget['validation_labels']} validation labels for selection",
            fontsize=9,
        )
        axis.legend(fontsize=8)
    n_distinct = sorted({a["n_distinct"] for a in means})
    fig.suptitle(
        f"{study['study_id']} ({PROTOCOL_ID}), {split.upper()} macro-F1: mean +/- sample SD over "
        f"distinct runs (n per point {n_distinct}; seeds {study['support_seeds']}; dots = runs)"
        + ("; VALIDATION ONLY, optimistic" if split == "val" else "; held-out test, sealed"),
        fontsize=8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _markdown(study, split, aggregates, pairs, budgets, rows, plot, generated) -> str:
    fixture = study["category"] == "fixture"
    statuses: dict[str, int] = defaultdict(int)
    for row in rows:
        statuses[row["status"] if not row["duplicate"] else "DEDUPLICATED"] += 1
    lines = [
        f"# {study['study_id']}: {'validation' if split == 'val' else 'held-out test'} "
        "aggregates by K and method",
        "",
        f"> **{BANNERS[split]}**" + (" **SOFTWARE FIXTURE: not real data.**" if fixture else ""),
        "",
        f"Generated {generated} from stored predictions (protocol {PROTOCOL_ID}). Slots: "
        f"{dict(statuses)}. Cells show mean +/- sample SD over distinct runs, with the distinct "
        "run count in brackets. SD is null when fewer than two distinct runs exist; a "
        "deterministic duplicate counts once.",
        "",
    ]
    methods = sorted({a["method"] for a in aggregates}, key=_method_key)
    for dataset in study["datasets"]:
        budget = budgets[dataset]
        lines += [
            f"## {dataset} ({budget['classes']} classes; {budget['validation_labels']} "
            "validation labels used for checkpoint selection)",
            "",
            "Macro-F1 (primary):",
            "",
            "| K | support labels | " + " | ".join(methods) + " |",
            "|---:|---:|" + "---:|" * len(methods),
        ]
        for k in sorted(study["shots"]):
            cells = []
            for method in methods:
                a = next(
                    (
                        x
                        for x in aggregates
                        if (x["dataset"], x["k"], x["method"], x["metric"])
                        == (dataset, k, method, "macro_f1")
                    ),
                    None,
                )
                cells.append(
                    "null"
                    if not a or a["mean"] is None
                    else f"{_fmt(a['mean'])} +/- {_fmt(a['sd'])} [{a['n_distinct']}]"
                )
            lines.append(
                f"| {k} | {budget['support_labels'][str(k)]} | " + " | ".join(cells) + " |"
            )
        lines += ["", "Top-1 and balanced accuracy (mean over distinct runs):", ""]
        lines += [
            "| K | " + " | ".join(f"{m} top-1 / bal." for m in methods) + " |",
            "|---:|" + "---|" * len(methods),
        ]
        for k in sorted(study["shots"]):
            cells = []
            for method in methods:
                pick = {
                    x["metric"]: x
                    for x in aggregates
                    if (x["dataset"], x["k"], x["method"]) == (dataset, k, method)
                }
                cells.append(
                    f"{_fmt(pick.get('top1', {}).get('mean'))} / "
                    f"{_fmt(pick.get('balanced_accuracy', {}).get('mean'))}"
                )
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "Paired macro-F1 differences (A - B, matched support seeds):",
            "",
            "| K | A - B | n pairs | mean | SD | min | max | A better |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
        for p in pairs:
            if p["dataset"] != dataset or p["metric"] != "macro_f1":
                continue
            note = (
                " (one distinct run on a side)"
                if min(p["distinct_a"], p["distinct_b"]) == 1 and p["n_pairs"] > 1
                else ""
            )
            lines.append(
                f"| {p['k']} | {p['a']} - {p['b']}{note} | {p['n_pairs']} | {_fmt(p['mean_diff'])} "
                f"| {_fmt(p['sd_diff'])} | {_fmt(p['min_diff'])} | {_fmt(p['max_diff'])} | "
                f"{p['a_better']}/{p['n_pairs']} |"
            )
        lines.append("")
    if plot:
        lines += [f"![macro-F1 by K]({plot})", ""]
    lines += [
        "Notes:",
        "",
        "- Values are fractions in [0, 1]; `null` means not measured.",
        "- Seeds vary the support images and the initialization stream together. The spread "
        "describes this benchmark's variability across five support draws, not a population "
        "confidence interval.",
        "- Every value regenerates from stored prediction files; per-slot rows are in the CSV.",
    ]
    if split == "val":
        lines.append(
            "- Validation values are optimistic: the selection rule (best validation macro-F1 "
            "epoch) used the same images."
        )
    return "\n".join(lines) + "\n"


def write_aggregate_report(
    study: dict[str, Any], paths: ProjectPaths, out_root: Path, split: str
) -> dict[str, Any]:
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    from fewshotlab.engine.seal import TEST_REPORT_DIR

    base = Path(out_root) / study["study_id"]
    out = base if split == "val" else base / TEST_REPORT_DIR
    rows = slot_values(study, paths, split)
    if split == "test" and not any(r["status"] == "EVALUATED" for r in rows):
        # nothing evaluated under an intact seal: write nothing (a test folder before the seal
        # would itself count as a test-split output)
        return {
            "study_id": study["study_id"],
            "split": split,
            "written": False,
            "out": out,
            "rows": rows,
        }
    aggregates = [a for metric in METRICS for a in aggregate(rows, metric)]
    pairs = [p for metric in METRICS for p in paired(rows, metric)]
    budgets = label_budgets(paths, study)
    plot = _plot(
        study, aggregates, rows, budgets, split, out / "plots" / f"{split}_macro_f1_by_k.png"
    )
    generated = utc_now()
    plot_rel = str(plot.relative_to(out)) if plot else None
    atomic_write_text(out / f"slots_{split}.csv", _csv(rows))
    atomic_write_text(out / f"aggregate_{split}.csv", _csv(aggregates))
    atomic_write_text(out / f"paired_{split}.csv", _csv(pairs))
    summary = {
        "study_id": study["study_id"],
        "protocol_id": PROTOCOL_ID,
        "split": split,
        "banner": BANNERS[split],
        "generated_utc": generated,
        "label_budgets": budgets,
        "aggregates": aggregates,
        "paired": pairs,
        "plot": plot_rel,
    }
    atomic_write_json(out / f"aggregate_{split}.json", summary)
    atomic_write_text(
        out / f"aggregate_{split}.md",
        _markdown(study, split, aggregates, pairs, budgets, rows, plot_rel, generated),
    )
    return {**summary, "written": True, "out": out, "rows": rows}


def write_robustness_report(
    study: dict[str, Any], paths: ProjectPaths, out_root: Path
) -> dict[str, Any] | None:
    """The sealed demo checkpoint under corruption, regenerated from its stored predictions."""
    from fewshotlab.engine.seal import ROBUSTNESS_DIR, TEST_REPORT_DIR, _sealed_studies
    from fewshotlab.simulation.review import review_metrics

    sealed = _sealed_studies(paths).get(study["study_id"])
    if not sealed:
        return None
    token = sealed["plan"]["audit_token"]
    events = [
        e
        for e in sealed["events"]
        if e.get("event") == "robustness_evaluation" and e.get("audit_token") == token
    ]
    if not events:
        return None
    event = events[-1]
    attempt_dir = artifacts.run_root(paths, event["run_id"]) / event["attempt_id"]
    problems = [
        name
        for name, digest in event["files"].items()
        if not (attempt_dir / name).is_file() or sha256_file(attempt_dir / name) != digest
    ]
    doc = read_json(attempt_dir / ROBUSTNESS_DIR / "robustness_test.json")
    num_classes = load_class_map(class_map_path(paths, doc["dataset"])).num_classes
    for row in doc["conditions"]:
        stored = artifacts.read_predictions(
            attempt_dir / ROBUSTNESS_DIR / f"predictions_test__{row['condition_id']}.csv"
        )
        again = classification_metrics(stored["true"], stored["pred"], num_classes)
        routing = review_metrics(
            stored["true"], stored["pred"], stored["scores"].max(axis=1), doc["threshold"]
        )
        if again["macro_f1"] != row["macro_f1"] or routing != row["review"]:
            problems.append(f"{row['condition_id']}: values do not regenerate")
    out = Path(out_root) / study["study_id"] / TEST_REPORT_DIR
    summary = {
        "study_id": study["study_id"],
        "generated_utc": utc_now(),
        "status": "EVIDENCE_FAILED" if problems else "VERIFIED",
        "problems": problems,
        "run_id": doc["run_id"],
        "method": doc["method"],
        "dataset": doc["dataset"],
        "threshold": doc["threshold"],
        "conditions": [
            {k: v for k, v in row.items() if k not in ("per_class", "confusion")}
            for row in doc["conditions"]
        ],
        "paired_vs_clean": doc["paired_vs_clean"],
        "clean_path_consistency": doc["clean_path_consistency"],
    }
    atomic_write_json(out / "robustness.json", summary)
    lines = [
        f"# {study['study_id']}: robustness of the sealed demo checkpoint",
        "",
        "> **ROBUSTNESS STUDY on held-out test images under controlled synthetic corruption. "
        "Reported apart from the primary benchmark. Each derivative is paired with its clean "
        "parent and is not a new test image. The review threshold was fixed on validation "
        "before test access; 80 % coverage is a workflow setting, not a safety guarantee.**",
        "",
        f"Run `{doc['run_id']}` ({doc['method']}), threshold {doc['threshold']:.2f}, evidence "
        f"{summary['status']}.",
        "",
        "| Condition | Macro-F1 | Top-1 | Coverage | Error among accepted | Review rate | "
        "Delta macro-F1 vs clean | Changed predictions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    deltas = {p["condition_id"]: p for p in doc["paired_vs_clean"]}
    for row in doc["conditions"]:
        p = deltas.get(row["condition_id"])
        lines.append(
            f"| {row['condition_id']} | {_fmt(row['macro_f1'])} | {_fmt(row['top1'])} | "
            f"{_fmt(row['review']['coverage'])} | {_fmt(row['review']['accepted_error_rate'])} | "
            f"{_fmt(row['review']['review_rate'])} | "
            f"{_fmt(p['delta_macro_f1']) if p else '(reference)'} | "
            f"{p['changed_predictions'] if p else '-'} |"
        )
    lines += [
        "",
        "Coverage and error among accepted predictions are read together: a low accepted "
        "error obtained by reviewing nearly everything comes with low coverage. NEU-CLS has no "
        "normal class, so nothing here measures false accepts on defect-free material.",
    ]
    atomic_write_text(out / "robustness.md", "\n".join(lines) + "\n")
    return {**summary, "out": out}
