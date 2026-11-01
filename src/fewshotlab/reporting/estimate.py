"""Compute-budget ESTIMATE derived from measured diagnostic runs.

The output is a planning estimate, never an observed runtime. It uses the steady-state (last
epoch) per-image costs that COMPLETED ``diagnostic`` runs recorded on this machine. It reports a
lower bound (the earliest possible early stop: 16 epochs) and an upper bound (100 epochs) for
the learned methods. Every input is traceable to a run ID.
"""

from __future__ import annotations

from typing import Any

from fewshotlab.config import NUM_CLASSES, SHOTS, SUPPORT_SEEDS
from fewshotlab.engine import artifacts
from fewshotlab.engine.ledger import latest_status
from fewshotlab.io_utils import atomic_write_json, atomic_write_text, read_json, utc_now
from fewshotlab.paths import ProjectPaths

VAL_IMAGES = {"flowers102": 1020, "neu_cls": 360}
TEST_IMAGES = {"flowers102": 6149, "neu_cls": 360}
MIN_EPOCHS, MAX_EPOCHS = 16, 100  # 1 + patience 15, and the recipe maximum
LEARNED = ("scratch", "frozen", "partial")


def diagnostic_costs(paths: ProjectPaths) -> dict[tuple[str, str], dict[str, Any]]:
    costs: dict[tuple[str, str], dict[str, Any]] = {}
    for (run_id, attempt_id), event in sorted(latest_status(paths).items()):
        if event.get("category") != "diagnostic" or event.get("status") != "COMPLETED":
            continue
        doc = read_json(artifacts.run_root(paths, run_id) / attempt_id / "val_metrics.json")
        timing, n_support = doc["timing"], doc["label_budget"]["fit_labels"]
        n_val = doc["label_budget"]["validation_labels"]
        if doc["method"] == "prototypes":
            entry = {
                "embed_s_per_image": timing["embedding_fit_seconds"] / n_support,
                "val_s_per_image": timing["val_seconds"] / n_val,
            }
        else:
            entry = {
                "fit_s_per_image": timing["fit_seconds_per_epoch"][-1] / n_support,
                "val_s_per_image": timing["val_seconds_per_epoch"][-1] / n_val,
                "first_epoch_fit_s": timing["fit_seconds_per_epoch"][0],
                "steady_epoch_fit_s": timing["fit_seconds_per_epoch"][-1],
                "steady_epoch_val_s": timing["val_seconds_per_epoch"][-1],
            }
        costs[(doc["dataset"], doc["method"])] = {
            **entry,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "device": doc["device"],
            "support_images": n_support,
            "val_images": n_val,
        }
    return costs


def _learned_seconds(fit_pi: float, val_pi: float, support: int, val: int, epochs: int) -> float:
    return epochs * (fit_pi * support + val_pi * val) + val_pi * val  # + the reload re-check


def estimate(paths: ProjectPaths) -> dict[str, Any]:
    costs = diagnostic_costs(paths)
    assumptions = [
        "per-image costs come from the last (steady-state) epoch of each diagnostic run",
        f"learned runs take between {MIN_EPOCHS} (earliest early stop) and {MAX_EPOCHS} epochs",
        "fit cost scales linearly with support images; validation cost with validation images",
    ]
    missing = []

    def learned_cost(dataset: str, method: str) -> tuple[float, float] | None:
        own = costs.get((dataset, method))
        if own:
            return own["fit_s_per_image"], own["val_s_per_image"]
        flowers_m, flowers_p = (
            costs.get(("flowers102", method)),
            costs.get(("flowers102", "partial")),
        )
        neu_p = costs.get((dataset, "partial"))
        if flowers_m and flowers_p and neu_p:  # scale by the Flowers method/partial ratio
            ratio = flowers_m["fit_s_per_image"] / flowers_p["fit_s_per_image"]
            note = (
                f"{dataset} {method} fit cost = {dataset} partial cost x Flowers "
                f"{method}/partial ratio ({ratio:.2f})"
            )
            if note not in assumptions:
                assumptions.append(note)
            return neu_p["fit_s_per_image"] * ratio, neu_p["val_s_per_image"]
        missing.append(f"{dataset}/{method}")
        return None

    def run_seconds(dataset: str, method: str, k: int, epochs: int) -> float | None:
        support, val = k * NUM_CLASSES[dataset], VAL_IMAGES[dataset]
        if method == "prototypes":
            proto = costs.get((dataset, "prototypes")) or costs.get(("flowers102", "prototypes"))
            if not proto:
                missing.append(f"{dataset}/prototypes")
                return None
            return proto["embed_s_per_image"] * support + proto["val_s_per_image"] * val
        cost = learned_cost(dataset, method)
        return None if cost is None else _learned_seconds(*cost, support, val, epochs)

    def total(jobs: list[tuple[str, str, int]], epochs: int) -> float | None:
        values = [run_seconds(d, m, k, epochs) for d, m, k in jobs]
        return None if any(v is None for v in values) else sum(values)

    mvp = [("flowers102", m, k) for k in (1, 10) for m in (*LEARNED, "prototypes")]
    full: list[tuple[str, str, int]] = []
    for dataset in ("flowers102", "neu_cls"):
        for k in SHOTS:
            for method in (*LEARNED, "prototypes"):
                seeds = SUPPORT_SEEDS
                if dataset == "flowers102" and k == 10 and method == "prototypes":
                    seeds = SUPPORT_SEEDS[:1]  # identical supports: one unique run
                full += [(dataset, method, k)] * len(seeds)
    test_eval = None
    if costs.get(("flowers102", "partial")) and costs.get(("neu_cls", "partial")):
        test_eval = sum(
            costs[(d, "partial")]["val_s_per_image"] * TEST_IMAGES[d] for d, _, _ in full
        )

    def hours(seconds: float | None) -> float | None:
        return None if seconds is None else round(seconds / 3600, 2)

    return {
        "kind": "ESTIMATE (not an observed runtime)",
        "generated_utc": utc_now(),
        "inputs": {f"{d}/{m}": v for (d, m), v in costs.items()},
        "assumptions": assumptions,
        "missing_inputs": sorted(set(missing)),
        "mvp_8_jobs_hours": {
            "lower_16_epochs": hours(total(mvp, MIN_EPOCHS)),
            "upper_100_epochs": hours(total(mvp, MAX_EPOCHS)),
        },
        "full_matrix_unique_runs": len(full),
        "full_matrix_training_hours": {
            "lower_16_epochs": hours(total(full, MIN_EPOCHS)),
            "upper_100_epochs": hours(total(full, MAX_EPOCHS)),
        },
        "full_matrix_test_evaluation_hours": hours(test_eval),
        "excluded": [
            "data preparation",
            "report generation",
            "failure contingency",
            "corruption study (P5)",
        ],
    }


def write_estimate(paths: ProjectPaths) -> dict[str, Any]:
    result = estimate(paths)
    atomic_write_json(paths.reports / "compute_estimate.json", result)
    lines = [
        "# Compute estimate",
        "",
        f"**{result['kind']}.** Generated {result['generated_utc']} from the measured "
        "diagnostic runs listed below. Training durations are not promised.",
        "",
        "| Scope | Lower bound (16 epochs) | Upper bound (100 epochs) |",
        "|---|---:|---:|",
        f"| MVP, 8 Flowers jobs | {result['mvp_8_jobs_hours']['lower_16_epochs']} h | "
        f"{result['mvp_8_jobs_hours']['upper_100_epochs']} h |",
        f"| Full matrix, {result['full_matrix_unique_runs']} unique runs (training) | "
        f"{result['full_matrix_training_hours']['lower_16_epochs']} h | "
        f"{result['full_matrix_training_hours']['upper_100_epochs']} h |",
        "",
        f"Test evaluation of every unique run (P5): about "
        f"{result['full_matrix_test_evaluation_hours']} h.",
        "",
        "## Inputs (measured)",
        "",
    ]
    for key, value in result["inputs"].items():
        lines.append(
            f"- `{key}` from `{value['run_id']}/{value['attempt_id']}` on "
            f"{value['device']}: "
            + ", ".join(f"{k} {v:.4g}" for k, v in value.items() if isinstance(v, float))
        )
    lines += [
        "",
        "## Assumptions",
        "",
        *[f"- {a}" for a in result["assumptions"]],
        "",
        "Not included: " + ", ".join(result["excluded"]) + ".",
        "",
    ]
    if result["missing_inputs"]:
        lines.append(f"Missing measurements: {result['missing_inputs']}.")
    atomic_write_text(paths.reports / "compute_estimate.md", "\n".join(lines))
    return result
