"""Error galleries with a selection rule fixed in advance (never chosen by appearance).

The galleries contain dataset images, so they are written to a gitignored folder and never
committed or published.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fewshotlab.config import load_run_config
from fewshotlab.data.imageio import open_image, to_rgb
from fewshotlab.data.manifest import load_class_map, read_manifest
from fewshotlab.data.registry import class_map_path, manifest_path
from fewshotlab.engine import artifacts
from fewshotlab.io_utils import atomic_write_json
from fewshotlab.paths import ProjectPaths

RULE = "the first 12 misclassified validation images in ascending image_id order"
LIMIT = 12


def select_errors(image_ids, true, pred, limit: int = LIMIT) -> list[int]:
    order = sorted(range(len(image_ids)), key=lambda i: image_ids[i])
    return [i for i in order if int(true[i]) != int(pred[i])][:limit]


def _names(paths: ProjectPaths, dataset: str) -> list[str]:
    cmap = load_class_map(class_map_path(paths, dataset))
    return [str(cmap.extra.get(n, {}).get("display_name", n)) for n in cmap.names]


def error_galleries(paths: ProjectPaths, rows: list[dict[str, Any]], out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = []
    for row in rows:
        if not row.get("_attempt_dir"):
            continue
        cfg = load_run_config(row["_attempt_dir"] / "config.resolved.yaml")
        stored = artifacts.read_predictions(row["_attempt_dir"] / "predictions_val.csv")
        chosen = select_errors(stored["image_ids"], stored["true"], stored["pred"])
        n_errors = int((stored["true"] != stored["pred"]).sum())
        results.append(
            {
                "run_id": row["run_id"],
                "n_images": len(stored["image_ids"]),
                "n_errors": n_errors,
                "shown": [stored["image_ids"][i] for i in chosen],
            }
        )
        if not chosen:
            continue
        names = _names(paths, cfg.dataset)
        records = {
            r.image_id: r for r in read_manifest(manifest_path(paths, cfg.dataset), splits=("val",))
        }
        fig, axes = plt.subplots(3, 4, figsize=(10, 8.4), constrained_layout=True)
        for axis in axes.ravel():
            axis.axis("off")
        for axis, i in zip(axes.ravel(), chosen, strict=False):
            image_id = stored["image_ids"][i]
            image = to_rgb(open_image(paths.data_root / records[image_id].relative_path))
            image.thumbnail((200, 200))
            axis.imshow(image)
            score = float(stored["scores"][i][int(stored["pred"][i])])
            axis.set_title(
                f"{image_id.split('/')[-1]}\ntrue: {names[int(stored['true'][i])]}\n"
                f"pred: {names[int(stored['pred'][i])]} ({score:.2f})",
                fontsize=7,
            )
        fig.suptitle(
            f"{row['run_id']}: {RULE} (validation; model scores are uncalibrated)", fontsize=8
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{row['run_id']}.png", dpi=90)
        plt.close(fig)
    summary = {
        "rule": RULE,
        "runs": results,
        "note": "gallery images contain dataset photos: local only, never committed",
    }
    atomic_write_json(Path(out_dir) / "errors.json", summary)
    return summary
