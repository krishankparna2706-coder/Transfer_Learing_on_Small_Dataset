"""Plots drawn only from stored run files (never from typed-in numbers)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def learning_curves(history: list[dict[str, Any]], path: Path, *, title: str) -> Path:
    epochs = [row["epoch"] for row in history]
    best = [row["epoch"] for row in history if row["is_best"]]
    selected = best[-1] if best else None
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    left.plot(epochs, [r["train_loss"] for r in history], label="support (train) loss")
    left.plot(epochs, [r["val_loss"] for r in history], label="validation loss")
    left.set_xlabel("epoch")
    left.set_ylabel("cross-entropy")
    left.legend()
    right.plot(epochs, [r["val_macro_f1"] for r in history], color="tab:green")
    right.set_xlabel("epoch")
    right.set_ylabel("validation macro-F1")
    right.set_ylim(0, 1)
    if selected is not None:
        for axis in (left, right):
            axis.axvline(selected, color="gray", linestyle="--", linewidth=1)
        right.annotate(f"selected epoch {selected}", (selected, 0.02), fontsize=8)
    fig.suptitle(title, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def confusion_plot(row_normalized: list[list[float] | None], path: Path, *, title: str) -> Path:
    import numpy as np

    matrix = np.array(
        [row if row is not None else [np.nan] * len(row_normalized) for row in row_normalized],
        dtype=float,
    )
    size = 3.5 if len(matrix) <= 10 else 8
    fig, axis = plt.subplots(figsize=(size, size), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    axis.set_xlabel("predicted class")
    axis.set_ylabel("true class")
    axis.set_title(title, fontsize=9)
    fig.colorbar(image, ax=axis, fraction=0.046, label="row-normalized fraction")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
