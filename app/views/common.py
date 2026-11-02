"""Shared loaders, labels and guards for the app pages.

Only the fewshotlab library computes anything. This module caches its results and formats them.
It never trains, never writes a file, and reads held-out test images only through the verified
sealed replay.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch

from fewshotlab import PROTOCOL_ID, __version__
from fewshotlab.app_launch import study_file
from fewshotlab.data.imageio import open_image, to_rgb
from fewshotlab.engine import predict
from fewshotlab.engine.study import load_study
from fewshotlab.errors import FewShotLabError
from fewshotlab.paths import ProjectPaths, get_paths
from fewshotlab.reporting import run_registry
from fewshotlab.simulation.replay_live import LiveReplay
from fewshotlab.simulation.sealed_replay import SealedReplay, load_sealed_replay

METHOD_ORDER = ("scratch", "frozen", "partial", "prototypes")
METHOD_TEXT = {
    "scratch": "a randomly initialized ResNet18; every layer is trained on the K images.",
    "frozen": "the ImageNet ResNet18 stays fixed; only a new linear head is trained.",
    "partial": "the ImageNet ResNet18's last block (layer4) and a new head are trained.",
    "prototypes": "ImageNet features; each class is the mean of its K embeddings. No training.",
}
DATASET_NAMES = {"flowers102": "Oxford Flowers102", "neu_cls": "NEU-CLS steel surface defects"}
METRIC_NAMES = {
    "macro_f1": "macro-F1",
    "top1": "top-1 accuracy",
    "balanced_accuracy": "balanced accuracy",
}
SPLIT_NAMES = {"val": "Validation (development)", "test": "Held-out test (sealed)"}

# Categorical slots 1-4 of the validated reference palette, in fixed order, with the steps for
# each surface. Colour follows the method, never its rank; marker shapes are a second channel.
SERIES = {
    "light": {
        "scratch": "#2a78d6",
        "frozen": "#eb6834",
        "partial": "#1baf7a",
        "prototypes": "#eda100",
    },
    "dark": {
        "scratch": "#3987e5",
        "frozen": "#d95926",
        "partial": "#199e70",
        "prototypes": "#c98500",
    },
}
INK = {
    "light": {
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "surface": "#fcfcfb",
    },
    "dark": {
        "primary": "#f0efec",
        "secondary": "#c3c2b7",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "surface": "#1a1a19",
    },
}
SHAPES = {
    "scratch": "circle",
    "frozen": "square",
    "partial": "triangle-up",
    "prototypes": "diamond",
}

BADGES = {
    "val": (
        "VALIDATION: development result",
        "Each checkpoint was selected on these same validation images, so the numbers are "
        "optimistic. They are not held-out test results.",
    ),
    "test": (
        "HELD-OUT TEST: sealed plan",
        "Every model, threshold and rule was fixed on validation before test access, and each "
        "run was evaluated on the test split once.",
    ),
    "live": (
        "LIVE PREDICTION: computed now",
        "A model score for this one image. It is not an accuracy estimate and not a benchmark "
        "result.",
    ),
    "replay": (
        "NEU HELD-OUT REPLAY: simulated inspection",
        "Stored held-out test images, played back with the sealed demo model and threshold. Not "
        "a deployed system, a camera, or a good-versus-defective decision.",
    ),
    "fixture": (
        "SOFTWARE FIXTURE: not a benchmark",
        "Synthetic images for layout development and tests only. No number here measures a "
        "real model.",
    ),
}


def theme() -> str:
    """The theme the app is drawn in: the configured base first (.streamlit/config.toml sets
    light), else what the browser session reports."""
    configured = st.get_option("theme.base")
    if configured in ("light", "dark"):
        return configured
    try:
        kind = st.context.theme.type
    except Exception:  # no browser session (tests) or an older Streamlit
        kind = None
    return "dark" if kind == "dark" else "light"


def badge(kind: str) -> None:
    title, text = BADGES[kind]
    if kind == "val":
        st.warning(text, title=title, icon=":material/science:")
    elif kind == "test":
        st.success(text, title=title, icon=":material/verified:")
    elif kind == "fixture":
        st.error(text, title=title, icon=":material/warning:")
    elif kind == "live":
        st.info(text, title=title, icon=":material/bolt:")
    else:
        st.info(text, title=title, icon=":material/precision_manufacturing:")


def fmt(value: float | None, digits: int = 3) -> str:
    return "NOT RUN" if value is None else f"{value:.{digits}f}"


def dataset_name(dataset: str) -> str:
    return DATASET_NAMES.get(dataset, dataset)


def class_label(facts: dict[str, Any], class_id: int) -> str:
    """'pink primrose (label_001)' when a display name exists, else the class name itself."""
    name, shown = facts["class_names"][class_id], facts["display_names"][class_id]
    return name if shown == name else f"{shown} ({name})"


def paths() -> ProjectPaths:
    return get_paths()


def study_path() -> Path:
    return study_file(get_paths())


def stop_with(message: str, action: str | None = None) -> None:
    st.error(message, icon=":material/error:")
    if action:
        st.info(action, icon=":material/arrow_forward:")
    st.stop()


@st.cache_data(show_spinner=False, max_entries=8)
def _load_study(path: str) -> dict[str, Any]:
    return load_study(path)


@st.cache_data(
    show_spinner="Loading the run registry: every metric is regenerated from stored predictions...",
    max_entries=4,
)
def _registry(path: str, fingerprint: str) -> dict[str, Any]:
    return run_registry.load_registry(_load_study(path), get_paths())


def registry() -> dict[str, Any]:
    """The verified registry for the study, refreshed whenever a ledger or the seal changes."""
    path = study_path()
    try:
        loaded = _load_study(str(path))
        return _registry(str(path), run_registry.fingerprint(loaded, get_paths()))
    except (FewShotLabError, OSError) as exc:
        stop_with(
            f"The study registry could not be loaded: {exc}",
            "Check the study config (set it with `python -m fewshotlab demo --study PATH`) and "
            "run `python -m fewshotlab demo --check` for a full list of missing prerequisites.",
        )
        raise  # unreachable: st.stop() ends the run


@st.cache_data(show_spinner=False, max_entries=64, ttl=600)
def _cell_ref(
    path: str, fingerprint: str, dataset: str, method: str, k: int, seed: int
) -> predict.ModelRef:
    return run_registry.cell_model(
        get_paths(), _registry(path, fingerprint), dataset, method, k, seed
    )


def cell_ref(reg: dict[str, Any], dataset: str, method: str, k: int, seed: int) -> predict.ModelRef:
    """The verified model behind one cell (raises ModelUnavailable with the next action)."""
    return _cell_ref(str(study_path()), reg["fingerprint"], dataset, method, k, seed)


@st.cache_resource(max_entries=4, show_spinner="Loading the model...")
def _model(
    run_id: str, attempt_id: str, artifact_sha256: str, device: str, _ref: predict.ModelRef
) -> predict.LoadedModel:
    return predict.load_model(get_paths(), _ref, device)


def model(ref: predict.ModelRef, device: str) -> predict.LoadedModel:
    """Loaded models are cached by run, attempt, checkpoint hash and device."""
    return _model(ref.run_id, ref.attempt_id, ref.artifact_sha256, device, ref)


@st.cache_resource(ttl=600, show_spinner="Verifying the sealed replay against the sealed plan...")
def _replay(path: str, fingerprint: str) -> SealedReplay:
    return load_sealed_replay(_load_study(path), get_paths())


@st.cache_resource(ttl=600, show_spinner="Loading the sealed demo model...")
def _live_replay(path: str, fingerprint: str, device: str) -> LiveReplay:
    return LiveReplay.load(_replay(path, fingerprint), get_paths(), device)


def live_replay(reg: dict[str, Any], device: str) -> LiveReplay:
    """The sealed demo checkpoint, identified by its sealed hash, cached per device."""
    return _live_replay(str(study_path()), reg["fingerprint"], device)


def replay(reg: dict[str, Any]) -> SealedReplay:
    return _replay(str(study_path()), reg["fingerprint"])


@st.cache_data(show_spinner=False, max_entries=512)
def replay_pixels(path: str, fingerprint: str, stream_id: str, index: int) -> np.ndarray:
    view = _replay(path, fingerprint)
    return view.pixels(view.items(stream_id)[index])


@st.cache_data(show_spinner=False, max_entries=1024)
def thumbnail(path: str, size: int = 160) -> np.ndarray:
    """A training or validation image (absolute path), decoded safely and shrunk for display."""
    image = to_rgb(open_image(path))
    image.thumbnail((size, size))
    return np.asarray(image)


def device_options() -> list[str]:
    options = ["cpu"]
    if torch.backends.mps.is_available():
        options.append("mps")
    if torch.cuda.is_available():
        options.append("cuda")
    return options


def device() -> str:
    chosen = st.session_state.get("device", "cpu")
    return chosen if chosen in device_options() else "cpu"


def budget_caption(reg: dict[str, Any], dataset: str) -> str:
    """Class count and both label budgets, shown beside every comparison."""
    budget = reg["budgets"][dataset]
    per_k = ", ".join(
        f"K={k}: {n:,}"
        for k, n in sorted(budget["support_labels"].items(), key=lambda kv: int(kv[0]))
    )
    return (
        f"{dataset_name(dataset)}: {budget['classes']} classes. Support labels per run "
        f"({per_k}), plus {budget['validation_labels']:,} validation labels used to choose each "
        "checkpoint."
    )


def status_counts(reg: dict[str, Any]) -> dict[str, int]:
    """Planned slots, and how many slots and distinct runs have validation and test values."""
    val, test = reg["rows"]["val"], reg["rows"]["test"]
    completed = [r for r in val if r["status"] == "COMPLETED"]
    evaluated = [r for r in test if r["status"] == "EVALUATED"]
    return {
        "slots": len(val),
        "runs": len({r["source_run_id"] for r in val}),
        "completed": len(completed),
        "completed_runs": len({r["source_run_id"] for r in completed}),
        "evaluated": len(evaluated),
        "evaluated_runs": len({r["source_run_id"] for r in evaluated}),
    }


def sidebar(reg: dict[str, Any]) -> None:
    study = reg["study"]
    counts = status_counts(reg)
    seal = reg["seal"]
    with st.sidebar:
        st.markdown(f"**Study** `{study['study_id']}`  \n{study.get('description', '')}")
        test_line = (
            f"{counts['evaluated_runs']} runs EVALUATED once" if counts["evaluated"] else "NOT RUN"
        )
        st.markdown(
            f"Validation: {counts['completed_runs']} of {counts['runs']} runs COMPLETED "
            f"({counts['slots']} planned slots)  \n"
            f"Held-out test: {test_line}  \n"
            f"Seal: {'`' + seal['audit_token'] + '`' if seal else 'NOT SEALED'}"
        )
        st.selectbox(
            "Device for live predictions",
            device_options(),
            key="device",
            help="CPU always works and is the default. It changes only live predictions; every "
            "stored benchmark number is fixed.",
        )
        st.caption(
            f"fewshotlab {__version__}, protocol {PROTOCOL_ID}. Nothing in this app trains a "
            "model or changes a dataset. Uploaded images stay in memory for your session only; "
            "they are never saved or logged."
        )
