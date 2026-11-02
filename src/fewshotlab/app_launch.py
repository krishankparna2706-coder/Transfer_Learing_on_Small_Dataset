"""Launch the Streamlit demonstration, or check offline that it can start.

- ``python -m fewshotlab demo`` runs ``streamlit run app/streamlit_app.py`` for this machine only
  (127.0.0.1 by default), with usage statistics off and a 10 MB upload limit.
- ``demo --check`` loads everything the app needs, offline and on the CPU, and reports what is
  missing instead of starting a server. It writes nothing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from fewshotlab.errors import FewShotLabError
from fewshotlab.paths import ENV_PROJECT_ROOT, ProjectPaths, portable_path

APP_FILE = Path("app") / "streamlit_app.py"
ENV_APP_STUDY = "FEWSHOTLAB_APP_STUDY"
DEFAULT_STUDY = Path("configs") / "full-study.yaml"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8501


def app_file(paths: ProjectPaths) -> Path:
    return paths.root / APP_FILE


def study_file(paths: ProjectPaths, value: str | os.PathLike[str] | None = None) -> Path:
    """The study the app shows: the argument, else ``FEWSHOTLAB_APP_STUDY``, else the core study."""
    chosen = Path(value or os.environ.get(ENV_APP_STUDY) or DEFAULT_STUDY).expanduser()
    return chosen if chosen.is_absolute() else paths.root / chosen


def streamlit_command(paths: ProjectPaths, *, host: str, port: int) -> list[str]:
    from fewshotlab.engine.predict import MAX_UPLOAD_BYTES

    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_file(paths)),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        "none",
        "--server.maxUploadSize",
        str(MAX_UPLOAD_BYTES // 2**20),
        "--browser.gatherUsageStats",
        "false",
    ]


def launch(paths: ProjectPaths, *, study: Path, host: str, port: int) -> int:
    """Run the app in the foreground until Ctrl-C; returns Streamlit's exit code."""
    env = {**os.environ, ENV_APP_STUDY: str(study), ENV_PROJECT_ROOT: str(paths.root)}
    print(
        f"FewShotLab demo on http://{host}:{port} (study {portable_path(study)}); Ctrl-C stops it"
    )
    try:
        return subprocess.call(
            streamlit_command(paths, host=host, port=port), cwd=paths.root, env=env
        )
    except KeyboardInterrupt:
        return 0


def _prototype_weights_needed(paths: ProjectPaths, registry: dict[str, Any]) -> bool:
    """True when a stored prototype model rebuilds its encoder from cached pretrained weights.

    Learned models keep every weight in their own checkpoint, so only prototypes need the cache.
    """
    from fewshotlab.engine import artifacts
    from fewshotlab.io_utils import read_json

    runs = {
        r["source_run_id"]
        for r in registry["rows"]["val"]
        if r["status"] == "COMPLETED" and r["method"] == "prototypes"
    }
    for run_id in runs:
        for attempt in reversed(artifacts.attempt_dirs(paths, run_id)):
            spec = attempt / "model_spec.json"
            if spec.is_file():
                if read_json(spec).get("encoder", {}).get("weights_id") not in (None, "none"):
                    return True
                break
    return False


def startup_check(paths: ProjectPaths, study: Path) -> dict[str, Any]:
    """Load what the app needs, offline and on the CPU; each check is PASS, WARN or FAIL."""
    from fewshotlab.data.manifest import read_manifest
    from fewshotlab.data.registry import manifest_path
    from fewshotlab.engine import predict
    from fewshotlab.engine.study import load_study
    from fewshotlab.models.backbone import ensure_imagenet_weights
    from fewshotlab.reporting import run_registry
    from fewshotlab.simulation.sealed_replay import ReplayUnavailable, load_sealed_replay

    checks: list[dict[str, str]] = []

    def record(name: str, status: str, detail: str) -> None:
        checks.append({"check": name, "status": status, "detail": detail})

    try:
        import streamlit

        record("streamlit", "PASS", f"version {streamlit.__version__}")
    except ImportError:
        record("streamlit", "FAIL", "not installed; run `uv sync --locked`")
    target = app_file(paths)
    record("app file", "PASS" if target.is_file() else "FAIL", portable_path(target))
    try:
        loaded = load_study(study)
        registry = run_registry.load_registry(loaded, paths)
    except (FewShotLabError, OSError) as exc:
        record("study registry", "FAIL", f"{portable_path(study)}: {exc}")
        return {"ok": False, "study": portable_path(study), "checks": checks}
    weights = "ImageNet weights in the local cache"
    if not _prototype_weights_needed(paths, registry):
        record(weights, "PASS", "not needed: no stored prototype model uses pretrained weights")
    else:
        try:
            _, info = ensure_imagenet_weights(paths.weights_dir, allow_download=False)
            record(weights, "PASS", f"{info.weights_id} {info.sha256[:12]}")
        except FewShotLabError as exc:
            record(weights, "FAIL", f"prototype models need them: {exc}")
    for split in ("val", "test"):
        counts = Counter(row["status"] for row in registry["rows"][split])
        measured = any(row["macro_f1"] is not None for row in registry["rows"][split])
        record(f"{split} numbers", "PASS" if measured else "WARN", dict(sorted(counts.items())))
    for dataset in loaded["datasets"]:
        chosen = run_registry.default_cell(registry, dataset)
        name = f"{dataset} model on cpu"
        if chosen is None:
            record(name, "WARN", "no COMPLETED run yet; the prediction page shows NOT RUN")
            continue
        try:
            ref = run_registry.cell_model(paths, registry, *chosen)
            model = predict.load_model(paths, ref, "cpu")
            image = read_manifest(manifest_path(paths, dataset), splits=("val",))[0]
            live = predict.predict_records(model, [image], paths.data_root)[0]
            stored = predict.stored_validation_prediction(paths, ref, image.image_id)
            same = predict.compare_scores(live, stored)
            ok = same["same_class"] and same["within_tolerance"]
            record(
                name,
                "PASS" if ok else "FAIL",
                f"{ref.label}: live prediction on {image.image_id} vs the stored validation "
                f"prediction: same class {same['same_class']}, max |score difference| "
                f"{same['max_abs_score_difference']:.1e} (tolerance {same['tolerance']:g})",
            )
        except FewShotLabError as exc:
            record(name, "FAIL", str(exc))
    try:
        replay = load_sealed_replay(loaded, paths)
        first = replay.items(replay.streams[0].stream_id)[0]
        replay.pixels(first)
        record(
            "sealed replay",
            "PASS",
            f"{len(replay.streams)} streams x {len(replay.items(first.stream_id))} images, "
            f"seal {replay.identity['audit_token']}",
        )
    except ReplayUnavailable as exc:
        unsealed = registry["seal"] is None
        record("sealed replay", "WARN" if unsealed else "FAIL", str(exc).splitlines()[0])
    return {
        "ok": all(c["status"] != "FAIL" for c in checks),
        "study": portable_path(study),
        "checks": checks,
    }
