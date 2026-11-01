"""Read-only environment report: ``python -m fewshotlab doctor``.

Doctor never downloads, trains or writes results. Its only filesystem side effect is a
temporary probe file in each existing artifact directory, deleted immediately.
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

from fewshotlab import PROTOCOL_ID, __version__
from fewshotlab.hashing import sha256_file
from fewshotlab.paths import ProjectPaths, get_paths
from fewshotlab.status import DataStatus, ExitCode, RunStatus

REQUIRED_DISTRIBUTIONS = (
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "pillow",
    "pyyaml",
    "pandas",
    "matplotlib",
)
APP_DISTRIBUTIONS = ("streamlit",)
DEV_DISTRIBUTIONS = ("pytest", "ruff", "scikit-learn")
SUPPORTED_PYTHON = (3, 11)
DATASET_NAMES = ("flowers102", "neu_cls")


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _nearest_existing(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if candidate.exists():
            return candidate
    return Path("/")


def _probe_dir(path: Path) -> dict[str, Any]:
    if not path.exists():
        parent = _nearest_existing(path)
        return {
            "path": str(path),
            "exists": False,
            "writable": os.access(parent, os.W_OK),
            "note": "created on first use",
        }
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix=".doctor-probe-", delete=True) as handle:
            handle.write(b"probe")
            handle.flush()
        return {"path": str(path), "exists": True, "writable": True}
    except OSError as exc:
        return {"path": str(path), "exists": True, "writable": False, "error": str(exc)}


def _devices() -> dict[str, Any]:
    try:
        import torch

        from fewshotlab.devices import probe_devices
    except Exception as exc:  # torch missing or broken: report, do not crash
        return {"error": f"{type(exc).__name__}: {exc}"}
    avail = probe_devices()
    return {
        "cpu": {
            "available": True,
            "logical_cores": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
        },
        "cuda": {
            "built": avail.cuda_built,
            "available": avail.cuda_available,
            "torch_cuda_version": avail.torch_cuda_version,
            "devices": list(avail.cuda_device_names),
        },
        "mps": {
            "built": avail.mps_built,
            "available": avail.mps_available,
            "probe": avail.mps_probe,
        },
        "auto_selected": avail.auto,
    }


def _weights(paths: ProjectPaths) -> dict[str, Any]:
    try:
        from fewshotlab.models.backbone import WEIGHTS_FILENAME, WEIGHTS_ID, WEIGHTS_URL
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    path = paths.weights_dir / WEIGHTS_FILENAME
    return {
        "resnet18": {
            "weights_id": WEIGHTS_ID,
            "url": WEIGHTS_URL,
            "path": str(path),
            "cached": path.is_file(),
        }
    }


def _dataset_status(paths: ProjectPaths, name: str) -> dict[str, Any]:
    raw_dir = paths.data_root / name
    record_path = paths.manifests / f"{name}.prepare.json"
    status = DataStatus.NOT_PREPARED.value
    detail = None
    if record_path.is_file():
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            status = record.get("status", status)
            detail = record.get("summary")
        except (OSError, json.JSONDecodeError) as exc:
            status, detail = DataStatus.FAILED.value, f"unreadable {record_path.name}: {exc}"
    return {
        "status": status,
        "detail": detail,
        "raw_dir": str(raw_dir),
        "raw_present": raw_dir.is_dir() and any(raw_dir.iterdir()),
        "prepare_record": str(record_path) if record_path.is_file() else None,
    }


def _ledger_summary(path: Path) -> dict[str, Any]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    unreadable = 0
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                unreadable += 1
                continue
            latest[(str(record.get("run_id")), str(record.get("attempt_id")))] = record
    completed_core = sum(
        1
        for r in latest.values()
        if r.get("status") == RunStatus.COMPLETED.value
        and r.get("category") == "core"
        and not r.get("is_fixture", False)
    )
    return {
        "path": str(path),
        "exists": path.is_file(),
        "attempts": len(latest),
        "unreadable_lines": unreadable,
        "completed_core_runs": completed_core,
        "benchmark_status": RunStatus.NOT_RUN.value
        if completed_core == 0
        else f"PARTIAL ({completed_core} completed core runs; see the benchmark report)",
    }


def _locks(paths: ProjectPaths) -> dict[str, Any]:
    out = {}
    for name in ("uv.lock", "requirements.lock"):
        path = paths.root / name
        out[name] = {
            "present": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    return out


def collect_report() -> dict[str, Any]:
    """Gather every doctor fact into one JSON-serializable mapping."""
    paths = get_paths()
    libraries = {name: _dist_version(name) for name in REQUIRED_DISTRIBUTIONS}
    app = {name: _dist_version(name) for name in APP_DISTRIBUTIONS}
    dev = {name: _dist_version(name) for name in DEV_DISTRIBUTIONS}
    report: dict[str, Any] = {
        "app": {"name": "fewshotlab", "version": __version__, "protocol_id": PROTOCOL_ID},
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "macos_version": platform.mac_ver()[0] or None,
        },
        "libraries": libraries,
        "optional": {"app": app, "dev": dev},
        "devices": _devices(),
        "artifact_dirs": {
            "runs": _probe_dir(paths.runs),
            "reports": _probe_dir(paths.reports),
            "manifests": _probe_dir(paths.manifests),
            "cache": _probe_dir(paths.cache),
            "data_root": _probe_dir(paths.data_root),
        },
        "caches": {"weights_dir": str(paths.weights_dir), "weights": _weights(paths)},
        "datasets": {name: _dataset_status(paths, name) for name in DATASET_NAMES},
        "locks": _locks(paths),
        "ledger": _ledger_summary(paths.ledger),
        "project_root": str(paths.root),
    }
    report["checks"] = _checks(report)
    return report


def _checks(report: dict[str, Any]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    major_minor = tuple(int(p) for p in report["python"]["version"].split(".")[:2])
    add(
        "python_version",
        "PASS" if major_minor == SUPPORTED_PYTHON else "WARN",
        f"Python {report['python']['version']} (protocol environment pins 3.11)",
    )
    missing = [n for n, v in report["libraries"].items() if v is None]
    add(
        "required_libraries",
        "FAIL" if missing else "PASS",
        f"missing: {missing}" if missing else "all required libraries import metadata found",
    )
    missing_app = [n for n, v in report["optional"]["app"].items() if v is None]
    add(
        "app_libraries",
        "WARN" if missing_app else "PASS",
        f"missing (Streamlit app unavailable): {missing_app}"
        if missing_app
        else "streamlit present",
    )
    devices = report["devices"]
    if "error" in devices:
        add("devices", "FAIL", devices["error"])
    else:
        add("devices", "PASS", f"auto-selected device: {devices['auto_selected']}")
    for key in ("runs", "reports"):
        probe = report["artifact_dirs"][key]
        add(
            f"{key}_writable",
            "PASS" if probe["writable"] else "FAIL",
            probe.get("error") or probe["path"],
        )
    weights = report["caches"]["weights"].get("resnet18", {})
    add(
        "resnet18_weights_cached",
        "PASS" if weights.get("cached") else "WARN",
        "cached"
        if weights.get("cached")
        else "not cached: pretrained methods need --allow-download once",
    )
    for name, info in report["datasets"].items():
        add(f"dataset_{name}", "PASS" if info["status"] == "PREPARED" else "WARN", info["status"])
    lock = report["locks"]["uv.lock"]
    add("uv_lock", "PASS" if lock["present"] else "WARN", lock["sha256"] or "uv.lock not found")
    add("benchmark", "INFO", report["ledger"]["benchmark_status"])
    return checks


def _render_text(report: dict[str, Any]) -> str:
    lines = [
        f"FewShotLab doctor (fewshotlab {report['app']['version']}, protocol "
        f"{report['app']['protocol_id']})",
        "Read-only report: no downloads, no training, no result files.",
        "",
        f"Python     {report['python']['version']} ({report['python']['implementation']}) at "
        f"{report['python']['executable']}",
        f"Platform   {report['platform']['system']} {report['platform']['release']} "
        f"{report['platform']['machine']}"
        + (
            f" (macOS {report['platform']['macos_version']})"
            if report["platform"]["macos_version"]
            else ""
        ),
        "Libraries  " + " | ".join(f"{n} {v or 'MISSING'}" for n, v in report["libraries"].items()),
        "Optional   "
        + " | ".join(
            f"{n} {v or 'missing'}"
            for group in report["optional"].values()
            for n, v in group.items()
        ),
    ]
    devices = report["devices"]
    if "error" in devices:
        lines.append(f"Devices    ERROR {devices['error']}")
    else:
        cuda, mps = devices["cuda"], devices["mps"]
        lines.append(
            f"Devices    cpu: available ({devices['cpu']['logical_cores']} logical cores, "
            f"torch threads {devices['cpu']['torch_threads']}) | cuda: "
            f"{'available ' + str(cuda['devices']) if cuda['available'] else 'not available'} "
            f"(built: {cuda['built']}) | mps: "
            f"{'available' if mps['available'] else 'not available'} (built: {mps['built']}, "
            f"probe: {mps['probe']})"
        )
        lines.append(f"           auto -> {devices['auto_selected']}")
    for key, probe in report["artifact_dirs"].items():
        state = "writable" if probe["writable"] else "NOT WRITABLE"
        if not probe["exists"]:
            state += f" ({probe.get('note', 'missing')})"
        lines.append(f"{'Artifacts' if key == 'runs' else '':<10} {key}: {state}  {probe['path']}")
    weights = report["caches"]["weights"].get("resnet18")
    if weights:
        lines.append(
            f"Weights    ResNet18 {weights['weights_id']}: "
            f"{'cached' if weights['cached'] else 'NOT CACHED'} at {weights['path']}"
        )
    for name, info in report["datasets"].items():
        lines.append(
            f"{'Datasets' if name == DATASET_NAMES[0] else '':<10} {name}: {info['status']}"
        )
    for name, lock in report["locks"].items():
        lines.append(
            f"{'Locks' if name == 'uv.lock' else '':<10} {name}: "
            + (f"sha256 {lock['sha256']}" if lock["present"] else "not found")
        )
    ledger = report["ledger"]
    lines.append(
        f"Ledger     {ledger['attempts']} recorded attempt(s); benchmark status "
        f"{ledger['benchmark_status']}"
    )
    lines.append("")
    lines.append("Checks")
    for check in report["checks"]:
        lines.append(f"  {check['status']:<5} {check['name']}: {check['detail']}")
    return "\n".join(lines)


def run_doctor(*, as_json: bool = False, require_device: str | None = None) -> int:
    report = collect_report()
    exit_code = (
        ExitCode.FAILURE if any(c["status"] == "FAIL" for c in report["checks"]) else ExitCode.OK
    )
    device_message = None
    if require_device is not None:
        from fewshotlab.devices import probe_devices, unavailable_message

        avail = probe_devices()
        usable = avail.usable(require_device)
        device_message = None if usable else unavailable_message(require_device, avail)
        report["required_device"] = {
            "name": require_device,
            "usable": usable,
            "message": device_message,
        }
        if not usable:
            exit_code = ExitCode.BLOCKED
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_render_text(report))
    if device_message:
        print(
            f"\nfewshotlab doctor: required device unavailable: {device_message}", file=sys.stderr
        )
    return exit_code
