"""What ran: software, hardware, device and determinism settings (``environment.json``)."""

from __future__ import annotations

import importlib.metadata as metadata
import os
import platform
import subprocess
import sys
from typing import Any

import torch

LIBRARIES = ("torch", "torchvision", "numpy", "scipy", "pillow", "pyyaml", "pandas", "matplotlib")


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _cpu_brand() -> str | None:
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return out.stdout.strip() or None
        except OSError:
            return None
    return platform.processor() or None


def configure_determinism(device: str) -> dict[str, Any]:
    """Request deterministic kernels: strict on CPU, warn-only on accelerators (recorded)."""
    strict = device == "cpu"
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "note": "strict on CPU; warn-only on MPS/CUDA because some kernels have no deterministic "
        "implementation. Bitwise reproducibility is not claimed across devices or platforms.",
    }


def environment_report(device: str, determinism: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0] or None,
        },
        "cpu": {
            "brand": _cpu_brand(),
            "logical_cores": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
        },
        "libraries": {name: _version(name) for name in LIBRARIES},
        "device": device,
        "torch_build": {"cuda": torch.version.cuda, "mps_built": torch.backends.mps.is_built()},
        "precision": "float32",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "determinism": determinism,
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
    }
    if device == "cuda" and torch.cuda.is_available():
        report["cuda"] = {
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    return report
