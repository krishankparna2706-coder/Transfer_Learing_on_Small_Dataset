"""Device discovery and resolution: CPU first; CUDA or MPS only when actually usable."""

from __future__ import annotations

from dataclasses import dataclass

from fewshotlab.errors import DeviceUnavailableError

DEVICE_NAMES = ("auto", "cpu", "mps", "cuda")


@dataclass(frozen=True)
class DeviceAvailability:
    cuda_built: bool
    cuda_available: bool
    cuda_device_names: tuple[str, ...]
    torch_cuda_version: str | None
    mps_built: bool
    mps_available: bool
    mps_probe: str  # "PASS", "FAIL: ..." or "SKIPPED (not available)"

    def usable(self, name: str) -> bool:
        if name == "cpu":
            return True
        if name == "cuda":
            return self.cuda_available
        if name == "mps":
            return self.mps_available and self.mps_probe == "PASS"
        raise ValueError(f"unknown device {name!r}")

    @property
    def auto(self) -> str:
        if self.usable("cuda"):
            return "cuda"
        if self.usable("mps"):
            return "mps"
        return "cpu"


def probe_devices() -> DeviceAvailability:
    """Query torch for CUDA/MPS support and run a tiny MPS operation as a live probe."""
    import torch

    cuda_available = bool(torch.cuda.is_available())
    names = (
        tuple(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
        if cuda_available
        else ()
    )
    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())
    probe = "SKIPPED (not available)"
    if mps_available:
        try:
            values = torch.arange(4, dtype=torch.float32, device="mps") * 2.0
            total = float(values.sum().to("cpu"))
            probe = "PASS" if total == 12.0 else f"FAIL: expected 12.0, got {total}"
        except Exception as exc:  # reported to the user, never hidden
            probe = f"FAIL: {type(exc).__name__}: {exc}"
    return DeviceAvailability(
        cuda_built=bool(torch.backends.cuda.is_built()),
        cuda_available=cuda_available,
        cuda_device_names=names,
        torch_cuda_version=torch.version.cuda,
        mps_built=mps_built,
        mps_available=mps_available,
        mps_probe=probe,
    )


def unavailable_message(name: str, avail: DeviceAvailability) -> str:
    """An actionable explanation of why ``name`` cannot be used here."""
    if name == "cuda":
        alternative = "mps" if avail.usable("mps") else "cpu"
        return (
            "CUDA was requested, but torch.cuda.is_available() is False on this machine "
            f"(this torch build has CUDA support: {avail.cuda_built}). Use --device "
            f"{alternative} here, or run on an NVIDIA GPU host with a CUDA-enabled PyTorch "
            "build and re-check with `python -m fewshotlab doctor`."
        )
    if name == "mps":
        return (
            "MPS (Apple GPU) was requested, but it is not usable here "
            f"(built: {avail.mps_built}, available: {avail.mps_available}, "
            f"probe: {avail.mps_probe}). MPS needs Apple Silicon, macOS 12.3+ and an arm64 "
            "Python; use --device cpu instead."
        )
    return f"device {name!r} is not available"


def resolve_device(name: str, avail: DeviceAvailability | None = None) -> str:
    """Map ``auto``/``cpu``/``mps``/``cuda`` to a usable device or raise with a fix-it hint."""
    if name not in DEVICE_NAMES:
        raise ValueError(f"unknown device {name!r}; choose from {', '.join(DEVICE_NAMES)}")
    avail = avail or probe_devices()
    if name == "auto":
        return avail.auto
    if not avail.usable(name):
        raise DeviceUnavailableError(unavailable_message(name, avail))
    return name
