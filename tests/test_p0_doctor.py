"""P0: doctor reports the actual backend, stays read-only, and fails actionably."""

from __future__ import annotations

import json

import pytest
import torch

from fewshotlab.cli import main
from fewshotlab.devices import probe_devices
from fewshotlab.doctor import collect_report
from helpers import snapshot_files


def test_report_matches_torch_backends(isolated_paths):
    report = collect_report()
    devices = report["devices"]
    assert devices["cuda"]["available"] == torch.cuda.is_available()
    assert devices["cuda"]["built"] == torch.backends.cuda.is_built()
    assert devices["mps"]["available"] == torch.backends.mps.is_available()
    assert devices["mps"]["built"] == torch.backends.mps.is_built()
    assert devices["auto_selected"] in ("cpu", "mps", "cuda")
    assert report["app"]["protocol_id"] == "fewshotlab-v1"
    assert report["ledger"]["benchmark_status"] == "NOT_RUN"
    names = {check["name"] for check in report["checks"]}
    assert {"python_version", "required_libraries", "runs_writable", "reports_writable"} <= names


def test_doctor_json_cli(isolated_paths, capsys):
    assert main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["datasets"]["flowers102"]["status"] == "NOT_PREPARED"
    assert report["datasets"]["neu_cls"]["status"] == "NOT_PREPARED"
    assert report["caches"]["weights"]["resnet18"]["cached"] is False
    assert report["artifact_dirs"]["runs"]["writable"] is True


def _unusable_accelerator() -> str | None:
    avail = probe_devices()
    for name in ("cuda", "mps"):
        if not avail.usable(name):
            return name
    return None


def test_requiring_an_unusable_device_is_actionable(isolated_paths, capsys):
    name = _unusable_accelerator()
    if name is None:
        pytest.skip("every accelerator is usable on this machine")
    assert main(["doctor", "--require-device", name]) == 4
    err = capsys.readouterr().err
    assert "required device unavailable" in err
    assert "--device" in err  # says what to use instead


def test_requiring_cpu_always_passes(isolated_paths):
    assert main(["doctor", "--require-device", "cpu"]) == 0


def test_doctor_is_read_only(isolated_paths):
    before = snapshot_files(isolated_paths.root)
    main(["doctor"])
    assert snapshot_files(isolated_paths.root) == before
    assert not isolated_paths.cache.exists()
    assert not isolated_paths.data_root.exists()
