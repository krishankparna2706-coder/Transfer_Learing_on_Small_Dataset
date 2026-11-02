"""P6: recorded paths keep their project spelling when run folders live behind a symlink."""

from __future__ import annotations

from fewshotlab.paths import ENV_PROJECT_ROOT, portable_path


def test_portable_path_keeps_the_project_spelling_through_a_symlink(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "fewshotlab"\n', encoding="utf-8")
    elsewhere = tmp_path / "local-disk" / "runs"
    (elsewhere / "run-1" / "a001" / "verifications").mkdir(parents=True)
    target = elsewhere / "run-1" / "a001" / "verifications" / "val-001.json"
    target.write_text("{}", encoding="utf-8")
    (root / "runs").mkdir()
    (root / "runs" / "artifacts").symlink_to(elsewhere, target_is_directory=True)
    monkeypatch.setenv(ENV_PROJECT_ROOT, str(root))

    linked = root / "runs" / "artifacts" / "run-1" / "a001" / "verifications" / "val-001.json"
    assert portable_path(linked) == "runs/artifacts/run-1/a001/verifications/val-001.json"
    assert portable_path(root / "pyproject.toml") == "pyproject.toml"
    # The real location is outside the project, and so is a path that climbs out with "..".
    assert portable_path(target) == "<outside project>/val-001.json"
    escaping = root / "runs" / ".." / ".." / "local-disk" / "runs" / "run-1" / "a001"
    assert portable_path(escaping) == "<outside project>/a001"
