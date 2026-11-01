"""Code provenance: git commit, dirty state, and a hash of the exact source that ran.

The source hash covers every code, config and lock file (never data, runs, reports or caches).
When the working tree is dirty, a deterministic tar.gz snapshot of exactly those files is saved
with the run, so the code that produced a result can always be recovered.
"""

from __future__ import annotations

import gzip
import io
import subprocess
import tarfile
from pathlib import Path
from typing import Any

from fewshotlab.hashing import sha256_bytes, sha256_file, sha256_json

CODE_ROOTS = ("src", "configs", "tests", "app")
CODE_FILES = ("pyproject.toml", "uv.lock", "requirements.lock", ".python-version")
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def code_files(root: Path) -> list[Path]:
    files: list[Path] = [root / name for name in CODE_FILES if (root / name).is_file()]
    for folder in CODE_ROOTS:
        base = root / folder
        if base.is_dir():
            files += [
                p
                for p in base.rglob("*")
                if p.is_file()
                and not p.is_symlink()
                and not SKIP_PARTS & set(p.parts)
                and p.suffix not in SKIP_SUFFIXES
                and not p.name.startswith((".env", ".DS_Store"))
            ]
    return sorted(set(files))


def source_digest(root: Path) -> tuple[str, int]:
    entries = [[p.relative_to(root).as_posix(), sha256_file(p)] for p in code_files(root)]
    return sha256_json(entries), len(entries)


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30
        )
    except OSError:
        return None
    return out.stdout if out.returncode == 0 else None


def git_state(root: Path) -> dict[str, Any]:
    commit = _git(root, "rev-parse", "HEAD")
    porcelain = _git(root, "status", "--porcelain")
    diff = _git(root, "diff", "HEAD")
    return {
        "commit": commit.strip() if commit else None,
        "branch": (_git(root, "rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None,
        "dirty": bool(porcelain and porcelain.strip()),
        "changed_paths": [line[3:] for line in (porcelain or "").splitlines()][:200],
        "tracked_diff_sha256": sha256_bytes(diff.encode()) if diff else None,
    }


def write_source_snapshot(root: Path, destination: Path) -> str:
    """Deterministic tar.gz of the code files (fixed mtimes and owners). Returns SHA-256."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in code_files(root):
            info = tar.gettarinfo(str(path), arcname=path.relative_to(root).as_posix())
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            with open(path, "rb") as handle:
                tar.addfile(info, handle)
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0) as gz:
        gz.write(raw.getvalue())
    destination.write_bytes(compressed.getvalue())
    return sha256_file(destination)
