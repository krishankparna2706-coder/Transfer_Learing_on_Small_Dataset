"""Safe handling of untrusted downloaded archives.

Archives are type-sniffed (HTML pages are rejected), every member path is checked before
extraction (no absolute paths, no ``..``, no links), extraction goes into a new, empty
directory, and the result is re-checked for links or entries outside that directory.
"""

from __future__ import annotations

import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from fewshotlab.errors import DataValidationError

MAX_MEMBERS = 20_000
MAX_TOTAL_BYTES = 4 * 1024**3
_MAGIC: tuple[tuple[str, tuple[bytes, ...]], ...] = (
    ("zip", (b"PK\x03\x04", b"PK\x05\x06")),
    ("rar", (b"Rar!\x1a\x07",)),
    ("7z", (b"7z\xbc\xaf\x27\x1c",)),
    ("gzip", (b"\x1f\x8b",)),
)


def sniff_kind(path: str | Path) -> str:
    """``zip``, ``rar``, ``7z``, ``gzip``, ``html`` or ``unknown`` from the leading bytes."""
    with open(path, "rb") as handle:
        head = handle.read(2048)
    for kind, magics in _MAGIC:
        if head.startswith(magics):
            return kind
    text = head.lstrip().lower()
    if text.startswith((b"<!doctype html", b"<html")) or b"<html" in text[:1024]:
        return "html"
    return "unknown"


def member_is_safe(name: str) -> bool:
    if not name or "\x00" in name or "\\" in name or name.startswith("/"):
        return False
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts:
        return False
    return not (pure.parts and pure.parts[0].endswith(":"))


def archive_tool() -> tuple[str, str]:
    """libarchive's bsdtar (``/usr/bin/tar`` on macOS) for RAR/7z/tar archives."""
    candidates = ["/usr/bin/tar"] if sys.platform == "darwin" else []
    found = shutil.which("bsdtar")
    if found:
        candidates.append(found)
    for candidate in candidates:
        try:
            out = subprocess.run(
                [candidate, "--version"], capture_output=True, text=True, timeout=30
            )
        except OSError:
            continue
        if "bsdtar" in out.stdout or "libarchive" in out.stdout:
            return candidate, out.stdout.strip().splitlines()[0]
    raise DataValidationError(
        "extracting RAR/7z archives needs libarchive's bsdtar (built into macOS as /usr/bin/tar); "
        "install it, or extract the archive yourself and pass the folder with --source"
    )


def safe_extract(archive: str | Path, dest: str | Path) -> dict[str, Any]:
    """Extract ``archive`` into the new, empty directory ``dest``."""
    archive, dest = Path(archive), Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise DataValidationError(f"extraction target {dest} must be a new, empty directory")
    kind = sniff_kind(archive)
    if kind == "html":
        raise DataValidationError(
            f"{archive.name} is an HTML page, not an archive (typically a download interstitial); "
            "download the file manually in a browser and pass it with --source"
        )
    if kind == "unknown":
        raise DataValidationError(f"{archive.name}: unrecognized archive format")
    dest.mkdir(parents=True, exist_ok=True)
    info = _extract_zip(archive, dest) if kind == "zip" else _extract_with_bsdtar(archive, dest)
    info.update(_post_check(dest))
    info["kind"] = kind
    return info


def _extract_zip(archive: Path, dest: Path) -> dict[str, Any]:
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        total = sum(i.file_size for i in infos)
        if len(infos) > MAX_MEMBERS or total > MAX_TOTAL_BYTES:
            raise DataValidationError(f"{archive.name}: archive exceeds member/size limits")
        for info in infos:
            if not member_is_safe(info.filename):
                raise DataValidationError(f"unsafe archive member path {info.filename!r}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise DataValidationError(f"archive member {info.filename!r} is a symbolic link")
        for info in infos:
            target = dest.joinpath(*PurePosixPath(info.filename).parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    return {"tool": "python zipfile", "members": len(infos)}


def _extract_with_bsdtar(archive: Path, dest: Path) -> dict[str, Any]:
    tool, version = archive_tool()
    listing = subprocess.run(
        [tool, "-t", "-f", str(archive)], capture_output=True, text=True, timeout=600
    )
    if listing.returncode != 0:
        raise DataValidationError(f"{tool} could not list {archive.name}: {listing.stderr[:500]}")
    names = [line for line in listing.stdout.splitlines() if line.strip()]
    if len(names) > MAX_MEMBERS:
        raise DataValidationError(f"{archive.name}: more than {MAX_MEMBERS} members")
    unsafe = [n for n in names if not member_is_safe(n.rstrip("/"))]
    if unsafe:
        raise DataValidationError(f"unsafe archive member path(s): {unsafe[:5]}")
    result = subprocess.run(
        [tool, "-x", "-f", str(archive), "-C", str(dest), "--no-same-owner"],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if result.returncode != 0:
        raise DataValidationError(f"{tool} failed to extract {archive.name}: {result.stderr[:500]}")
    return {"tool": version, "members": len(names)}


def _post_check(dest: Path) -> dict[str, int]:
    root = dest.resolve()
    files = 0
    total = 0
    for path in dest.rglob("*"):
        if path.is_symlink():
            raise DataValidationError(f"extracted entry {path} is a symbolic link; archive refused")
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            raise DataValidationError(f"extracted entry {path} resolves outside {dest}")
        if path.is_file():
            files += 1
            total += path.stat().st_size
    if files > MAX_MEMBERS or total > MAX_TOTAL_BYTES:
        raise DataValidationError(f"extracted content in {dest} exceeds member/size limits")
    return {"extracted_files": files, "extracted_bytes": total}
