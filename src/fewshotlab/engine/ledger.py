"""The append-only run ledger (``runs/ledger.jsonl``).

Each line is an event, and status changes are appended, never edited. The latest status event
of a (run_id, attempt_id) pair is that attempt's current status.
"""

from __future__ import annotations

import json
from typing import Any

from fewshotlab.io_utils import append_jsonl, utc_now
from fewshotlab.paths import ProjectPaths

LEDGER_VERSION = 1


def append_event(paths: ProjectPaths, **fields: Any) -> dict[str, Any]:
    event = {"ledger_version": LEDGER_VERSION, "time_utc": utc_now(), **fields}
    append_jsonl(paths.ledger, event)
    return event


def read_events(paths: ProjectPaths) -> list[dict[str, Any]]:
    if not paths.ledger.is_file():
        return []
    events = []
    for line in paths.ledger.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def latest_status(paths: ProjectPaths) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for event in read_events(paths):
        if event.get("event") == "status":
            latest[(event["run_id"], event["attempt_id"])] = event
    return latest
