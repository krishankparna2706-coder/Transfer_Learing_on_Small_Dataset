"""Status vocabularies and process exit codes shared by the CLI, engine and reports."""

from __future__ import annotations

from enum import StrEnum


class RunStatus(StrEnum):
    """Lifecycle of one run. A missing metric is ``null``; it is never written as zero."""

    NOT_RUN = "NOT_RUN"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INVALIDATED = "INVALIDATED"


#: Legal transitions. Finished results are never edited in place; they can only be marked
#: INVALIDATED, and the original record is retained.
ALLOWED_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.NOT_RUN: frozenset({RunStatus.RUNNING}),
    RunStatus.RUNNING: frozenset({RunStatus.COMPLETED, RunStatus.FAILED}),
    RunStatus.COMPLETED: frozenset({RunStatus.INVALIDATED}),
    RunStatus.FAILED: frozenset({RunStatus.INVALIDATED}),
    RunStatus.INVALIDATED: frozenset(),
}


def check_transition(old: RunStatus, new: RunStatus) -> None:
    """Raise ``ValueError`` for a transition the lifecycle does not allow."""
    if new not in ALLOWED_TRANSITIONS[old]:
        raise ValueError(f"illegal run status transition {old.value} -> {new.value}")


class DataStatus(StrEnum):
    """Preparation state of a dataset."""

    NOT_PREPARED = "NOT_PREPARED"
    PREPARED = "PREPARED"
    BLOCKED = "BLOCKED"  # an external prerequisite (source, archive, permission) is unavailable
    FAILED = "FAILED"  # preparation ran and found a problem


class ExitCode:
    """Process exit codes used by every command."""

    OK = 0
    FAILURE = 1  # the command ran and failed (validation error, bad input, ...)
    USAGE = 2  # argparse usage error
    NOT_IMPLEMENTED = 3  # in the interface, but planned for a later phase
    BLOCKED = 4  # missing external prerequisite (dataset source, weights, device)
    REFUSED = 5  # a protocol guard refused the action (e.g. test access before sealing)
