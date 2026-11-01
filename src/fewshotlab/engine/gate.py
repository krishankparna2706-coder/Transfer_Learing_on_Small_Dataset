"""The final-test gate.

Test images are reachable only through the sealed study workflow: ``lock-test --plan PATH`` seals
the plan, then ``evaluate --split test --study PATH`` verifies the seal and evaluates each sealed
run once (:mod:`fewshotlab.engine.test_eval`). Any other request for the test split is refused,
and nothing is computed or written.
"""

from __future__ import annotations

from fewshotlab.errors import ProtocolRefusal
from fewshotlab.paths import ProjectPaths

SEALED_PLAN_NAME = "sealed_plan.json"


def require_sealed_plan(paths: ProjectPaths) -> None:
    """Refuse a test request made outside the sealed study workflow."""
    raise ProtocolRefusal(
        "test evaluation is locked: test access runs only through the sealed study workflow "
        "(`lock-test --plan PATH`, then `evaluate --split test --study PATH`), which verifies "
        "the sealed plan, code, data, supports, checkpoints and selection rules first. Nothing "
        "was evaluated and no file was written."
    )
