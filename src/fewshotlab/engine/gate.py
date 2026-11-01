"""The final-test gate. Until the P5 ``lock-test`` workflow seals an evaluation plan, every request
to evaluate on the held-out test split is refused, and nothing is computed or written."""

from __future__ import annotations

from fewshotlab.errors import ProtocolRefusal
from fewshotlab.paths import ProjectPaths

SEALED_PLAN_NAME = "sealed_plan.json"


def require_sealed_plan(paths: ProjectPaths) -> None:
    raise ProtocolRefusal(
        "test evaluation is locked: no sealed evaluation plan exists. The P5 `lock-test` "
        "workflow must seal the plan, code, data, supports, checkpoints and selection rules "
        "first. Nothing was evaluated and no file was written."
    )
