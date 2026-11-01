"""Exception types that carry the process exit code the CLI should return."""

from __future__ import annotations

from fewshotlab.status import ExitCode


class FewShotLabError(Exception):
    """Base error; ``exit_code`` is what the CLI returns."""

    exit_code: int = ExitCode.FAILURE


class ConfigError(FewShotLabError):
    """A configuration file is missing, malformed or violates the protocol."""


class DataValidationError(FewShotLabError):
    """A dataset, manifest or support set failed validation."""


class NotImplementedPhaseError(FewShotLabError):
    """A command that exists in the interface but belongs to a later phase."""

    exit_code = ExitCode.NOT_IMPLEMENTED

    def __init__(self, command: str, phase: str, purpose: str) -> None:
        self.command = command
        self.phase = phase
        self.purpose = purpose
        super().__init__(
            f"NOT IMPLEMENTED: '{command}' ({purpose}) is planned for phase {phase}. "
            "Nothing was run and no files were written."
        )


class BlockedError(FewShotLabError):
    """An external prerequisite is unavailable; the message says how to resolve it."""

    exit_code = ExitCode.BLOCKED


class WeightsUnavailableError(BlockedError):
    """Pretrained weights are not cached and downloading was not permitted."""


class DeviceUnavailableError(BlockedError):
    """The requested compute device is not usable on this machine."""


class DatasetUnavailableError(BlockedError):
    """The original dataset source or archive is unavailable."""


class ProtocolRefusal(FewShotLabError):
    """A protocol guard refused the action (for example, test access before sealing)."""

    exit_code = ExitCode.REFUSED
