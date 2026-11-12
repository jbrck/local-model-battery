"""Run guard for the LiveCodeBench leg: budget, stall and failure-streak stops."""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass


class FailureKind(enum.Enum):
    """Why a problem produced no code, in the guard's terms."""

    NONE = "none"
    ENGINE = "engine"
    NO_CODE = "no_code"


@dataclass
class GuardDecision:
    """The guard's ruling after one problem."""

    stop: bool
    reason: str = ""


class Guard:
    """Stops a leg that cannot finish, without discarding the work it already has."""

    def __init__(self, max_hours: float, stall_min: int, max_failures: int,
                 started_at: float | None = None):
        self.max_hours = max_hours
        self.stall_min = stall_min
        self.max_failures = max_failures
        self.started_at = started_at if started_at is not None else time.monotonic()
        self.last_code_at = self.started_at
        self.engine_streak = 0
        self.no_code_streak = 0

    def remaining_seconds(self, now: float) -> float:
        """Seconds left in the wall-clock budget, floored at zero."""
        return max(0.0, self.max_hours * 3600 - (now - self.started_at))

    def record(self, kind: FailureKind, now: float) -> GuardDecision:
        """Account for one finished problem and rule on whether to keep going."""
        if kind is FailureKind.ENGINE:
            self.engine_streak += 1
            self.no_code_streak = 0
        elif kind is FailureKind.NO_CODE:
            self.no_code_streak += 1
            self.engine_streak = 0
        else:
            self.engine_streak = 0
            self.no_code_streak = 0
            self.last_code_at = now
        if self.engine_streak >= self.max_failures:
            return GuardDecision(True, f"{self.engine_streak} consecutive engine failures")
        if self.no_code_streak >= self.max_failures:
            return GuardDecision(True,
                                 f"{self.no_code_streak} consecutive problems with no code")
        if now - self.started_at >= self.max_hours * 3600:
            return GuardDecision(True, f"wall clock reached the {self.max_hours:.1f}h budget")
        if self.stall_min > 0 and now - self.last_code_at >= self.stall_min * 60:
            return GuardDecision(True, f"stall: no extractable code for {self.stall_min} minutes")
        return GuardDecision(False)
