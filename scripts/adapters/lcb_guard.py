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
        self.last_answer_at = self.started_at
        self.engine_streak = 0
        self.no_code_streak = 0

    def remaining_seconds(self, now: float) -> float:
        """Seconds left in the wall-clock budget, floored at zero."""
        return max(0.0, self.max_hours * 3600 - (now - self.started_at))

    def _account(self, kind: FailureKind, now: float) -> None:
        """Update the streaks and the last-code timestamp for one finished problem."""
        if kind is FailureKind.ENGINE:
            self.engine_streak += 1
            self.no_code_streak = 0
        elif kind is FailureKind.NO_CODE:
            # The engine answered, it just gave no code. That is progress, not a stall:
            # a reasoner running to its cap takes longer than the stall window on one
            # problem. The no-code streak catches an extractor that never matches.
            self.no_code_streak += 1
            self.engine_streak = 0
            self.last_answer_at = now
        else:
            self.engine_streak = 0
            self.no_code_streak = 0
            self.last_answer_at = now

    def _verdict(self, now: float) -> GuardDecision:
        """Rule on the counters as they stand after one problem."""
        if self.engine_streak >= self.max_failures:
            return GuardDecision(True, f"{self.engine_streak} consecutive engine failures")
        if self.no_code_streak >= self.max_failures:
            return GuardDecision(True,
                                 f"{self.no_code_streak} consecutive problems with no code")
        if now - self.started_at >= self.max_hours * 3600:
            return GuardDecision(True, f"wall clock reached the {self.max_hours:.1f}h budget")
        if self.stall_min > 0 and now - self.last_answer_at >= self.stall_min * 60:
            return GuardDecision(True, f"stall: no engine answer for {self.stall_min} minutes")
        return GuardDecision(False)

    def record(self, kind: FailureKind, now: float) -> GuardDecision:
        """Account for one finished problem and rule on whether to keep going."""
        self._account(kind, now)
        return self._verdict(now)
