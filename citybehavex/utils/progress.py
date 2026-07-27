from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"


@dataclass
class ProgressReporter:
    label: str
    total: int
    unit: str
    rate_precision: int = 1
    min_interval_seconds: float = 0.0
    checkpoint_every: int | None = None
    emit: Callable[[str], None] = print
    on_report: Callable[[], None] | None = None
    started: float = field(default_factory=time.perf_counter)
    last_report_elapsed: float = 0.0

    def report(
        self,
        done: int,
        *,
        detail: str | None = None,
        checkpoint_count: int | None = None,
        done_batches: int | None = None,
        total_batches: int | None = None,
        elapsed: float | None = None,
    ) -> bool:
        elapsed = time.perf_counter() - self.started if elapsed is None else elapsed
        if not self._should_report(done, elapsed, checkpoint_count):
            return False

        self.last_report_elapsed = elapsed
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = self.total - done
        eta = remaining / rate if rate > 0 else float("nan")

        parts = [f"{self.label}: "]
        if done_batches is not None and total_batches is not None:
            parts.append(f"{done_batches}/{total_batches} batches, ")
        parts.append(f"{done}/{self.total} {self.unit}")
        if detail:
            parts.append(f" {detail}")
        parts.append(
            f", {rate:.{self.rate_precision}f} {self.unit}/sec, "
            f"{elapsed:.1f}s elapsed, ETA {format_duration(eta)}"
        )
        self.emit("".join(parts))
        if self.on_report is not None:
            self.on_report()
        return True

    def _should_report(
        self,
        done: int,
        elapsed: float,
        checkpoint_count: int | None,
    ) -> bool:
        if self.checkpoint_every is not None and checkpoint_count is not None:
            return self.checkpoint_every > 0 and checkpoint_count % self.checkpoint_every == 0
        if done >= self.total:
            return True
        if self.min_interval_seconds > 0:
            return elapsed - self.last_report_elapsed >= self.min_interval_seconds
        return True
