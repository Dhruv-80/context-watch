"""Latency tracking for LLM inference runs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class LatencySnapshot:
    """A single per-step snapshot of latency metrics."""

    step: int
    timestamp: float          # perf_counter timestamp
    latency_ms: float | None  # None for step 0 (TTFT not yet measured)


@dataclass
class LatencySummary:
    """Final summary produced after an inference run completes."""

    ttft_ms: float | None                           # Time To First Token
    current_token_latency_ms: float | None          # Last token latency
    rolling_avg_ms: float | None                    # Rolling average (last N tokens)
    trend_ms_per_100_tokens: float | None           # Linear trend slope
    per_step_snapshots: list[LatencySnapshot] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
class LatencyTracker:
    """Tracks latency metrics during stepwise inference.

    Args:
        rolling_window: Number of recent tokens to use for rolling average.
            Defaults to 20.
    """

    def __init__(self, rolling_window: int = 20) -> None:
        self._rolling_window = rolling_window
        self._snapshots: list[LatencySnapshot] = []
        self._start_time: float | None = None
        self._ttft_ms: float | None = None

    # -- public API ---------------------------------------------------------

    def start(self) -> None:
        """Mark the start of inference (before first forward pass)."""
        self._start_time = time.perf_counter()

    def record_step(self, step: int, step_start_time: float, step_end_time: float) -> LatencySnapshot:
        """Record latency for a single token generation step.

        Args:
            step: Zero-based generation step index.
            step_start_time: perf_counter timestamp before model forward pass.
            step_end_time: perf_counter timestamp after token selection.

        Returns:
            A :class:`LatencySnapshot` for this step.
        """
        latency_ms = (step_end_time - step_start_time) * 1000.0

        # Capture TTFT on first token
        if step == 0 and self._start_time is not None:
            self._ttft_ms = (step_end_time - self._start_time) * 1000.0

        snap = LatencySnapshot(
            step=step,
            timestamp=step_end_time,
            latency_ms=latency_ms,
        )
        self._snapshots.append(snap)

        return snap

    def summarize(self) -> LatencySummary:
        """Build a :class:`LatencySummary` from the recorded snapshots."""
        if not self._snapshots:
            return LatencySummary(
                ttft_ms=None,
                current_token_latency_ms=None,
                rolling_avg_ms=None,
                trend_ms_per_100_tokens=None,
                per_step_snapshots=[],
            )

        # Current token latency (last snapshot)
        current_latency = self._snapshots[-1].latency_ms

        # Rolling average (last N tokens)
        rolling_avg = self._compute_rolling_average()

        # Trend slope (ms per 100 tokens)
        trend_slope = self._compute_trend_slope()

        return LatencySummary(
            ttft_ms=self._ttft_ms,
            current_token_latency_ms=current_latency,
            rolling_avg_ms=rolling_avg,
            trend_ms_per_100_tokens=trend_slope,
            per_step_snapshots=list(self._snapshots),
        )

    # -- internals ----------------------------------------------------------

    def _compute_rolling_average(self) -> float | None:
        """Compute rolling average latency over the last N tokens."""
        if not self._snapshots:
            return None

        # Take last N snapshots
        window = self._snapshots[-self._rolling_window :]
        latencies = [s.latency_ms for s in window if s.latency_ms is not None]

        if not latencies:
            return None

        return sum(latencies) / len(latencies)

    def _compute_trend_slope(self) -> float | None:
        """Compute linear trend slope using least-squares regression.

        Returns:
            Slope in ms per 100 tokens, or None if insufficient data.
        """
        if len(self._snapshots) < 2:
            return None

        # Extract (x, y) pairs: x = step index, y = latency_ms
        pairs = [(s.step, s.latency_ms) for s in self._snapshots if s.latency_ms is not None]

        if len(pairs) < 2:
            return None

        n = len(pairs)
        sum_x = sum(x for x, _ in pairs)
        sum_y = sum(y for _, y in pairs)
        sum_xy = sum(x * y for x, y in pairs)
        sum_x2 = sum(x * x for x, _ in pairs)

        # Least-squares slope: (n·Σxy - Σx·Σy) / (n·Σx² - (Σx)²)
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return None

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        # Normalize to ms per 100 tokens
        return slope * 100.0


# TODO(Phase 4): Consider adding latency percentiles (p50, p95, p99)
# TODO(Phase 4): Consider adding latency histogram for visualization
