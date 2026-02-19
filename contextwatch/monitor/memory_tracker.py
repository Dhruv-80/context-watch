"""Process memory tracking for LLM inference runs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import psutil


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BYTES_PER_MB = 1024 * 1024
_BYTES_PER_GB = 1024 * 1024 * 1024


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class MemorySnapshot:
    """A single per-step snapshot of process memory usage."""

    step: int
    rss_bytes: int
    rss_mb: float
    delta_from_start_mb: float


@dataclass
class MemorySummary:
    """Final summary produced after an inference run completes."""

    initial_memory_mb: float
    current_memory_mb: float
    peak_memory_mb: float
    memory_growth_total_mb: float
    avg_growth_per_token_mb: float
    growth_per_100_tokens_mb: float
    per_step_snapshots: list[MemorySnapshot] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
class MemoryTracker:
    """Tracks process memory (RSS) during stepwise inference.

    Uses ``psutil.Process().memory_info().rss`` to sample the resident
    set size of the current process after each generation step.

    Args:
        rolling_window: Reserved for future use.  Defaults to 20.
    """

    def __init__(self, rolling_window: int = 20) -> None:
        self._process = psutil.Process(os.getpid())
        self._initial_rss: int | None = None
        self._peak_rss: int = 0
        self._snapshots: list[MemorySnapshot] = []

    # -- public API ---------------------------------------------------------

    def start(self) -> None:
        """Record baseline memory before generation begins."""
        self._initial_rss = self._sample_rss()
        self._peak_rss = self._initial_rss

    def record_step(self, step: int) -> MemorySnapshot:
        """Record memory usage after a token generation step.

        Args:
            step: Zero-based generation step index.

        Returns:
            A :class:`MemorySnapshot` for this step.
        """
        rss = self._sample_rss()
        self._peak_rss = max(self._peak_rss, rss)

        initial = self._initial_rss or rss
        delta_bytes = rss - initial

        snap = MemorySnapshot(
            step=step,
            rss_bytes=rss,
            rss_mb=round(rss / _BYTES_PER_MB, 2),
            delta_from_start_mb=round(delta_bytes / _BYTES_PER_MB, 2),
        )
        self._snapshots.append(snap)

        return snap

    def summarize(self) -> MemorySummary:
        """Build a :class:`MemorySummary` from the recorded snapshots.

        Growth math:
            * ``memory_growth_total_mb`` = current RSS − initial RSS
            * ``avg_growth_per_token_mb`` = total growth / number of tokens
            * ``growth_per_100_tokens_mb`` = avg_growth_per_token * 100
        """
        initial = self._initial_rss or 0
        initial_mb = initial / _BYTES_PER_MB

        if not self._snapshots:
            return MemorySummary(
                initial_memory_mb=round(initial_mb, 2),
                current_memory_mb=round(initial_mb, 2),
                peak_memory_mb=round(self._peak_rss / _BYTES_PER_MB, 2),
                memory_growth_total_mb=0.0,
                avg_growth_per_token_mb=0.0,
                growth_per_100_tokens_mb=0.0,
                per_step_snapshots=[],
            )

        last = self._snapshots[-1]
        current_mb = last.rss_bytes / _BYTES_PER_MB
        peak_mb = self._peak_rss / _BYTES_PER_MB
        growth_total_mb = (last.rss_bytes - initial) / _BYTES_PER_MB

        n_tokens = len(self._snapshots)
        avg_growth = growth_total_mb / n_tokens if n_tokens > 0 else 0.0
        growth_per_100 = avg_growth * 100.0

        return MemorySummary(
            initial_memory_mb=round(initial_mb, 2),
            current_memory_mb=round(current_mb, 2),
            peak_memory_mb=round(peak_mb, 2),
            memory_growth_total_mb=round(growth_total_mb, 2),
            avg_growth_per_token_mb=round(avg_growth, 4),
            growth_per_100_tokens_mb=round(growth_per_100, 2),
            per_step_snapshots=list(self._snapshots),
        )

    # -- internals ----------------------------------------------------------

    def _sample_rss(self) -> int:
        """Return current process RSS in bytes."""
        return self._process.memory_info().rss


# TODO: Phase 5 will consume growth rate for forecasting
