"""Monitoring subsystem for ContextWatch.

Imports are lazy so light-weight commands (for example report generation)
can run without torch/transformers/psutil installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextwatch.monitor.advisor import Diagnosis as Diagnosis
    from contextwatch.monitor.advisor import Finding as Finding
    from contextwatch.monitor.advisor import build_diagnosis as build_diagnosis
    from contextwatch.monitor.context_tracker import ContextSnapshot as ContextSnapshot
    from contextwatch.monitor.context_tracker import ContextSummary as ContextSummary
    from contextwatch.monitor.context_tracker import ContextTracker as ContextTracker
    from contextwatch.monitor.forecaster import ForecastResult as ForecastResult
    from contextwatch.monitor.forecaster import compute_forecast as compute_forecast
    from contextwatch.monitor.latency_tracker import LatencySnapshot as LatencySnapshot
    from contextwatch.monitor.latency_tracker import LatencySummary as LatencySummary
    from contextwatch.monitor.latency_tracker import LatencyTracker as LatencyTracker
    from contextwatch.monitor.memory_tracker import MemorySnapshot as MemorySnapshot
    from contextwatch.monitor.memory_tracker import MemorySummary as MemorySummary
    from contextwatch.monitor.memory_tracker import MemoryTracker as MemoryTracker

__all__ = [
    "Diagnosis",
    "Finding",
    "build_diagnosis",
    "ContextSnapshot",
    "ContextSummary",
    "ContextTracker",
    "ForecastResult",
    "compute_forecast",
    "LatencySnapshot",
    "LatencySummary",
    "LatencyTracker",
    "MemorySnapshot",
    "MemorySummary",
    "MemoryTracker",
]


def __getattr__(name: str):
    if name in {"Diagnosis", "Finding", "build_diagnosis"}:
        from contextwatch.monitor import advisor as mod

        return getattr(mod, name)
    if name in {"ContextSnapshot", "ContextSummary", "ContextTracker"}:
        from contextwatch.monitor import context_tracker as mod

        return getattr(mod, name)
    if name in {"ForecastResult", "compute_forecast"}:
        from contextwatch.monitor import forecaster as mod

        return getattr(mod, name)
    if name in {"LatencySnapshot", "LatencySummary", "LatencyTracker"}:
        from contextwatch.monitor import latency_tracker as mod

        return getattr(mod, name)
    if name in {"MemorySnapshot", "MemorySummary", "MemoryTracker"}:
        from contextwatch.monitor import memory_tracker as mod

        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
