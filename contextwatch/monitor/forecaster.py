"""Forecasting engine — predicts resource exhaustion from observed metrics.

Uses simple, deterministic math (no ML, no sklearn).  All formulas are
linear extrapolations from metrics already collected by the context,
latency, and memory trackers.

Assumptions & limitations
-------------------------
* Context forecast is exact (deterministic subtraction).
* Memory forecast assumes constant average growth per token.  In practice
  the first few tokens cause a burst (KV-cache allocation), so the
  forecast is conservative for short runs and more accurate for long runs.
* Latency forecast assumes the linear trend holds.  Real latency often
  has non-linear characteristics at high context utilisation.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ForecastResult:
    """Structured predictions for resource exhaustion.

    All ``tokens_until_*`` fields are expressed as the estimated number
    of **additional** tokens that can be generated before the limit is
    reached.  ``None`` means the forecast is not applicable (e.g. no
    latency degradation trend, or the limit was not specified).
    """

    # Context
    tokens_until_context_limit: int | None = None
    context_already_saturated: bool = False

    # Memory
    tokens_until_memory_limit: int | None = None
    memory_already_exceeded: bool = False
    memory_limit_mb: float | None = None

    # Latency
    tokens_until_latency_threshold: int | None = None
    latency_already_exceeded: bool = False
    latency_threshold_ms: float | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_forecast(
    *,
    # Context inputs
    total_tokens: int,
    max_context: int,
    # Memory inputs (all in MB)
    current_memory_mb: float | None = None,
    avg_growth_per_token_mb: float | None = None,
    memory_limit_mb: float | None = None,
    # Latency inputs
    current_latency_ms: float | None = None,
    latency_slope_per_token_ms: float | None = None,
    latency_threshold_ms: float | None = None,
) -> ForecastResult:
    """Compute resource-exhaustion forecasts from observed metrics.

    Args:
        total_tokens: Current total token count (prompt + generated).
        max_context: Maximum context window size for the model.
        current_memory_mb: Current process RSS in MB.
        avg_growth_per_token_mb: Average RSS growth per token in MB.
        memory_limit_mb: User-specified memory ceiling in MB.
        current_latency_ms: Most recent per-token latency in ms.
        latency_slope_per_token_ms: Linear trend slope in ms per token.
        latency_threshold_ms: User-specified latency ceiling in ms.

    Returns:
        A :class:`ForecastResult` with all applicable predictions.
    """
    result = ForecastResult()

    # -----------------------------------------------------------------------
    # 1. Context forecast  (deterministic)
    #    tokens_remaining = max_context - total_tokens
    # -----------------------------------------------------------------------
    remaining_ctx = max_context - total_tokens
    if remaining_ctx <= 0:
        result.context_already_saturated = True
        result.tokens_until_context_limit = 0
    else:
        result.tokens_until_context_limit = remaining_ctx

    # -----------------------------------------------------------------------
    # 2. Memory forecast  (linear extrapolation)
    #    remaining_memory = memory_limit - current_memory
    #    tokens_until     = remaining_memory / avg_growth_per_token
    #
    #    Edge cases:
    #      - avg_growth <= 0 → memory is not growing → None (won't hit limit)
    #      - remaining <= 0  → already exceeded
    # -----------------------------------------------------------------------
    if memory_limit_mb is not None:
        result.memory_limit_mb = memory_limit_mb

        if current_memory_mb is not None:
            remaining_mem = memory_limit_mb - current_memory_mb

            if remaining_mem <= 0:
                result.memory_already_exceeded = True
                result.tokens_until_memory_limit = 0
            elif (
                avg_growth_per_token_mb is not None
                and avg_growth_per_token_mb > 0
            ):
                # Linear extrapolation
                tokens = remaining_mem / avg_growth_per_token_mb
                result.tokens_until_memory_limit = max(0, int(tokens))
            # else: growth is zero or negative → memory won't reach limit

    # -----------------------------------------------------------------------
    # 3. Latency forecast  (solve linear equation)
    #    threshold = current + slope * n
    #    ⟹  n = (threshold - current) / slope
    #
    #    Edge cases:
    #      - slope <= 0 → latency is not increasing → None
    #      - current >= threshold → already exceeded
    # -----------------------------------------------------------------------
    if latency_threshold_ms is not None:
        result.latency_threshold_ms = latency_threshold_ms

        if current_latency_ms is not None:
            if current_latency_ms >= latency_threshold_ms:
                result.latency_already_exceeded = True
                result.tokens_until_latency_threshold = 0
            elif (
                latency_slope_per_token_ms is not None
                and latency_slope_per_token_ms > 0
            ):
                # Solve: threshold = current + slope * n
                n = (latency_threshold_ms - current_latency_ms) / latency_slope_per_token_ms
                result.tokens_until_latency_threshold = max(0, int(n))
            # else: slope <= 0 → latency is decreasing → won't reach threshold

    return result


# TODO: Future improvement: non-linear modeling
