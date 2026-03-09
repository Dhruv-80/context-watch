"""Run advisor — convert metrics and forecasts into actionable guidance."""

from __future__ import annotations

from dataclasses import dataclass, field

from contextwatch.monitor.forecaster import ForecastResult


@dataclass
class Finding:
    """Single diagnostic finding."""

    area: str
    severity: str
    message: str


@dataclass
class Diagnosis:
    """Actionable diagnosis for an inference run."""

    risk_score: int
    status: str
    findings: list[Finding] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


def build_diagnosis(
    *,
    context_used_pct: float | None,
    latency_trend_ms_per_100_tokens: float | None,
    forecast: ForecastResult,
    mode: str,
) -> Diagnosis:
    """Build a deterministic diagnosis from tracker summaries and forecast."""
    score = 0
    findings: list[Finding] = []
    recommendations: list[str] = []

    # Context pressure
    if forecast.context_already_saturated:
        score += 45
        findings.append(
            Finding("context", "critical", "Context window is saturated.")
        )
        recommendations.append(
            "Shorten prompt/history or switch to a model with a larger context window."
        )
    elif context_used_pct is not None:
        if context_used_pct >= 0.90:
            score += 35
            findings.append(
                Finding("context", "high", "Context usage is above 90%.")
            )
            recommendations.append(
                "Reserve headroom: cap generation or trim prompt to avoid hard stops."
            )
        elif context_used_pct >= 0.75:
            score += 20
            findings.append(
                Finding("context", "medium", "Context usage is above 75%.")
            )

    # Memory pressure
    if forecast.memory_already_exceeded:
        score += 40
        findings.append(
            Finding("memory", "critical", "Memory limit has already been exceeded.")
        )
        recommendations.append(
            "Lower max tokens, use smaller/quantized weights, or raise memory budget."
        )
    elif forecast.tokens_until_memory_limit is not None:
        if forecast.tokens_until_memory_limit < 200:
            score += 25
            findings.append(
                Finding(
                    "memory",
                    "high",
                    f"Estimated memory limit in ~{forecast.tokens_until_memory_limit} tokens.",
                )
            )
            recommendations.append(
                "Reduce generation length or increase available memory."
            )
        elif forecast.tokens_until_memory_limit < 1000:
            score += 12
            findings.append(
                Finding(
                    "memory",
                    "medium",
                    f"Memory headroom is moderate (~{forecast.tokens_until_memory_limit} tokens).",
                )
            )

    # Latency pressure
    if forecast.latency_already_exceeded:
        score += 35
        findings.append(
            Finding("latency", "high", "Latency threshold is already exceeded.")
        )
        recommendations.append(
            "Reduce max tokens or context length; consider faster hardware/backend settings."
        )
    elif forecast.tokens_until_latency_threshold is not None:
        if forecast.tokens_until_latency_threshold < 200:
            score += 20
            findings.append(
                Finding(
                    "latency",
                    "high",
                    f"Estimated latency threshold in ~{forecast.tokens_until_latency_threshold} tokens.",
                )
            )
        elif forecast.tokens_until_latency_threshold < 1000:
            score += 10
            findings.append(
                Finding(
                    "latency",
                    "medium",
                    f"Latency threshold may be hit in ~{forecast.tokens_until_latency_threshold} tokens.",
                )
            )

    if latency_trend_ms_per_100_tokens is not None and latency_trend_ms_per_100_tokens > 5:
        score += 8
        findings.append(
            Finding(
                "latency",
                "medium",
                f"Latency trend is rising (+{latency_trend_ms_per_100_tokens:.1f} ms/100 tokens).",
            )
        )

    if mode == "vllm":
        findings.append(
            Finding(
                "memory",
                "info",
                "Memory is server-managed in vLLM mode; local RSS is unavailable.",
            )
        )
        recommendations.append(
            "Track GPU memory externally (for example, nvidia-smi or vLLM observability stack)."
        )

    # Deduplicate recommendation lines while preserving order.
    deduped_recs: list[str] = []
    seen = set()
    for rec in recommendations:
        if rec not in seen:
            seen.add(rec)
            deduped_recs.append(rec)

    risk_score = max(0, min(100, score))
    if risk_score >= 70:
        status = "critical"
    elif risk_score >= 40:
        status = "elevated"
    elif risk_score >= 20:
        status = "watch"
    else:
        status = "stable"

    if not deduped_recs:
        deduped_recs.append("System is stable. Continue monitoring longer runs for drift.")

    return Diagnosis(
        risk_score=risk_score,
        status=status,
        findings=findings,
        recommendations=deduped_recs,
    )
