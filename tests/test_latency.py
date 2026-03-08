"""Test 3: Latency tracking.

Runs inference and verifies that per-token latency values are captured
for every step, TTFT is recorded, and the rolling average is computed.
"""

from __future__ import annotations

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from contextwatch.inference_loop import run_inference


def run(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> list[str]:
    """Run latency tracking validation. Returns a list of failure messages (empty = pass)."""
    failures: list[str] = []

    result = run_inference(
        model=model,
        tokenizer=tokenizer,
        prompt="Hello",
        max_tokens=50,
    )

    lat = result.latency_summary

    # 1. Latency summary must exist
    if lat is None:
        failures.append("Latency summary is None")
        return failures

    # 2. TTFT must be captured and positive
    if lat.ttft_ms is None or lat.ttft_ms <= 0:
        failures.append(f"TTFT must be > 0, got {lat.ttft_ms}")

    # 3. Current token latency must be positive
    if lat.current_token_latency_ms is None or lat.current_token_latency_ms <= 0:
        failures.append(
            f"Current token latency must be > 0, got {lat.current_token_latency_ms}"
        )

    # 4. Per-step snapshots should match generated token count
    if len(lat.per_step_snapshots) != result.generated_token_count:
        failures.append(
            f"Latency snapshot count={len(lat.per_step_snapshots)}, "
            f"expected={result.generated_token_count}"
        )

    # 5. All per-step latencies must be > 0
    for snap in lat.per_step_snapshots:
        if snap.latency_ms is not None and snap.latency_ms <= 0:
            failures.append(f"Step {snap.step}: latency_ms={snap.latency_ms}, expected > 0")
            break  # one failure is enough to flag

    # 6. Rolling average should exist with 50 tokens (window=20)
    if result.generated_token_count >= 20:
        if lat.rolling_avg_ms is None or lat.rolling_avg_ms <= 0:
            failures.append(
                f"Rolling avg should be > 0 with {result.generated_token_count} tokens, "
                f"got {lat.rolling_avg_ms}"
            )

    # 7. Trend slope should be a number (may be positive or negative)
    if lat.trend_ms_per_100_tokens is not None:
        if not isinstance(lat.trend_ms_per_100_tokens, (int, float)):
            failures.append(
                f"Trend must be numeric, got {type(lat.trend_ms_per_100_tokens)}"
            )

    return failures
