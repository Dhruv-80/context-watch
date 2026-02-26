#!/usr/bin/env python3
"""Validation script — runs ContextWatch with distilgpt2 and checks token counts."""

from __future__ import annotations

from contextwatch.inference_loop import run_inference
from contextwatch.monitor.forecaster import compute_forecast
from contextwatch.utils import load_model


def main() -> None:
    model_name = "distilgpt2"
    prompt = "Hello"
    max_tokens = 20

    print(f"=== ContextWatch Validation ===")
    print(f"Model:      {model_name}")
    print(f'Prompt:     "{prompt}"')
    print(f"Max tokens: {max_tokens}\n")

    model, tokenizer = load_model(model_name)

    result = run_inference(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
    )

    print(f"\nPrompt tokens: {result.prompt_token_count}")
    print(f"Generated tokens: {result.generated_token_count}")
    print(f"Total tokens: {result.total_token_count}")
    print(f'Generated text: "{result.generated_text}"')

    # --- context tracking output (Phase 2) ---------------------------------
    ctx = result.context_summary
    if ctx is not None:
        pct = round(ctx.context_used_pct * 100, 1)
        print(f"\nContext: {pct}% ({ctx.final_total_tokens}/{ctx.max_context})")
        print(f"Remaining tokens: {ctx.remaining_tokens}")
        print(f"Per-step snapshots recorded: {len(ctx.per_step_snapshots)}")
        if ctx.warning_issued:
            print("(!) Context warning was triggered during generation.")

    # --- latency tracking output (Phase 3) ---------------------------------
    lat = result.latency_summary
    if lat is not None:
        print("\nLatency Metrics:")
        if lat.ttft_ms is not None:
            print(f"  TTFT: {lat.ttft_ms:.1f} ms")
        if lat.current_token_latency_ms is not None:
            print(f"  Current token latency: {lat.current_token_latency_ms:.1f} ms")
        if lat.rolling_avg_ms is not None:
            print(f"  Rolling avg (last 20): {lat.rolling_avg_ms:.1f} ms")
        if lat.trend_ms_per_100_tokens is not None:
            sign = "+" if lat.trend_ms_per_100_tokens >= 0 else ""
            print(f"  Trend: {sign}{lat.trend_ms_per_100_tokens:.1f} ms per 100 tokens")
        print(f"  Per-step snapshots recorded: {len(lat.per_step_snapshots)}")

    # --- memory tracking output (Phase 4) ----------------------------------
    mem = result.memory_summary
    if mem is not None:
        print("\nMemory Metrics:")
        print(f"  Initial memory: {mem.initial_memory_mb:.2f} MB")
        print(f"  Current memory: {mem.current_memory_mb:.2f} MB")
        print(f"  Peak memory: {mem.peak_memory_mb:.2f} MB")
        print(f"  Total growth: +{mem.memory_growth_total_mb:.2f} MB")
        print(f"  Avg per token: {mem.avg_growth_per_token_mb:.4f} MB")
        print(f"  Growth rate: +{mem.growth_per_100_tokens_mb:.2f} MB per 100 tokens")
        print(f"  Per-step snapshots recorded: {len(mem.per_step_snapshots)}")

    # --- forecast output (Phase 5) -----------------------------------------
    forecast = None
    if ctx is not None:
        latency_slope_per_token = None
        if lat is not None and lat.trend_ms_per_100_tokens is not None:
            latency_slope_per_token = lat.trend_ms_per_100_tokens / 100.0

        # Use 1 GB as a test memory limit
        memory_limit_mb = 1024.0

        forecast = compute_forecast(
            total_tokens=result.total_token_count,
            max_context=ctx.max_context,
            current_memory_mb=mem.current_memory_mb if mem else None,
            avg_growth_per_token_mb=mem.avg_growth_per_token_mb if mem else None,
            memory_limit_mb=memory_limit_mb,
            current_latency_ms=lat.current_token_latency_ms if lat else None,
            latency_slope_per_token_ms=latency_slope_per_token,
            latency_threshold_ms=100.0,  # test with 100ms threshold
        )

        print("\nForecast (test: memory_limit=1GB, latency_limit=100ms):")
        if forecast.tokens_until_context_limit is not None:
            print(f"  Context saturation in: ~{forecast.tokens_until_context_limit} tokens")
        if forecast.tokens_until_memory_limit is not None:
            print(f"  Memory limit (1 GB) in: ~{forecast.tokens_until_memory_limit} tokens")
        elif not forecast.memory_already_exceeded:
            print("  Memory limit (1 GB): unlikely to be reached")
        if forecast.tokens_until_latency_threshold is not None:
            print(f"  Latency >100ms in: ~{forecast.tokens_until_latency_threshold} tokens")
        elif not forecast.latency_already_exceeded:
            print("  Latency >100ms: no degradation trend")

    # --- basic assertions --------------------------------------------------
    assert result.total_token_count == result.prompt_token_count + result.generated_token_count, (
        f"Token count mismatch: {result.total_token_count} != "
        f"{result.prompt_token_count} + {result.generated_token_count}"
    )
    assert result.generated_token_count <= max_tokens, (
        f"Generated more tokens than allowed: {result.generated_token_count} > {max_tokens}"
    )
    assert result.prompt_token_count > 0, "Prompt token count must be > 0"

    # --- context tracking assertions (Phase 2) -----------------------------
    assert ctx is not None, "Context summary must not be None"
    assert ctx.max_context > 0, f"Max context must be > 0, got {ctx.max_context}"
    assert ctx.remaining_tokens == ctx.max_context - ctx.final_total_tokens, (
        f"Remaining mismatch: {ctx.remaining_tokens} != "
        f"{ctx.max_context} - {ctx.final_total_tokens}"
    )
    assert 0.0 <= ctx.context_used_pct <= 1.0, (
        f"Context used % out of range: {ctx.context_used_pct}"
    )
    assert len(ctx.per_step_snapshots) == result.generated_token_count, (
        f"Snapshot count mismatch: {len(ctx.per_step_snapshots)} != "
        f"{result.generated_token_count}"
    )

    # --- latency tracking assertions (Phase 3) -----------------------------
    assert lat is not None, "Latency summary must not be None"
    assert lat.ttft_ms is not None and lat.ttft_ms > 0, (
        f"TTFT must be > 0, got {lat.ttft_ms}"
    )
    assert lat.current_token_latency_ms is not None and lat.current_token_latency_ms > 0, (
        f"Current token latency must be > 0, got {lat.current_token_latency_ms}"
    )
    assert len(lat.per_step_snapshots) == result.generated_token_count, (
        f"Latency snapshot count mismatch: {len(lat.per_step_snapshots)} != "
        f"{result.generated_token_count}"
    )
    if result.generated_token_count >= 20:
        assert lat.rolling_avg_ms is not None and lat.rolling_avg_ms > 0, (
            f"Rolling avg should exist for {result.generated_token_count} tokens"
        )
    if lat.trend_ms_per_100_tokens is not None:
        assert isinstance(lat.trend_ms_per_100_tokens, (int, float)), (
            f"Trend must be numeric, got {type(lat.trend_ms_per_100_tokens)}"
        )

    # --- memory tracking assertions (Phase 4) ------------------------------
    assert mem is not None, "Memory summary must not be None"
    assert mem.initial_memory_mb > 0, (
        f"Initial memory must be > 0, got {mem.initial_memory_mb}"
    )
    assert mem.current_memory_mb > 0, (
        f"Current memory must be > 0, got {mem.current_memory_mb}"
    )
    assert mem.peak_memory_mb >= mem.current_memory_mb, (
        f"Peak memory ({mem.peak_memory_mb}) must be >= current ({mem.current_memory_mb})"
    )
    assert mem.memory_growth_total_mb >= 0, (
        f"Memory growth should be >= 0, got {mem.memory_growth_total_mb}"
    )
    assert len(mem.per_step_snapshots) == result.generated_token_count, (
        f"Memory snapshot count mismatch: {len(mem.per_step_snapshots)} != "
        f"{result.generated_token_count}"
    )

    # --- forecast assertions (Phase 5) -------------------------------------
    assert forecast is not None, "Forecast must not be None"

    # Context forecast: must exactly match remaining tokens from context tracker
    assert forecast.tokens_until_context_limit == ctx.remaining_tokens, (
        f"Context forecast mismatch: {forecast.tokens_until_context_limit} != "
        f"{ctx.remaining_tokens}"
    )
    assert not forecast.context_already_saturated, (
        "Context should not be saturated after only 20 tokens"
    )

    # Memory forecast: with 1GB limit and ~660MB current, should have room
    if forecast.tokens_until_memory_limit is not None:
        assert forecast.tokens_until_memory_limit > 0, (
            f"Should have tokens remaining before 1GB limit, got {forecast.tokens_until_memory_limit}"
        )
    assert not forecast.memory_already_exceeded, (
        "Memory should not exceed 1GB after 20 tokens with distilgpt2"
    )

    # Latency forecast: with 100ms threshold and ~5-7ms current latency,
    # either we get a positive token count or None (no degradation trend)
    if forecast.tokens_until_latency_threshold is not None:
        assert forecast.tokens_until_latency_threshold >= 0, (
            f"Latency forecast must be >= 0, got {forecast.tokens_until_latency_threshold}"
        )
    assert not forecast.latency_already_exceeded, (
        "Latency should not exceed 100ms with distilgpt2 on CPU"
    )

    print("\n✅ All assertions passed.")


if __name__ == "__main__":
    main()
