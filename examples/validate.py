#!/usr/bin/env python3
"""Validation script — runs ContextWatch with distilgpt2 and checks token counts."""

from __future__ import annotations

from contextwatch.inference_loop import run_inference
from contextwatch.utils import load_model


def main() -> None:
    model_name = "distilgpt2"
    prompt = "Hello"
    max_tokens = 20

    print(f"=== ContextWatch Validation ===")
    print(f"Model:      {model_name}")
    print(f"Prompt:     \"{prompt}\"")
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
    print(f"Generated text: \"{result.generated_text}\"")

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
    # Rolling average should exist for 20 tokens
    if result.generated_token_count >= 20:
        assert lat.rolling_avg_ms is not None and lat.rolling_avg_ms > 0, (
            f"Rolling avg should exist for {result.generated_token_count} tokens"
        )
    # Trend should be numeric (can be positive, negative, or zero)
    if lat.trend_ms_per_100_tokens is not None:
        assert isinstance(lat.trend_ms_per_100_tokens, (int, float)), (
            f"Trend must be numeric, got {type(lat.trend_ms_per_100_tokens)}"
        )

    print("\n✅ All assertions passed.")


if __name__ == "__main__":
    main()
