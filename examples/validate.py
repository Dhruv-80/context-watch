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

    print("\n✅ All assertions passed.")


if __name__ == "__main__":
    main()
