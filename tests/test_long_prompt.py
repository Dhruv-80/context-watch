"""Test 2: Long prompt test.

Builds a ~1000-token prompt by repeating text, then generates a small
number of tokens. Validates that context tracking values are correct.
"""

from __future__ import annotations

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from contextwatch.inference_loop import run_inference


def run(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> list[str]:
    """Run long prompt validation. Returns a list of failure messages (empty = pass)."""
    failures: list[str] = []

    # Build a prompt that tokenizes to roughly 900 tokens — well within
    # distilgpt2's 1024-token window but large enough to stress context tracking.
    base_sentence = "The quick brown fox jumps over the lazy dog. "
    # Start with a small count and grow until we reach ~900 tokens
    repeat_count = 10
    while True:
        candidate = base_sentence * repeat_count
        token_ids = tokenizer.encode(candidate)
        if len(token_ids) >= 900:
            break
        repeat_count += 5

    # Trim to exactly ~900 tokens by decoding that many tokens
    target_tokens = 900
    token_ids = tokenizer.encode(base_sentence * repeat_count)
    if len(token_ids) > target_tokens:
        token_ids = token_ids[:target_tokens]
    long_prompt = tokenizer.decode(token_ids, skip_special_tokens=True)

    # Verify actual token count is in the ballpark
    actual_prompt_tokens = len(tokenizer.encode(long_prompt))
    if actual_prompt_tokens < 800:
        failures.append(
            f"Long prompt only produced {actual_prompt_tokens} tokens, "
            f"expected ~1000. Adjust repetitions."
        )
        # Still continue the test

    max_tokens = 10

    result = run_inference(
        model=model,
        tokenizer=tokenizer,
        prompt=long_prompt,
        max_tokens=max_tokens,
    )

    ctx = result.context_summary

    # 1. Context summary must exist
    if ctx is None:
        failures.append("Context summary is None")
        return failures

    # 2. final_total_tokens should match prompt + generated
    expected_total = result.prompt_token_count + result.generated_token_count
    if ctx.final_total_tokens != expected_total:
        failures.append(
            f"Context final_total_tokens={ctx.final_total_tokens}, "
            f"expected={expected_total}"
        )

    # 3. context_used_pct should be > 0 (a ~1000-token prompt in a 1024 window)
    if ctx.context_used_pct <= 0:
        failures.append(f"context_used_pct should be > 0, got {ctx.context_used_pct}")

    # 4. remaining_tokens should be correct
    expected_remaining = ctx.max_context - ctx.final_total_tokens
    if ctx.remaining_tokens != max(expected_remaining, 0):
        failures.append(
            f"remaining_tokens={ctx.remaining_tokens}, "
            f"expected={max(expected_remaining, 0)}"
        )

    # 5. Snapshot count should match generated token count
    if len(ctx.per_step_snapshots) != result.generated_token_count:
        failures.append(
            f"Snapshot count={len(ctx.per_step_snapshots)}, "
            f"expected={result.generated_token_count}"
        )

    return failures
