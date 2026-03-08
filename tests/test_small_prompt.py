"""Test 1: Small prompt generation.

Runs distilgpt2 with prompt "Hello" and generates 50 tokens.
Validates that token counts are consistent and within expected bounds.
"""

from __future__ import annotations

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from contextwatch.inference_loop import run_inference


def run(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> list[str]:
    """Run small prompt validation. Returns a list of failure messages (empty = pass)."""
    failures: list[str] = []

    prompt = "Hello"
    max_tokens = 50

    result = run_inference(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
    )

    # 1. Total must equal prompt + generated
    expected_total = result.prompt_token_count + result.generated_token_count
    if result.total_token_count != expected_total:
        failures.append(
            f"Token count mismatch: total={result.total_token_count}, "
            f"expected={expected_total} (prompt={result.prompt_token_count} "
            f"+ generated={result.generated_token_count})"
        )

    # 2. Generated must not exceed max_tokens
    if result.generated_token_count > max_tokens:
        failures.append(
            f"Generated {result.generated_token_count} tokens, "
            f"exceeds max_tokens={max_tokens}"
        )

    # 3. Prompt token count must be positive
    if result.prompt_token_count <= 0:
        failures.append(f"Prompt token count must be > 0, got {result.prompt_token_count}")

    # 4. Generated text must be non-empty (50 tokens should produce output)
    if not result.generated_text.strip():
        failures.append("Generated text is empty")

    return failures
