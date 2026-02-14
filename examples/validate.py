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

    # --- basic assertions --------------------------------------------------
    assert result.total_token_count == result.prompt_token_count + result.generated_token_count, (
        f"Token count mismatch: {result.total_token_count} != "
        f"{result.prompt_token_count} + {result.generated_token_count}"
    )
    assert result.generated_token_count <= max_tokens, (
        f"Generated more tokens than allowed: {result.generated_token_count} > {max_tokens}"
    )
    assert result.prompt_token_count > 0, "Prompt token count must be > 0"

    print("\n✅ All assertions passed.")


if __name__ == "__main__":
    main()
