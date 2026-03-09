"""Hugging Face adapter — wraps the existing stepwise inference loop.

This adapter loads a HF model locally and runs the manual token-by-token
inference loop defined in :mod:`contextwatch.inference_loop`.  It exists
so that the CLI can route between backends using a consistent interface.
"""

from __future__ import annotations

from contextwatch.inference_loop import InferenceResult, run_inference
from contextwatch.utils import load_model


def run_hf(
    model_name: str,
    prompt: str,
    max_tokens: int = 50,
    warn_threshold: float = 0.75,
) -> InferenceResult:
    """Run inference using a local Hugging Face model.

    Args:
        model_name: HF Hub model identifier (e.g. ``"distilgpt2"``).
        prompt: Text prompt to feed to the model.
        max_tokens: Maximum number of new tokens to generate.
        warn_threshold: Context usage fraction (0–1) at which to warn.

    Returns:
        An :class:`InferenceResult` with token counts, text, and
        monitoring summaries.
    """
    model, tokenizer = load_model(model_name)

    return run_inference(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        warn_threshold=warn_threshold,
    )
