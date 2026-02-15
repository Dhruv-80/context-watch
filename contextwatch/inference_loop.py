"""Manual stepwise inference loop — no model.generate() allowed."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from contextwatch.monitor.context_tracker import ContextSummary, ContextTracker


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class InferenceResult:
    """Holds the output of a single inference run."""

    prompt_token_count: int
    generated_token_count: int
    total_token_count: int
    generated_text: str
    context_summary: ContextSummary | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Prompt tokenisation (Phase 0)
# ---------------------------------------------------------------------------
def tokenize_prompt(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    device: torch.device | None = None,
) -> dict[str, object]:
    """Tokenize *prompt* and return input IDs with the prompt token count.

    Args:
        tokenizer: A Hugging Face tokenizer.
        prompt: The raw text prompt.
        device: Device to place tensors on.  Defaults to CPU.

    Returns:
        A dict with keys ``"input_ids"`` (``torch.Tensor``) and
        ``"prompt_token_count"`` (``int``).
    """
    if device is None:
        device = torch.device("cpu")

    encoding = tokenizer(prompt, return_tensors="pt")
    input_ids: torch.Tensor = encoding["input_ids"].to(device)

    return {
        "input_ids": input_ids,
        "prompt_token_count": input_ids.shape[1],
    }


# ---------------------------------------------------------------------------
# Stepwise inference loop (Phase 1)
# ---------------------------------------------------------------------------
def run_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_tokens: int,
    warn_threshold: float = 0.75,
) -> InferenceResult:
    """Run manual stepwise token generation.

    This function **never** calls ``model.generate()``.  Instead it:

    1. Tokenizes the prompt.
    2. Feeds the full prompt through the model to obtain initial
       ``past_key_values``.
    3. Iterates for up to *max_tokens* steps, extracting the next token
       via ``argmax`` on the final logits and feeding it back with the
       cached ``past_key_values``.
    4. Stops early if the EOS token is produced.
    5. Stops early if the context window is full.

    Args:
        model: A Hugging Face causal language model.
        tokenizer: The corresponding tokenizer.
        prompt: The text prompt.
        max_tokens: Maximum number of new tokens to generate.
        warn_threshold: Context usage fraction (0–1) at which to warn.

    Returns:
        An :class:`InferenceResult` with token counts and generated text.
    """
    device = next(model.parameters()).device

    # --- tokenize prompt ---------------------------------------------------
    prompt_data = tokenize_prompt(tokenizer, prompt, device=device)
    current_input_ids: torch.Tensor = prompt_data["input_ids"]
    prompt_token_count: int = prompt_data["prompt_token_count"]

    # --- storage for generated token ids -----------------------------------
    generated_ids: list[int] = []

    # --- manual inference loop ---------------------------------------------
    past_key_values = None
    eos_token_id: int | None = tokenizer.eos_token_id

    # --- context tracking (Phase 2) ---------------------------------------
    tracker = ContextTracker(model, warn_threshold=warn_threshold)

    with torch.no_grad():
        for step in range(max_tokens):
            outputs = model(
                input_ids=current_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs.logits                    # (batch, seq_len, vocab)
            past_key_values = outputs.past_key_values  # KV-cache for next step

            # greedy: pick the highest-probability token
            next_token_id: int = int(torch.argmax(logits[:, -1, :], dim=-1).item())

            # EOS early stop
            if next_token_id == eos_token_id:
                break

            generated_ids.append(next_token_id)

            # --- per-step context tracking (Phase 2) ----------------------
            current_total = prompt_token_count + len(generated_ids)
            tracker.record_step(step, current_total)

            if tracker.is_context_full(current_total):
                break

            # prepare input for the next step (single token, reuse cache)
            current_input_ids = torch.tensor(
                [[next_token_id]], dtype=torch.long, device=device
            )

    generated_token_count = len(generated_ids)
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return InferenceResult(
        prompt_token_count=prompt_token_count,
        generated_token_count=generated_token_count,
        total_token_count=prompt_token_count + generated_token_count,
        generated_text=generated_text,
        context_summary=tracker.summarize(),
    )
