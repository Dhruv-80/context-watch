"""vLLM adapter — streams tokens from an OpenAI-compatible vLLM server.

Connects to a running vLLM instance (or any OpenAI-compatible endpoint),
streams generated tokens, and feeds per-token timing into the existing
ContextWatch monitoring modules.

Because vLLM manages GPU memory internally and does not expose KV-cache
sizes through the API, **memory tracking is unavailable** in this mode.
Context and latency tracking work normally.

Usage::

    # Start vLLM server first:
    #   python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-v0.1

    from contextwatch.core.vllm_adapter import run_vllm

    result = run_vllm(
        endpoint="http://localhost:8000",
        model="mistralai/Mistral-7B-v0.1",
        prompt="Explain transformers",
        max_tokens=200,
    )

.. note::
   This module deliberately avoids top-level imports of ``torch``,
   ``transformers``, and ``openai`` so that it can be imported without
   those heavy packages being loaded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache


# ---------------------------------------------------------------------------
# vLLM-specific result metadata
# ---------------------------------------------------------------------------
@dataclass
class VllmMetadata:
    """Extra metadata specific to vLLM runs."""

    endpoint: str
    memory_note: str = "Memory tracking unavailable — vLLM manages GPU memory internally."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_vllm(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int = 50,
    max_context: int | None = None,
    warn_threshold: float = 0.75,
):
    """Run streaming inference against a vLLM OpenAI-compatible server.

    Args:
        endpoint: Base URL of the vLLM server (e.g. ``"http://localhost:8000"``).
        model: Model name as registered on the vLLM server.
        prompt: Text prompt to send.
        max_tokens: Maximum tokens to generate.
        max_context: Model context window size. If ``None``, the adapter
            attempts to query ``/v1/models`` for this value; falls back
            to 4096 if unavailable.
        warn_threshold: Context usage fraction (0–1) at which to warn.

    Returns:
        An :class:`~contextwatch.inference_loop.InferenceResult` with token
        counts, text, and latency monitoring summary.  Memory summary will
        be ``None``.

    Raises:
        ImportError: If the ``openai`` package is not installed.
        ConnectionError: If the vLLM server is unreachable.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "The 'openai' package is required for vLLM mode. "
            "Install it with:  pip install contextwatch[vllm]"
        ) from None

    from contextwatch.inference_loop import InferenceResult
    from contextwatch.monitor.context_tracker import ContextSummary
    from contextwatch.monitor.latency_tracker import LatencyTracker

    client = OpenAI(
        base_url=f"{endpoint.rstrip('/')}/v1",
        api_key="EMPTY",  # vLLM doesn't require a real key
    )

    # --- resolve max context -----------------------------------------------
    if max_context is None:
        max_context = _query_max_context(client, model)

    # --- latency tracker ---------------------------------------------------
    latency_tracker = LatencyTracker(rolling_window=20)
    latency_tracker.start()

    # --- stream tokens -----------------------------------------------------
    generated_chunks: list[str] = []
    prompt_token_count: int = 0
    generated_token_count: int = 0

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
    except Exception as exc:
        _raise_connection_error(endpoint, exc)

    step = 0
    last_token_time = time.perf_counter()
    try:
        for chunk in stream:
            now = time.perf_counter()

            # Usage info arrives in the final chunk
            if chunk.usage is not None:
                prompt_token_count = chunk.usage.prompt_tokens
                generated_token_count = chunk.usage.completion_tokens

            # Content delta
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    generated_chunks.append(delta.content)
                    latency_tracker.record_step(step, last_token_time, now)
                    last_token_time = now
                    step += 1
    except Exception as exc:
        _raise_connection_error(endpoint, exc)

    generated_text = "".join(generated_chunks)

    # If the server didn't report usage, estimate with the model tokenizer.
    if generated_token_count == 0 and generated_text:
        generated_token_count = _estimate_token_count(model, generated_text) or step
    if prompt_token_count == 0:
        prompt_token_count = _estimate_token_count(model, prompt) or 0

    total_token_count = prompt_token_count + generated_token_count

    # --- context summary (manual construction) -----------------------------
    context_used_pct = (
        total_token_count / max_context if max_context > 0 else 0.0
    )
    remaining = max(max_context - total_token_count, 0)
    warning_issued = context_used_pct >= warn_threshold

    ctx_summary = ContextSummary(
        max_context=max_context,
        final_total_tokens=total_token_count,
        context_used_pct=round(context_used_pct, 6),
        remaining_tokens=remaining,
        per_step_snapshots=[],  # no per-step snapshots in streaming mode
        warning_issued=warning_issued,
    )

    return InferenceResult(
        prompt_token_count=prompt_token_count,
        generated_token_count=generated_token_count,
        total_token_count=total_token_count,
        generated_text=generated_text,
        context_summary=ctx_summary,
        latency_summary=latency_tracker.summarize(),
        memory_summary=None,  # vLLM manages memory server-side
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _raise_connection_error(endpoint: str, exc: Exception) -> None:
    """Wrap common connection failures with an actionable message."""
    msg = str(exc)
    if "Connection" in type(exc).__name__ or "connect" in msg.lower():
        raise ConnectionError(
            f"Could not connect to vLLM server at {endpoint}. "
            f"Please verify that:\n"
            f"  1. The vLLM server is running (python -m vllm.entrypoints.openai.api_server ...)\n"
            f"  2. The endpoint URL is correct\n"
            f"  3. No firewall is blocking the port\n"
            f"Original error: {exc}"
        ) from exc
    raise


def _query_max_context(client, model: str) -> int:
    """Try to fetch max_model_len from the vLLM ``/v1/models`` endpoint.

    Falls back to 4096 if the query fails.
    """
    try:
        models = client.models.list()
        for m in models.data:
            if m.id == model:
                max_len = getattr(m, "max_model_len", None)
                if max_len is not None:
                    return int(max_len)
        return 4096
    except Exception:
        return 4096


def _estimate_token_count(model: str, text: str) -> int | None:
    """Best-effort token count estimate using a local HF tokenizer."""
    try:
        tokenizer = _get_hf_tokenizer(model)
        return len(tokenizer.encode(text))
    except Exception:
        return None


@lru_cache(maxsize=8)
def _get_hf_tokenizer(model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)
