"""Core adapter layer — HF and vLLM inference backends.

Imports are lazy so that ``import contextwatch.core`` does not pull in
``torch`` / ``transformers`` (needed only by the HF adapter) or ``openai``
(needed only by the vLLM adapter).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextwatch.core.hf_adapter import run_hf as run_hf
    from contextwatch.core.vllm_adapter import run_vllm as run_vllm

__all__ = ["run_hf", "run_vllm"]


def __getattr__(name: str):
    if name == "run_hf":
        from contextwatch.core.hf_adapter import run_hf

        return run_hf
    if name == "run_vllm":
        from contextwatch.core.vllm_adapter import run_vllm

        return run_vllm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
