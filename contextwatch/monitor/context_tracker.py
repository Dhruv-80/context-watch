"""Context window tracking for LLM inference runs."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from transformers import PreTrainedModel


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class ContextSnapshot:
    """A single per-step snapshot of context window usage."""

    step: int
    total_tokens: int
    max_context: int
    context_used_pct: float          # 0.0 – 1.0
    remaining_tokens: int


@dataclass
class ContextSummary:
    """Final summary produced after an inference run completes."""

    max_context: int
    final_total_tokens: int
    context_used_pct: float          # 0.0 – 1.0
    remaining_tokens: int
    per_step_snapshots: list[ContextSnapshot] = field(default_factory=list)
    warning_issued: bool = False


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
class ContextTracker:
    """Tracks context window usage during stepwise inference.

    Args:
        model: A HuggingFace causal language model (used to read
            ``model.config`` for the maximum context length).
        warn_threshold: Fraction of context usage (0–1) at which a
            warning is printed to *stderr*.  Defaults to ``0.75``.
    """

    def __init__(self, model: PreTrainedModel, warn_threshold: float = 0.75) -> None:
        self._max_context = self._get_max_context(model)
        self._warn_threshold = warn_threshold
        self._snapshots: list[ContextSnapshot] = []
        self._warning_issued = False

    # -- public API ---------------------------------------------------------

    @property
    def max_context(self) -> int:
        """Maximum context length reported by the model config."""
        return self._max_context

    def record_step(self, step: int, total_tokens: int) -> ContextSnapshot:
        """Record token usage after generating a new token.

        Args:
            step: Zero-based generation step index.
            total_tokens: Prompt tokens + generated tokens so far.

        Returns:
            A :class:`ContextSnapshot` for this step.
        """
        used_pct = total_tokens / self._max_context if self._max_context > 0 else 0.0
        remaining = max(self._max_context - total_tokens, 0)

        snap = ContextSnapshot(
            step=step,
            total_tokens=total_tokens,
            max_context=self._max_context,
            context_used_pct=round(used_pct, 6),
            remaining_tokens=remaining,
        )
        self._snapshots.append(snap)

        # Emit a one-time warning when the threshold is crossed.
        if not self._warning_issued and used_pct >= self._warn_threshold:
            pct_display = round(used_pct * 100, 1)
            print(
                f"\n⚠  Context usage reached {pct_display}% "
                f"({total_tokens}/{self._max_context}) — threshold is "
                f"{round(self._warn_threshold * 100, 1)}%",
                file=sys.stderr,
            )
            self._warning_issued = True

        return snap

    def is_context_full(self, total_tokens: int) -> bool:
        """Return ``True`` if *total_tokens* has reached the context limit."""
        return total_tokens >= self._max_context

    def summarize(self) -> ContextSummary:
        """Build a :class:`ContextSummary` from the recorded snapshots."""
        if self._snapshots:
            last = self._snapshots[-1]
            return ContextSummary(
                max_context=self._max_context,
                final_total_tokens=last.total_tokens,
                context_used_pct=last.context_used_pct,
                remaining_tokens=last.remaining_tokens,
                per_step_snapshots=list(self._snapshots),
                warning_issued=self._warning_issued,
            )
        # Edge case: no tokens were generated (e.g. prompt already at limit).
        return ContextSummary(
            max_context=self._max_context,
            final_total_tokens=0,
            context_used_pct=0.0,
            remaining_tokens=self._max_context,
            per_step_snapshots=[],
            warning_issued=self._warning_issued,
        )

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _get_max_context(model: PreTrainedModel) -> int:
        """Extract the maximum context length from the model config.

        HuggingFace models store this under different attribute names:

        * ``max_position_embeddings`` (most modern models)
        * ``n_positions`` (GPT-2 family)
        * ``n_ctx`` (some older GPT-2 checkpoints)

        Raises:
            ValueError: If none of the known attributes are found.
        """
        config = model.config
        for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
            value = getattr(config, attr, None)
            if value is not None:
                return int(value)
        raise ValueError(
            f"Cannot determine max context length from model config. "
            f"Checked: max_position_embeddings, n_positions, n_ctx. "
            f"Available config keys: {list(config.to_dict().keys())}"
        )
