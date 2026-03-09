"""Unit tests for context tracker edge cases."""

from __future__ import annotations

import sys
import types
import unittest

try:
    import transformers  # noqa: F401
except ModuleNotFoundError:
    fake_transformers = types.ModuleType("transformers")

    class _PreTrainedModel:  # pragma: no cover - test shim
        pass

    fake_transformers.PreTrainedModel = _PreTrainedModel
    sys.modules["transformers"] = fake_transformers

try:
    import psutil  # noqa: F401
except ModuleNotFoundError:
    fake_psutil = types.ModuleType("psutil")

    class _FakeMemInfo:  # pragma: no cover - test shim
        rss = 0

    class _FakeProcess:  # pragma: no cover - test shim
        def __init__(self, _pid):
            pass

        def memory_info(self):
            return _FakeMemInfo()

    fake_psutil.Process = _FakeProcess
    sys.modules["psutil"] = fake_psutil

from contextwatch.monitor.context_tracker import ContextTracker


class _DummyConfig:
    max_position_embeddings = 1024

    def to_dict(self):
        return {"max_position_embeddings": self.max_position_embeddings}


class _DummyModel:
    config = _DummyConfig()


class ContextTrackerUnitTests(unittest.TestCase):
    def test_summarize_without_snapshots_uses_final_total_tokens(self) -> None:
        tracker = ContextTracker(_DummyModel(), warn_threshold=0.75)

        summary = tracker.summarize(final_total_tokens=512)

        self.assertEqual(summary.max_context, 1024)
        self.assertEqual(summary.final_total_tokens, 512)
        self.assertEqual(summary.context_used_pct, 0.5)
        self.assertEqual(summary.remaining_tokens, 512)
        self.assertEqual(len(summary.per_step_snapshots), 0)


if __name__ == "__main__":
    unittest.main()
