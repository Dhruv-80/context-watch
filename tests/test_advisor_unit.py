"""Unit tests for diagnosis advisor logic."""

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

from contextwatch.monitor.advisor import build_diagnosis
from contextwatch.monitor.forecaster import ForecastResult


class AdvisorUnitTests(unittest.TestCase):
    def test_stable_case_returns_low_risk(self) -> None:
        forecast = ForecastResult(tokens_until_context_limit=2000)
        diagnosis = build_diagnosis(
            context_used_pct=0.20,
            latency_trend_ms_per_100_tokens=0.2,
            forecast=forecast,
            mode="hf",
        )

        self.assertLess(diagnosis.risk_score, 20)
        self.assertEqual(diagnosis.status, "stable")
        self.assertGreaterEqual(len(diagnosis.recommendations), 1)

    def test_critical_case_flags_multiple_risks(self) -> None:
        forecast = ForecastResult(
            tokens_until_context_limit=0,
            context_already_saturated=True,
            tokens_until_memory_limit=0,
            memory_already_exceeded=True,
            tokens_until_latency_threshold=0,
            latency_already_exceeded=True,
        )
        diagnosis = build_diagnosis(
            context_used_pct=1.0,
            latency_trend_ms_per_100_tokens=8.0,
            forecast=forecast,
            mode="hf",
        )

        self.assertGreaterEqual(diagnosis.risk_score, 70)
        self.assertEqual(diagnosis.status, "critical")
        self.assertTrue(any(f.area == "context" for f in diagnosis.findings))
        self.assertTrue(any(f.area == "memory" for f in diagnosis.findings))
        self.assertTrue(any(f.area == "latency" for f in diagnosis.findings))

    def test_vllm_adds_memory_info_finding(self) -> None:
        forecast = ForecastResult(tokens_until_context_limit=1000)
        diagnosis = build_diagnosis(
            context_used_pct=0.5,
            latency_trend_ms_per_100_tokens=None,
            forecast=forecast,
            mode="vllm",
        )

        self.assertTrue(
            any(
                f.area == "memory" and f.severity == "info"
                for f in diagnosis.findings
            )
        )
        self.assertTrue(any("GPU memory" in r for r in diagnosis.recommendations))


if __name__ == "__main__":
    unittest.main()
