"""Unit tests for markdown report generation."""

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

from contextwatch.reporter import default_report_path, generate_report_markdown


class ReporterUnitTests(unittest.TestCase):
    def test_report_uses_existing_diagnosis_and_forecast(self) -> None:
        data = {
            "timestamp": "2026-03-09T12:00:00",
            "mode": "hf",
            "model": "distilgpt2",
            "generated_token_count": 50,
            "total_token_count": 120,
            "context_summary": {
                "context_used_pct": 0.8,
                "final_total_tokens": 120,
                "max_context": 1024,
                "remaining_tokens": 904,
            },
            "latency_summary": {
                "ttft_ms": 150.0,
                "current_token_latency_ms": 8.2,
                "trend_ms_per_100_tokens": 1.7,
            },
            "memory_summary": {
                "current_memory_mb": 980.0,
                "peak_memory_mb": 1005.0,
            },
            "forecast": {
                "tokens_until_context_limit": 904,
                "context_already_saturated": False,
                "tokens_until_memory_limit": 3200,
                "memory_already_exceeded": False,
                "tokens_until_latency_threshold": 2500,
                "latency_already_exceeded": False,
            },
            "diagnosis": {
                "risk_score": 31,
                "status": "watch",
                "findings": [
                    {
                        "area": "context",
                        "severity": "medium",
                        "message": "Context usage is above 75%.",
                    }
                ],
                "recommendations": ["Reserve headroom: cap generation."],
            },
        }
        md = generate_report_markdown(data, source_path="runs/sample.json")

        self.assertIn("# ContextWatch Performance Brief", md)
        self.assertIn("risk_score=31/100", md)
        self.assertIn("Primary bottleneck: **context**", md)
        self.assertIn("Reserve headroom: cap generation.", md)

    def test_report_fallback_for_legacy_log(self) -> None:
        legacy = {
            "timestamp": "2026-03-09T12:00:00",
            "mode": "vllm",
            "model": "mistral",
            "generated_token_count": 20,
            "total_token_count": 800,
            "context_summary": {
                "context_used_pct": 0.95,
                "final_total_tokens": 800,
                "max_context": 840,
                "remaining_tokens": 40,
            },
            "latency_summary": {
                "ttft_ms": 120.0,
                "current_token_latency_ms": 12.0,
                "trend_ms_per_100_tokens": 6.0,
            },
        }
        md = generate_report_markdown(legacy, source_path="runs/legacy.json")

        self.assertIn("## Executive Summary", md)
        self.assertIn("Primary bottleneck:", md)
        self.assertIn("Track GPU memory externally", md)

    def test_default_report_path(self) -> None:
        path = default_report_path("/tmp/run_2026_03_09_120000.json")
        self.assertEqual(path, "/tmp/run_2026_03_09_120000_brief.md")


if __name__ == "__main__":
    unittest.main()
