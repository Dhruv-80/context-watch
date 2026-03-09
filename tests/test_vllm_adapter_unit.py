"""Unit tests for vLLM adapter fallback behavior."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch


class _FakeDelta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content: str | None = None, usage=None) -> None:
        self.usage = usage
        self.choices = [_FakeChoice(content)] if content is not None else []


class _FakeCompletions:
    def create(self, **kwargs):
        # No usage chunk returned: exercises fallback path.
        return iter(
            [
                _FakeChunk("Hello"),
                _FakeChunk(" world"),
            ]
        )


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeOpenAIClient:
    def __init__(self, *args, **kwargs) -> None:
        self.chat = _FakeChat()


class VllmAdapterUnitTests(unittest.TestCase):
    def test_usage_missing_falls_back_to_tokenizer_estimate(self) -> None:
        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = _FakeOpenAIClient
        fake_torch = types.ModuleType("torch")
        fake_transformers = types.ModuleType("transformers")
        fake_psutil = types.ModuleType("psutil")

        class _PreTrainedModel:  # pragma: no cover - test shim
            pass

        class _PreTrainedTokenizerBase:  # pragma: no cover - test shim
            pass

        fake_transformers.PreTrainedModel = _PreTrainedModel
        fake_transformers.PreTrainedTokenizerBase = _PreTrainedTokenizerBase

        class _FakeMemInfo:  # pragma: no cover - test shim
            rss = 0

        class _FakeProcess:  # pragma: no cover - test shim
            def __init__(self, _pid):
                pass

            def memory_info(self):
                return _FakeMemInfo()

        fake_psutil.Process = _FakeProcess

        with patch.dict(
            sys.modules,
            {
                "openai": fake_openai,
                "torch": fake_torch,
                "transformers": fake_transformers,
                "psutil": fake_psutil,
            },
        ):
            from contextwatch.core.vllm_adapter import run_vllm

            with patch(
                "contextwatch.core.vllm_adapter._estimate_token_count",
                side_effect=[5, 2],  # generated text first, then prompt
            ):
                result = run_vllm(
                    endpoint="http://localhost:8000",
                    model="fake-model",
                    prompt="Hi",
                    max_tokens=10,
                    max_context=1024,
                )

        self.assertEqual(result.generated_text, "Hello world")
        self.assertEqual(result.generated_token_count, 5)
        self.assertEqual(result.prompt_token_count, 2)
        self.assertEqual(result.total_token_count, 7)
        self.assertIsNotNone(result.latency_summary)
        self.assertEqual(len(result.latency_summary.per_step_snapshots), 2)


if __name__ == "__main__":
    unittest.main()
