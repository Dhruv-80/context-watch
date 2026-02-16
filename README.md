# ContextWatch

A Python CLI tool for controlled LLM inference with token accounting.

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

contextwatch run --model distilgpt2 --prompt "Hello" --max-tokens 50
```

Output:

```
Prompt tokens: 1
Generated tokens: 20
Total tokens: 21

Context: 2.1% (21/1024)
Remaining tokens: 1003

Latency Metrics:
  TTFT: 544.4 ms
  Current token latency: 7.0 ms
  Rolling avg (last 20): 33.9 ms
  Trend: -770.3 ms per 100 tokens
```

## Features (v0.3.0)

- Manual stepwise inference — no `model.generate()`
- KV-cache management with `past_key_values`
- EOS-token early stopping
- Token accounting: prompt / generated / total
- **Context window tracking** — max context detection, % used, remaining tokens
- **Configurable warning** — alerts when context usage crosses a threshold (`--warn-threshold`)
- **Context exhaustion early stopping** — generation halts when the context window is full
- **Latency tracking** — Time To First Token (TTFT), per-token latency, rolling average
- **Latency trend analysis** — Linear regression slope showing latency growth (ms per 100 tokens)

## Versioning

We use simple semantic-style versioning tied to project phases:

| Version | Phase | Description |
|---------|-------|-------------|
| 0.1.0   | 1     | Manual inference loop + token counting |
| 0.2.0   | 2     | Context window tracking + warnings |
| 0.3.0   | 3     | Latency tracking (TTFT, rolling avg, trend) |

The version is defined in two places — keep them in sync:
- `contextwatch/__init__.py` → `__version__`
- `pyproject.toml` → `[project] version`

All changes are tracked in [CHANGELOG.md](CHANGELOG.md).

## License

MIT
