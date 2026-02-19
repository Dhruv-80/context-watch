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
Generated tokens: 50
Total tokens: 51

Context: 5.0% (51/1024)
Remaining tokens: 973

Latency Metrics:
  TTFT: 541.1 ms
  Current token latency: 5.6 ms
  Rolling avg (last 20): 5.5 ms
  Trend: -126.1 ms per 100 tokens

Memory Metrics:
  Current memory: 662.05 MB
  Peak memory: 662.05 MB
  Total growth: +312.70 MB
  Growth rate: +625.41 MB per 100 tokens
  Avg per token: 6.2541 MB
```

## Features (v0.4.0)

- Manual stepwise inference — no `model.generate()`
- KV-cache management with `past_key_values`
- EOS-token early stopping
- Token accounting: prompt / generated / total
- **Context window tracking** — max context detection, % used, remaining tokens
- **Configurable warning** — alerts when context usage crosses a threshold (`--warn-threshold`)
- **Context exhaustion early stopping** — generation halts when the context window is full
- **Latency tracking** — Time To First Token (TTFT), per-token latency, rolling average
- **Latency trend analysis** — Linear regression slope showing latency growth (ms per 100 tokens)
- **Memory tracking** — Process RSS monitoring via `psutil` (current, peak, growth)
- **KV cache growth rate** — Average memory growth per token and per 100 tokens

## Versioning

We use simple semantic-style versioning tied to project phases:

| Version | Phase | Description |
|---------|-------|-------------|
| 0.1.0   | 1     | Manual inference loop + token counting |
| 0.2.0   | 2     | Context window tracking + warnings |
| 0.3.0   | 3     | Latency tracking (TTFT, rolling avg, trend) |
| 0.4.0   | 4     | Memory & KV cache growth tracking |

The version is defined in two places — keep them in sync:
- `contextwatch/__init__.py` → `__version__`
- `pyproject.toml` → `[project] version`

All changes are tracked in [CHANGELOG.md](CHANGELOG.md).

## License

MIT
