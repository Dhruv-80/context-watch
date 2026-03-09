# ContextWatch

**Real-time monitoring for LLM inference — track context window usage, memory growth, and per-token latency before they become problems.**

---

## Why ContextWatch?

Large language models don't fail gracefully. As you generate tokens:

- **Context windows fill up silently.** When they saturate, the model's output degrades or generation halts — often with no warning.
- **Memory grows unpredictably.** KV-cache allocations scale with sequence length, and a single long generation can OOM your process.
- **Latency creeps upward.** Each token takes slightly longer as the attention matrix grows. A prompt that starts at 5ms/token may end at 50ms/token.

ContextWatch gives you **live visibility** into these metrics during inference, with **forecasts** that tell you when limits will be hit — before they are.

It runs a manual, stepwise inference loop (no `model.generate()`) so every token is accounted for.

---

## Example CLI Output

```
Loading model 'distilgpt2' on cpu ...

Prompt tokens: 4
Generated tokens: 200
Total tokens: 204

Context: 19.9% (204/1024)
Remaining tokens: 820

Latency Metrics:
  TTFT: 312.4 ms
  Current token latency: 6.1 ms
  Rolling avg (last 20): 5.8 ms
  Trend: +0.3 ms per 100 tokens

Memory Metrics:
  Current memory: 4.20 GB
  Peak memory: 4.20 GB
  Total growth: +312.70 MB
  Growth rate: +156.35 MB per 100 tokens
  Avg per token: 1.5635 MB

Forecast:
  Context saturation in: ~820 tokens
  Memory limit (8.00 GB) in: ~2483 tokens
  Latency >100ms: no degradation trend — threshold unlikely to be reached

Diagnosis:
  Risk score: 36/100 (watch)
  [MEDIUM] context: Context usage is above 75%.
  Recommendations:
    - Reserve headroom: cap generation or trim prompt to avoid hard stops.

Run log saved to: runs/run_2026_03_08_154500.json
```

---

## Installation

```bash
git clone https://github.com/yourusername/contextwatch.git
cd contextwatch
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

For rich terminal formatting (optional):

```bash
pip install -e ".[rich]"
```

---

## Usage

### Run inference with monitoring

```bash
contextwatch run --model distilgpt2 --prompt "Explain transformers" --max-tokens 200
```

With resource limit forecasting:

```bash
contextwatch run \
  --model distilgpt2 \
  --prompt "Explain transformers" \
  --max-tokens 200 \
  --memory-limit 8GB \
  --latency-limit 100ms
```

### Analyze a run log

Every `run` command saves a JSON log to the `runs/` directory. Generate plots from a log:

```bash
contextwatch analyze runs/run_2026_03_08_154500.json
```

This produces three PNG plots in the same directory:
- `latency.png` — per-token latency over generation steps
- `memory.png` — process RSS memory over generation steps
- `context.png` — context window utilisation (%) with warning thresholds

### Generate a one-page brief

Create a shareable markdown summary with risk score, bottleneck attribution, and prioritized actions:

```bash
contextwatch report runs/run_2026_03_08_154500.json
```

Optional output path:

```bash
contextwatch report runs/run_2026_03_08_154500.json --output reports/perf_brief.md
```

### CLI help

```bash
contextwatch --help
contextwatch run --help
contextwatch analyze --help
contextwatch report --help
```

---

## Architecture Overview

```
contextwatch/
├── cli.py                  # CLI entry point (argparse, run/analyze commands)
├── inference_loop.py       # Manual stepwise inference (no model.generate)
├── utils.py                # Model loading, device selection
├── analyzer.py             # JSON log loading, matplotlib plot generation
├── reporter.py             # One-page markdown incident/perf brief generation
└── monitor/
    ├── advisor.py          # Risk score + findings + action recommendations
    ├── context_tracker.py  # Context window tracking (% used, remaining)
    ├── latency_tracker.py  # Per-token latency, TTFT, rolling avg, trend
    ├── memory_tracker.py   # Process RSS tracking via psutil
    └── forecaster.py       # Linear extrapolation forecasts
```

### Inference Loop

The core of ContextWatch is a **manual token-by-token generation loop** in `inference_loop.py`. Instead of calling `model.generate()`, it:

1. Tokenizes the prompt
2. Runs the first forward pass to get initial KV-cache (`past_key_values`)
3. Iterates step-by-step: extracts next token via `argmax`, feeds it back with the cached KV state
4. Stops on EOS token or context window exhaustion

This gives full control over every token, enabling per-step metric collection.

### Monitoring Modules

Each tracker follows the same pattern: `start()` → `record_step()` per token → `summarize()` at the end.

| Module | Tracks | Key Metrics |
|--------|--------|-------------|
| `context_tracker` | Context window fill level | % used, remaining tokens, configurable warning |
| `latency_tracker` | Per-token generation time | TTFT, current latency, rolling avg, linear trend |
| `memory_tracker` | Process RSS via `psutil` | Current/peak memory, total growth, growth rate |

### Forecasting Logic

`forecaster.py` uses **simple linear extrapolation** (no ML) to predict:

- **Context saturation**: `remaining = max_context - total_tokens` (exact)
- **Memory limit**: `tokens = (limit - current) / avg_growth_per_token`
- **Latency threshold**: `tokens = (threshold - current) / slope_per_token`

### Actionable Diagnosis

Beyond raw metrics, ContextWatch produces a deterministic **diagnosis**:
- `risk_score` (0–100) and run `status` (`stable`, `watch`, `elevated`, `critical`)
- Structured findings by area (`context`, `memory`, `latency`)
- Human-readable recommendations for remediation

Diagnosis is shown in the CLI and Streamlit UI, and stored in the run log JSON for downstream automation.

---

## Limitations

ContextWatch is a lightweight monitoring tool, not a production profiler. Keep these in mind:

- **Memory estimates are approximate.** RSS measures total process memory, not just the model. Shared libraries, Python objects, and OS caching all contribute.
- **KV cache is tracked indirectly.** We measure memory growth per token rather than reading KV-cache tensors directly. The first few tokens show a burst (initial allocation), making short-run forecasts conservative.
- **Forecasting assumes stable generation rate.** The linear extrapolation works well for long runs but can be noisy for short sequences. Real latency often has non-linear characteristics at high context utilisation.
- **CPU-only latency patterns differ from GPU.** On GPU, latency is dominated by memory bandwidth; on CPU, by compute. Trends may look different across hardware.
- **Greedy decoding only.** The inference loop uses `argmax` (greedy) token selection. Sampling strategies are not currently supported.

---

## Running Tests

ContextWatch includes a validation test suite:

```bash
python run_tests.py
```

This loads `distilgpt2` once and runs four tests:

| Test | Validates |
|------|-----------|
| Small prompt (50 tokens) | Token counts match expectations |
| Long prompt (~1000 tokens) | Context tracking values are correct |
| Latency tracking | Per-step latency captured, TTFT, rolling average |
| Memory tracking | RSS captured each step, peak ≥ current |

Output:

```
============================================================
  ContextWatch — Validation Test Suite
============================================================

Loading model (distilgpt2) ...

  Running: Small prompt generation (50 tokens) ... PASS
  Running: Long prompt context tracking (~1000 tokens) ... PASS
  Running: Latency tracking ... PASS
  Running: Memory tracking ... PASS

============================================================
  Results: 4 passed, 0 failed, 4 total
============================================================

✅ All tests passed.
```

---

## Versioning

| Version | Phase | Description |
|---------|-------|-------------|
| 0.1.0   | 1     | Manual inference loop + token counting |
| 0.2.0   | 2     | Context window tracking + warnings |
| 0.3.0   | 3     | Latency tracking (TTFT, rolling avg, trend) |
| 0.4.0   | 4     | Memory & KV cache growth tracking |
| 0.5.0   | 5     | Forecasting engine (context, memory, latency) |
| 0.6.0   | 6–8   | Test suite, CLI packaging, documentation |

---

## License

MIT
