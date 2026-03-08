"""CLI entry point for ContextWatch."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime

from contextwatch.inference_loop import run_inference
from contextwatch.monitor.forecaster import ForecastResult, compute_forecast
from contextwatch.utils import load_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_memory_limit(value: str) -> float:
    """Parse a human-readable memory limit string into MB.

    Accepts formats like ``8GB``, ``8 GB``, ``512MB``, ``512 mb``.
    """
    match = re.match(r"^\s*([\d.]+)\s*(GB|MB)\s*$", value.strip(), re.IGNORECASE)
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid memory format: '{value}'. Use e.g. '8GB' or '512MB'."
        )
    number = float(match.group(1))
    unit = match.group(2).upper()
    return number * 1024.0 if unit == "GB" else number


def _parse_latency_limit(value: str) -> float:
    """Parse a latency limit string into ms.

    Accepts formats like ``100ms``, ``100 ms``, or plain ``100``.
    """
    match = re.match(r"^\s*([\d.]+)\s*(ms)?\s*$", value.strip(), re.IGNORECASE)
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid latency format: '{value}'. Use e.g. '100ms' or '100'."
        )
    return float(match.group(1))


def _fmt_mb(val: float) -> str:
    """Format a value in MB — uses GB for values >= 1024 MB."""
    if val >= 1024.0:
        return f"{val / 1024:.2f} GB"
    return f"{val:.2f} MB"


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="contextwatch",
        description="ContextWatch — Controlled LLM inference with real-time monitoring.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- `run` subcommand --------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Run stepwise inference with token, context, latency, and memory monitoring.",
    )
    run_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face model name or path (e.g. 'distilgpt2').",
    )
    run_parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt to feed to the model.",
    )
    run_parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate (default: 50).",
    )
    run_parser.add_argument(
        "--warn-threshold",
        type=float,
        default=0.75,
        help="Context usage fraction (0–1) at which to emit a warning (default: 0.75).",
    )
    run_parser.add_argument(
        "--memory-limit",
        type=_parse_memory_limit,
        default=None,
        help="Memory ceiling for forecasting, e.g. '8GB' or '512MB'.",
    )
    run_parser.add_argument(
        "--latency-limit",
        type=_parse_latency_limit,
        default=None,
        help="Latency threshold in ms for forecasting, e.g. '100ms' or '100'.",
    )

    # --- `analyze` subcommand ----------------------------------------------
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Load a JSON run log and generate latency, memory, and context plots.",
    )
    analyze_parser.add_argument(
        "log_file",
        type=str,
        help="Path to a JSON run log file (e.g. 'runs/run_2024_01_01_120000.json').",
    )
    analyze_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same directory as the log file).",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point invoked by the ``contextwatch`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        _handle_run(args)
    elif args.command == "analyze":
        _handle_analyze(args)


# ---------------------------------------------------------------------------
# `run` handler
# ---------------------------------------------------------------------------
def _handle_run(args: argparse.Namespace) -> None:
    """Execute the ``run`` subcommand."""
    model, tokenizer = load_model(args.model)

    result = run_inference(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        warn_threshold=args.warn_threshold,
    )

    # --- token counts ------------------------------------------------------
    print(f"\nPrompt tokens: {result.prompt_token_count}")
    print(f"Generated tokens: {result.generated_token_count}")
    print(f"Total tokens: {result.total_token_count}")

    # --- context tracking (Phase 2) ----------------------------------------
    ctx = result.context_summary
    if ctx is not None:
        pct = round(ctx.context_used_pct * 100, 1)
        print(f"\nContext: {pct}% ({ctx.final_total_tokens}/{ctx.max_context})")
        print(f"Remaining tokens: {ctx.remaining_tokens}")
        if ctx.warning_issued:
            print("(!) Context warning was triggered during generation.")

    # --- latency tracking (Phase 3) ----------------------------------------
    lat = result.latency_summary
    if lat is not None:
        print("\nLatency Metrics:")
        if lat.ttft_ms is not None:
            print(f"  TTFT: {lat.ttft_ms:.1f} ms")
        if lat.current_token_latency_ms is not None:
            print(f"  Current token latency: {lat.current_token_latency_ms:.1f} ms")
        if lat.rolling_avg_ms is not None:
            print(f"  Rolling avg (last 20): {lat.rolling_avg_ms:.1f} ms")
        if lat.trend_ms_per_100_tokens is not None:
            sign = "+" if lat.trend_ms_per_100_tokens >= 0 else ""
            print(f"  Trend: {sign}{lat.trend_ms_per_100_tokens:.1f} ms per 100 tokens")

    # --- memory tracking (Phase 4) -----------------------------------------
    mem = result.memory_summary
    if mem is not None:
        print("\nMemory Metrics:")
        print(f"  Current memory: {_fmt_mb(mem.current_memory_mb)}")
        print(f"  Peak memory: {_fmt_mb(mem.peak_memory_mb)}")
        sign = "+" if mem.memory_growth_total_mb >= 0 else ""
        print(f"  Total growth: {sign}{mem.memory_growth_total_mb:.2f} MB")
        sign = "+" if mem.growth_per_100_tokens_mb >= 0 else ""
        print(f"  Growth rate: {sign}{mem.growth_per_100_tokens_mb:.2f} MB per 100 tokens")
        print(f"  Avg per token: {mem.avg_growth_per_token_mb:.4f} MB")

    # --- forecasting (Phase 5) ---------------------------------------------
    latency_slope_per_token: float | None = None
    if lat is not None and lat.trend_ms_per_100_tokens is not None:
        latency_slope_per_token = lat.trend_ms_per_100_tokens / 100.0

    if ctx is not None:
        forecast = compute_forecast(
            total_tokens=result.total_token_count,
            max_context=ctx.max_context,
            current_memory_mb=mem.current_memory_mb if mem else None,
            avg_growth_per_token_mb=mem.avg_growth_per_token_mb if mem else None,
            memory_limit_mb=args.memory_limit,
            current_latency_ms=(
                lat.current_token_latency_ms if lat else None
            ),
            latency_slope_per_token_ms=latency_slope_per_token,
            latency_threshold_ms=args.latency_limit,
        )

        print("\nForecast:")

        # Context
        if forecast.context_already_saturated:
            print("  ⚠ Context window already saturated")
        elif forecast.tokens_until_context_limit is not None:
            print(f"  Context saturation in: ~{forecast.tokens_until_context_limit} tokens")

        # Memory
        if args.memory_limit is not None:
            limit_str = _fmt_mb(args.memory_limit)
            if forecast.memory_already_exceeded:
                print(f"  ⚠ Memory limit ({limit_str}) already exceeded")
            elif forecast.tokens_until_memory_limit is not None:
                print(f"  Memory limit ({limit_str}) in: ~{forecast.tokens_until_memory_limit} tokens")
            else:
                print(f"  Memory limit ({limit_str}): memory not growing — limit unlikely to be reached")

        # Latency
        if args.latency_limit is not None:
            if forecast.latency_already_exceeded:
                print(f"  ⚠ Latency already exceeds {args.latency_limit:.0f}ms")
            elif forecast.tokens_until_latency_threshold is not None:
                print(f"  Latency >{args.latency_limit:.0f}ms in: ~{forecast.tokens_until_latency_threshold} tokens")
            else:
                print(f"  Latency >{args.latency_limit:.0f}ms: no degradation trend — threshold unlikely to be reached")

    # --- JSON logging ------------------------------------------------------
    _save_run_log(args, result)


def _save_run_log(args: argparse.Namespace, result) -> None:
    """Save the run results as a JSON log file in the ``runs/`` directory."""
    runs_dir = os.path.join(os.getcwd(), "runs")
    os.makedirs(runs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    log_path = os.path.join(runs_dir, f"run_{timestamp}.json")

    log_data: dict = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "prompt_token_count": result.prompt_token_count,
        "generated_token_count": result.generated_token_count,
        "total_token_count": result.total_token_count,
        "generated_text": result.generated_text,
    }

    # Context snapshots
    ctx = result.context_summary
    if ctx is not None:
        log_data["context_summary"] = {
            "max_context": ctx.max_context,
            "final_total_tokens": ctx.final_total_tokens,
            "context_used_pct": ctx.context_used_pct,
            "remaining_tokens": ctx.remaining_tokens,
            "warning_issued": ctx.warning_issued,
        }
        log_data["context_snapshots"] = [
            {
                "step": s.step,
                "total_tokens": s.total_tokens,
                "max_context": s.max_context,
                "context_used_pct": s.context_used_pct,
                "remaining_tokens": s.remaining_tokens,
            }
            for s in ctx.per_step_snapshots
        ]

    # Latency snapshots
    lat = result.latency_summary
    if lat is not None:
        log_data["latency_summary"] = {
            "ttft_ms": lat.ttft_ms,
            "current_token_latency_ms": lat.current_token_latency_ms,
            "rolling_avg_ms": lat.rolling_avg_ms,
            "trend_ms_per_100_tokens": lat.trend_ms_per_100_tokens,
        }
        log_data["latency_snapshots"] = [
            {
                "step": s.step,
                "timestamp": s.timestamp,
                "latency_ms": s.latency_ms,
            }
            for s in lat.per_step_snapshots
        ]

    # Memory snapshots
    mem = result.memory_summary
    if mem is not None:
        log_data["memory_summary"] = {
            "initial_memory_mb": mem.initial_memory_mb,
            "current_memory_mb": mem.current_memory_mb,
            "peak_memory_mb": mem.peak_memory_mb,
            "memory_growth_total_mb": mem.memory_growth_total_mb,
            "avg_growth_per_token_mb": mem.avg_growth_per_token_mb,
            "growth_per_100_tokens_mb": mem.growth_per_100_tokens_mb,
        }
        log_data["memory_snapshots"] = [
            {
                "step": s.step,
                "rss_bytes": s.rss_bytes,
                "rss_mb": s.rss_mb,
                "delta_from_start_mb": s.delta_from_start_mb,
            }
            for s in mem.per_step_snapshots
        ]

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"\nRun log saved to: {log_path}")


# ---------------------------------------------------------------------------
# `analyze` handler
# ---------------------------------------------------------------------------
def _handle_analyze(args: argparse.Namespace) -> None:
    """Execute the ``analyze`` subcommand."""
    from contextwatch.analyzer import load_run, plot_context, plot_latency, plot_memory

    log_path = args.log_file
    if not os.path.isfile(log_path):
        print(f"Error: file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    data = load_run(log_path)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(log_path))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Analyzing run log: {log_path}")
    print(f"Output directory: {output_dir}\n")

    plots_created = []
    for name, plot_fn in [
        ("Latency", plot_latency),
        ("Memory", plot_memory),
        ("Context", plot_context),
    ]:
        path = plot_fn(data, output_dir)
        if path:
            plots_created.append(path)
            print(f"  ✓ {name} plot saved: {path}")

    if plots_created:
        print(f"\n✅ Generated {len(plots_created)} plot(s).")
    else:
        print("\n⚠ No plots generated — log file may be missing snapshot data.")


if __name__ == "__main__":
    main()
