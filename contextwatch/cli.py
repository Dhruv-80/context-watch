"""CLI entry point for ContextWatch."""

from __future__ import annotations

import argparse
import re
import sys

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


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="contextwatch",
        description="Controlled LLM inference with token accounting.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- `run` subcommand --------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Load a model, run stepwise inference, and report token counts.",
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
        # Use GB for values >= 1024 MB, otherwise MB
        def _fmt_mb(val: float) -> str:
            if val >= 1024.0:
                return f"{val / 1024:.2f} GB"
            return f"{val:.2f} MB"

        print("\nMemory Metrics:")
        print(f"  Current memory: {_fmt_mb(mem.current_memory_mb)}")
        print(f"  Peak memory: {_fmt_mb(mem.peak_memory_mb)}")
        sign = "+" if mem.memory_growth_total_mb >= 0 else ""
        print(f"  Total growth: {sign}{mem.memory_growth_total_mb:.2f} MB")
        sign = "+" if mem.growth_per_100_tokens_mb >= 0 else ""
        print(f"  Growth rate: {sign}{mem.growth_per_100_tokens_mb:.2f} MB per 100 tokens")
        print(f"  Avg per token: {mem.avg_growth_per_token_mb:.4f} MB")

    # --- forecasting (Phase 5) ---------------------------------------------
    # Derive latency slope per token from the per-100-tokens value
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
            limit_str = _fmt_mb(args.memory_limit) if mem else f"{args.memory_limit:.0f} MB"
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


if __name__ == "__main__":
    main()
