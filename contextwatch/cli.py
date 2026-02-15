"""CLI entry point for ContextWatch."""

from __future__ import annotations

import argparse
import sys

from contextwatch.inference_loop import run_inference
from contextwatch.utils import load_model


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


if __name__ == "__main__":
    main()
