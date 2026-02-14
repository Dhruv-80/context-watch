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

    # TODO (Phase 2): add subcommands for monitoring, forecasting, etc.
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
    )

    # --- output (kept deliberately simple for Phase 0/1) -------------------
    print(f"\nPrompt tokens: {result.prompt_token_count}")
    print(f"Generated tokens: {result.generated_token_count}")
    print(f"Total tokens: {result.total_token_count}")

    # TODO (Phase 2): structured output / JSON mode


if __name__ == "__main__":
    main()
