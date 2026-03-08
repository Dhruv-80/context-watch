#!/usr/bin/env python3
"""ContextWatch test runner.

Loads the model once and runs all validation tests, printing clear
PASS/FAIL results for each test.

Usage:
    python run_tests.py
"""

from __future__ import annotations

import sys
import traceback

from contextwatch.utils import load_model

# Import test modules — each exposes a run(model, tokenizer) -> list[str]
from tests import test_small_prompt, test_long_prompt, test_latency, test_memory


# ---------------------------------------------------------------------------
# Registry of tests
# ---------------------------------------------------------------------------
TESTS: list[tuple[str, object]] = [
    ("Small prompt generation (50 tokens)", test_small_prompt),
    ("Long prompt context tracking (~1000 tokens)", test_long_prompt),
    ("Latency tracking", test_latency),
    ("Memory tracking", test_memory),
]


def main() -> None:
    """Run all validation tests with a shared model instance."""
    print("=" * 60)
    print("  ContextWatch — Validation Test Suite")
    print("=" * 60)

    # Load model once for all tests
    print("\nLoading model (distilgpt2) ...\n")
    model, tokenizer = load_model("distilgpt2")

    passed = 0
    failed = 0
    results: list[tuple[str, bool, list[str]]] = []

    for name, module in TESTS:
        print(f"  Running: {name} ...", end=" ", flush=True)
        try:
            failures = module.run(model, tokenizer)
            if failures:
                print("FAIL")
                results.append((name, False, failures))
                failed += 1
            else:
                print("PASS")
                results.append((name, True, []))
                passed += 1
        except Exception:
            tb = traceback.format_exc()
            print("FAIL (exception)")
            results.append((name, False, [tb]))
            failed += 1

    # --- Summary -----------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if failed > 0:
        print("\nFailure details:\n")
        for name, ok, errors in results:
            if not ok:
                print(f"  ✗ {name}")
                for err in errors:
                    for line in err.strip().splitlines():
                        print(f"      {line}")
                print()
        sys.exit(1)
    else:
        print("\n✅ All tests passed.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
