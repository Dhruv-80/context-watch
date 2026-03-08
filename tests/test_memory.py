"""Test 4: Memory tracking.

Runs inference and verifies that memory values are captured at each
generation step and the summary statistics are correct.
"""

from __future__ import annotations

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from contextwatch.inference_loop import run_inference


def run(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> list[str]:
    """Run memory tracking validation. Returns a list of failure messages (empty = pass)."""
    failures: list[str] = []

    result = run_inference(
        model=model,
        tokenizer=tokenizer,
        prompt="Hello",
        max_tokens=50,
    )

    mem = result.memory_summary

    # 1. Memory summary must exist
    if mem is None:
        failures.append("Memory summary is None")
        return failures

    # 2. Initial memory must be positive
    if mem.initial_memory_mb <= 0:
        failures.append(f"Initial memory must be > 0, got {mem.initial_memory_mb}")

    # 3. Current memory must be positive
    if mem.current_memory_mb <= 0:
        failures.append(f"Current memory must be > 0, got {mem.current_memory_mb}")

    # 4. Peak memory must be >= current memory
    if mem.peak_memory_mb < mem.current_memory_mb:
        failures.append(
            f"Peak memory ({mem.peak_memory_mb}) must be >= "
            f"current ({mem.current_memory_mb})"
        )

    # 5. Per-step snapshots should match generated token count
    if len(mem.per_step_snapshots) != result.generated_token_count:
        failures.append(
            f"Memory snapshot count={len(mem.per_step_snapshots)}, "
            f"expected={result.generated_token_count}"
        )

    # 6. Each snapshot should have positive rss_mb
    for snap in mem.per_step_snapshots:
        if snap.rss_mb <= 0:
            failures.append(f"Step {snap.step}: rss_mb={snap.rss_mb}, expected > 0")
            break  # one failure is enough

    # 7. Memory growth total — RSS can shrink slightly due to GC or OS
    #    page reclamation, so allow a small negative value.
    if mem.memory_growth_total_mb < -50.0:
        failures.append(
            f"Memory growth unexpectedly negative: {mem.memory_growth_total_mb:.2f} MB"
        )

    return failures
