"""Analyzer module — loads JSON run logs and generates matplotlib plots.

Provides utilities used by the ``contextwatch analyze`` subcommand to
visualise latency, memory, and context utilisation over time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_run(path: str) -> dict[str, Any]:
    """Load a JSON run log from *path*.

    Args:
        path: Path to the JSON run log file.

    Returns:
        The parsed run data as a dictionary.

    Raises:
        FileNotFoundError: If *path* does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path, "r") as f:
        return json.load(f)


def plot_latency(data: dict[str, Any], output_dir: str) -> str:
    """Generate a latency-per-token line chart.

    Args:
        data: Run data dictionary (from :func:`load_run`).
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved PNG file.
    """
    import matplotlib.pyplot as plt

    snapshots = data.get("latency_snapshots", [])
    if not snapshots:
        print("  No latency data to plot.")
        return ""

    steps = [s["step"] for s in snapshots]
    latencies = [s["latency_ms"] for s in snapshots]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, latencies, marker=".", markersize=3, linewidth=1, color="#4f8ff7")
    ax.set_xlabel("Generation Step")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Token Latency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "latency.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_memory(data: dict[str, Any], output_dir: str) -> str:
    """Generate a memory growth line chart.

    Args:
        data: Run data dictionary (from :func:`load_run`).
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved PNG file.
    """
    import matplotlib.pyplot as plt

    snapshots = data.get("memory_snapshots", [])
    if not snapshots:
        print("  No memory data to plot.")
        return ""

    steps = [s["step"] for s in snapshots]
    rss_mb = [s["rss_mb"] for s in snapshots]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, rss_mb, marker=".", markersize=3, linewidth=1, color="#f7734f")
    ax.set_xlabel("Generation Step")
    ax.set_ylabel("RSS Memory (MB)")
    ax.set_title("Process Memory During Inference")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "memory.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_context(data: dict[str, Any], output_dir: str) -> str:
    """Generate a context utilisation chart.

    Args:
        data: Run data dictionary (from :func:`load_run`).
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved PNG file.
    """
    import matplotlib.pyplot as plt

    snapshots = data.get("context_snapshots", [])
    if not snapshots:
        print("  No context data to plot.")
        return ""

    steps = [s["step"] for s in snapshots]
    pcts = [s["context_used_pct"] * 100 for s in snapshots]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, pcts, marker=".", markersize=3, linewidth=1, color="#4fc76f")
    ax.axhline(y=75, color="#ff6b6b", linestyle="--", alpha=0.7, label="75% warning")
    ax.axhline(y=100, color="#ff0000", linestyle="-", alpha=0.5, label="100% limit")
    ax.set_xlabel("Generation Step")
    ax.set_ylabel("Context Used (%)")
    ax.set_title("Context Window Utilisation")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "context.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
