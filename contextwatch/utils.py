"""Utility helpers for model loading and device selection."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def get_device() -> torch.device:
    """Return the best available device (CUDA → CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_name: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a Hugging Face causal language model and its tokenizer.

    Args:
        model_name: A Hugging Face Hub model identifier (e.g. ``"distilgpt2"``).

    Returns:
        A ``(model, tokenizer)`` tuple placed on the best available device.
    """
    device = get_device()
    print(f"Loading model '{model_name}' on {device} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    # TODO (Phase 2): accept dtype / quantisation options
    return model, tokenizer
