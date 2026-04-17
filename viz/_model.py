"""Viz-only base model loader that exposes attention weights.

`prepare.load_base_model` uses the default (sdpa) attention backend, which
doesn't return attention matrices. For the tour we need `output_attentions=True`
to work, so we load a separate copy with `attn_implementation="eager"`.

This is strictly read-only and does not touch the frozen prepare.py pipeline.
"""

from __future__ import annotations

import os

import torch

from prepare import MODEL_CACHE_DIR, MODEL_NAME, get_device


_EAGER_MODEL = None
_EAGER_TOKENIZER = None


def load_eager_model():
    """Return (model, tokenizer) with eager attention so output_attentions works."""
    global _EAGER_MODEL, _EAGER_TOKENIZER
    if _EAGER_MODEL is not None:
        return _EAGER_MODEL, _EAGER_TOKENIZER

    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=str(MODEL_CACHE_DIR)
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=str(MODEL_CACHE_DIR),
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.eval()
    model.to(get_device())
    for p in model.parameters():
        p.requires_grad_(False)

    _EAGER_MODEL, _EAGER_TOKENIZER = model, tokenizer
    return model, tokenizer
