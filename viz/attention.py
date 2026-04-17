"""Attention-pattern capture for a single short prompt.

For a tour we want one concrete example where the reader can see the raw
attention matrices across all layers and heads. We use a short english
sentence so token strings are printable.
"""

from __future__ import annotations

import numpy as np
import torch

from viz._model import load_eager_model


DEFAULT_PROMPT = (
    "When Mary went to the store, Mary bought apples. John went to the store "
    "and John bought oranges."
)


@torch.no_grad()
def capture_attention(prompt: str = DEFAULT_PROMPT) -> tuple[np.ndarray, list[str]]:
    """Run `prompt` through the base model. Return (attentions, token_strings).

    attentions: np.ndarray [n_layers, n_heads, T, T], float32.
    token_strings: list of T decoded tokens.
    """
    model, tokenizer = load_eager_model()
    device = next(model.parameters()).device

    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model(ids, output_attentions=True, use_cache=False)
    # Stack: [n_layers, 1, n_heads, T, T] -> squeeze batch
    attn = torch.stack(out.attentions, dim=0)[:, 0]  # [n_layers, n_heads, T, T]
    attn_np = attn.float().cpu().numpy()

    tokens = [tokenizer.decode([t]) for t in ids[0].tolist()]
    return attn_np, tokens
