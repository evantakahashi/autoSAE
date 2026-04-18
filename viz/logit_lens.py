"""Logit lens — project every layer's residual stream through the unembedding.

For a short prompt, at each layer L we take the residual stream h^L, apply the
final LayerNorm, multiply by the unembedding W_U, and read off the top-K
predictions. This shows how the model's "best guess for next token" evolves as
computation proceeds through the stack.

Reference: nostalgebraist (2020), "interpreting GPT: the logit lens."
"""

from __future__ import annotations

import numpy as np
import torch

from viz._model import load_eager_model


DEFAULT_PROMPT = (
    "The Eiffel Tower is located in the city of"
)


@torch.no_grad()
def logit_lens(
    prompt: str = DEFAULT_PROMPT,
    top_k: int = 5,
) -> tuple[list[str], list[list[list[tuple[str, float]]]]]:
    """Return (tokens, lens) where lens[layer][pos] = [(tok, prob), ...] top-K.

    Layer 0 is the embeddings; layers 1..N are after each transformer block.
    """
    model, tokenizer = load_eager_model()
    device = next(model.parameters()).device

    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model(ids, output_hidden_states=True, use_cache=False)
    # hidden_states: tuple of [1, T, D] — len = n_layers + 1 (embeds + each block).
    hidden = out.hidden_states

    # Apply final layer norm and unembedding. GPTNeoX structure:
    #   model.gpt_neox.final_layer_norm  : Pre-unembed LN
    #   model.embed_out                  : Unembedding (lm_head)
    ln_f = model.gpt_neox.final_layer_norm
    W_U = model.embed_out

    tokens = [tokenizer.decode([t]) for t in ids[0].tolist()]
    T = ids.shape[1]

    lens: list[list[list[tuple[str, float]]]] = []
    for h in hidden:
        logits = W_U(ln_f(h))  # [1, T, V]
        probs = logits.float().softmax(dim=-1)[0]  # [T, V]
        top = probs.topk(top_k, dim=-1)
        layer_entries = []
        for pos in range(T):
            ids_p = top.indices[pos].cpu().tolist()
            probs_p = top.values[pos].cpu().tolist()
            layer_entries.append([
                (tokenizer.decode([i]), float(p)) for i, p in zip(ids_p, probs_p)
            ])
        lens.append(layer_entries)

    return tokens, lens
