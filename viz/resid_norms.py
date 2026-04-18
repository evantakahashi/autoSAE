"""Per-layer residual stream norm statistics on real text.

Tracks how ||h|| grows through the network — useful context for why the SAE
target layer has a specific scale. We use a handful of diverse natural-language
sentences rather than the cached activation memmap (keeps the tour independent
of `prepare.py` having been run).
"""

from __future__ import annotations

import numpy as np
import torch

from viz._model import load_eager_model


DEFAULT_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In 1969, Neil Armstrong became the first human to walk on the Moon.",
    "Python is a widely used high-level programming language.",
    "The mitochondrion is the powerhouse of the cell.",
    "Shakespeare wrote plays such as Hamlet, Macbeth, and Othello.",
    "Machine learning models are trained by minimizing a loss function.",
]


@torch.no_grad()
def residual_norms(texts: list[str] | None = None) -> np.ndarray:
    """Return [n_layers+1] mean ||h||_2 per layer, averaged over all token positions."""
    if texts is None:
        texts = DEFAULT_TEXTS
    model, tokenizer = load_eager_model()
    device = next(model.parameters()).device

    sums = None
    counts = 0
    for text in texts:
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        out = model(ids, output_hidden_states=True, use_cache=False)
        # hidden_states: tuple of [1, T, D], length n_layers+1
        norms = torch.stack(
            [h[0].float().norm(dim=-1).mean() for h in out.hidden_states]
        )  # [n_layers+1]
        if sums is None:
            sums = norms.detach().cpu().numpy().astype(np.float64)
        else:
            sums += norms.detach().cpu().numpy().astype(np.float64)
        counts += 1

    return sums / max(counts, 1)
