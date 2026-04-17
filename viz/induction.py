"""Induction-head detection via the Olsson+2022 prefix-matching test.

The test: feed the model a sequence `[BOS, r_1..r_N, r_1..r_N]` where `r_1..r_N`
are random tokens. A head is an induction head if, on the repeated half, it
attends from position `i` (in the repeat) to position `i - N + 1` (the token
after the matching one in the first half) — i.e., the head is implementing
"I saw X Y earlier, I'm seeing X now, attend to Y."

We score each (layer, head) pair by averaging, over many random sequences and
over all positions in the repeated half, the attention weight paid to the
"correct" prefix-match position.

References:
    - Olsson et al. 2022, "In-context Learning and Induction Heads" (Anthropic).
"""

from __future__ import annotations

import numpy as np
import torch

from viz._model import load_eager_model


@torch.no_grad()
def induction_scores(
    n_seqs: int = 16,
    seq_len: int = 50,
    seed: int = 0,
) -> np.ndarray:
    """Return a [n_layers, n_heads] array of prefix-matching scores in [0, 1]."""
    model, tokenizer = load_eager_model()
    device = next(model.parameters()).device
    vocab_size = model.config.vocab_size
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id  # Pythia uses EOS as a sentinel
    if bos is None:
        bos = 0

    g = torch.Generator(device="cpu").manual_seed(seed)

    scores = np.zeros((n_layers, n_heads), dtype=np.float64)
    count = 0

    for _ in range(n_seqs):
        rand = torch.randint(0, vocab_size, (seq_len,), generator=g)
        seq = torch.cat([torch.tensor([bos]), rand, rand]).to(device)
        out = model(seq.unsqueeze(0), output_attentions=True, use_cache=False)
        # out.attentions: tuple of [1, n_heads, T, T] per layer.
        # Positions:
        #   0       : BOS
        #   1..N    : first copy of r_1..r_N
        #   N+1..2N : second copy (the "repeat")
        # For position p in [N+2, 2N] (second copy, skipping first token which
        # has no prefix-match target in the cached copy), the induction target
        # is (p - N). That is: at position p we see token r_{p-N}, and we want
        # to attend to the token at position (p - N + 1) - 1 + 1 = p - N, which
        # held r_{p-N+1} in the first copy... wait, let's redo this carefully.
        #
        # Layout with 1-indexed positions:
        #   pos 0 : BOS
        #   pos k (1 <= k <= N)   : r_k  (first copy)
        #   pos N+k (1 <= k <= N) : r_k  (second copy)
        # At position N+k, we are *predicting* r_{k+1}. The prefix-match head
        # looks back at position k (which holds r_k, matching the current
        # token) and shifts by +1 via the OV circuit. So the attention from
        # position N+k should be peaked at position k. (For k=N there is no
        # next token to predict, but the attention pattern at N+N still scores
        # match->k=N.)
        # We score position N+k for k in [1, N], reading off attention to k.
        for layer in range(n_layers):
            attn = out.attentions[layer][0]  # [n_heads, T, T]
            for k in range(1, seq_len + 1):
                src = seq_len + k       # position in repeat
                tgt = k                 # matching position in first copy
                scores[layer] += attn[:, src, tgt].detach().float().cpu().numpy().astype(np.float64)
            del attn
        count += seq_len

    scores /= max(count, 1)
    return scores
