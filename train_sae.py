"""train_sae.py — the single file the agent edits.

Contains:
  - the SAE architecture
  - the loss function
  - the training loop
  - the summary block that prints to stdout at the end

Everything here is fair game for the agent: architecture (ReLU / TopK / JumpReLU /
Gated / whatever), auxiliary losses, optimizer, LR schedule, batch size, dict size,
initialization. The only constraint is that at the end of training, the module
exposes `forward(x) -> (recon, features)` so `prepare.evaluate_sae` can score it.

DO NOT import or modify `prepare.py`'s constants or eval harness — those are the
fixed experimental environment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import D_MODEL


# ------------------------------------------------------------------------------
# Baseline SAE: ReLU encoder + linear decoder + L1 sparsity penalty.
#
# This is the simplest thing that works. Everything the agent tries should be
# compared against the baseline numbers this produces on the first run.
#
# References:
#   - Bricken et al. 2023, "Towards Monosemanticity" (Anthropic)
#   - Cunningham et al. 2023, "Sparse Autoencoders Find Highly Interpretable
#     Features in Language Models"
# ------------------------------------------------------------------------------

class SAE(nn.Module):
    def __init__(self, d_model: int = D_MODEL, expansion: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_features = d_model * expansion

        # Pre-encoder centering bias. Subtracting the mean activation before
        # encoding is standard (Bricken+2023) — it lets features learn pure
        # directions rather than absorbing the activation mean.
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Encoder: d_model -> n_features, ReLU.
        self.W_enc = nn.Parameter(torch.empty(d_model, self.n_features))
        self.b_enc = nn.Parameter(torch.zeros(self.n_features))

        # Decoder: n_features -> d_model.
        self.W_dec = nn.Parameter(torch.empty(self.n_features, d_model))

        self._init_weights()

    def _init_weights(self) -> None:
        # Kaiming on encoder; decoder initialized as encoder transpose, then
        # unit-normed columns (the "normalize decoder" trick keeps gradients
        # balanced and encourages distinct features).
        nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t().contiguous())
            self.W_dec.div_(self.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        recon = self.decode(features)
        return recon, features


def sae_loss(
    x: torch.Tensor,
    recon: torch.Tensor,
    features: torch.Tensor,
    decoder_norms: torch.Tensor,
    l1_coeff: float,
) -> tuple[torch.Tensor, dict]:
    """MSE reconstruction + L1 sparsity penalty weighted by decoder column norms.

    Weighting the L1 by ||W_dec_i|| is the "norm-aware L1" trick — it prevents
    the model from trivially shrinking feature activations while inflating
    decoder norms to compensate.
    """
    mse = F.mse_loss(recon, x)
    weighted_acts = features * decoder_norms[None, :]
    l1 = weighted_acts.abs().sum(dim=-1).mean()
    loss = mse + l1_coeff * l1
    return loss, {"mse": mse.detach().item(), "l1": l1.detach().item()}
