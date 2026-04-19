"""SAE variants used by the README demo sweep.

Keeping these here (not in `train_sae.py`) so that `train_sae.py` remains the
single file the agent edits. If the agent later reproduces any of these
architectures, it can import from here or copy the body — both are fine.

References:
    - Bricken et al. 2023 (ReLU + L1 baseline)
    - Gao et al. 2024 "Scaling and evaluating sparse autoencoders" (TopK)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReluSAE(nn.Module):
    """ReLU encoder + linear decoder, L1 sparsity (norm-weighted)."""

    def __init__(self, d_model: int, expansion: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_features = d_model * expansion
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.W_enc = nn.Parameter(torch.empty(d_model, self.n_features))
        self.b_enc = nn.Parameter(torch.zeros(self.n_features))
        self.W_dec = nn.Parameter(torch.empty(self.n_features, d_model))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t().contiguous())
            self.W_dec.div_(self.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        return self.decode(f), f


class TopKAuxSAE(nn.Module):
    """TopK SAE with Gao+2024's AuxK dead-feature revival loss.

    On top of plain TopK, we maintain a counter of steps-since-last-active per
    feature. Features that haven't fired for `dead_threshold` steps are "dead."
    During training we additionally reconstruct the residual `x - recon` using
    the top `k_aux` of the dead features — this forces a gradient through them
    and lets them come back to life instead of staying at zero forever.

    Use `model.forward_with_aux(x)` during training; `model.forward(x)` during
    eval (what `prepare.evaluate_sae` calls).
    """

    def __init__(
        self,
        d_model: int,
        expansion: int = 8,
        k: int = 32,
        k_aux: int = 512,
        dead_threshold: int = 400,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_features = d_model * expansion
        self.k = k
        self.k_aux = k_aux
        self.dead_threshold = dead_threshold
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.W_enc = nn.Parameter(torch.empty(d_model, self.n_features))
        self.b_enc = nn.Parameter(torch.zeros(self.n_features))
        self.W_dec = nn.Parameter(torch.empty(self.n_features, d_model))
        self.register_buffer(
            "steps_since_active",
            torch.zeros(self.n_features, dtype=torch.long),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t().contiguous())
            self.W_dec.div_(self.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8))

    def _topk_mask(self, pre: torch.Tensor, k: int) -> torch.Tensor:
        topv, topi = pre.topk(k, dim=-1)
        mask = torch.zeros_like(pre)
        mask.scatter_(-1, topi, F.relu(topv))
        return mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        return self._topk_mask(pre, self.k)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        return self.decode(f), f

    def forward_with_aux(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Return (recon, features, aux_loss, n_dead)."""
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        features = self._topk_mask(pre, self.k)
        recon = self.decode(features)

        with torch.no_grad():
            active_this_batch = (features > 0).any(dim=0)  # [F]
            self.steps_since_active += 1
            self.steps_since_active[active_this_batch] = 0

        dead = self.steps_since_active > self.dead_threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            aux_loss = torch.zeros((), device=x.device)
            return recon, features, aux_loss, 0

        k_aux = min(self.k_aux, n_dead)
        pre_dead = pre.masked_fill(~dead[None, :], float("-inf"))
        aux_features = self._topk_mask(pre_dead, k_aux)
        # Reconstruct residual with dead-feature-only aux recon. No b_dec.
        residual = (x - recon).detach()
        aux_recon = aux_features @ self.W_dec
        aux_loss = F.mse_loss(aux_recon, residual)
        return recon, features, aux_loss, n_dead


class TopKSAE(nn.Module):
    """TopK SAE (Gao+2024).

    Exactly K features active per token by construction — no L1 needed. This
    gives a hard L0 guarantee and typically a cleaner MSE/L0 Pareto.
    """

    def __init__(self, d_model: int, expansion: int = 8, k: int = 32):
        super().__init__()
        self.d_model = d_model
        self.n_features = d_model * expansion
        self.k = k
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.W_enc = nn.Parameter(torch.empty(d_model, self.n_features))
        self.b_enc = nn.Parameter(torch.zeros(self.n_features))
        self.W_dec = nn.Parameter(torch.empty(self.n_features, d_model))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t().contiguous())
            self.W_dec.div_(self.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        # TopK over the feature axis, ReLU'd to keep the "activation" semantics.
        topv, topi = pre.topk(self.k, dim=-1)
        mask = torch.zeros_like(pre)
        mask.scatter_(-1, topi, F.relu(topv))
        return mask

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        return self.decode(f), f
