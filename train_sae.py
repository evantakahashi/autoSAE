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

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    D_MODEL,
    SEED,
    TIME_BUDGET_SEC,
    evaluate_sae,
    get_device,
    load_activations,
    validate_constraints,
)


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


# ------------------------------------------------------------------------------
# Activation buffer: yields shuffled fp32 activation batches from the on-disk
# fp16 memmap, upcasting on the GPU. Cheap, memory-bounded.
# ------------------------------------------------------------------------------

def activation_loader(batch_size: int, device: torch.device, seed: int = SEED):
    acts = load_activations("train")  # np.memmap, [N, D] fp16
    n = acts.shape[0]
    rng = np.random.default_rng(seed)
    # Pre-shuffled index buffer, refilled as we consume it.
    while True:
        perm = rng.permutation(n)
        for start in range(0, n - batch_size, batch_size):
            idx = perm[start : start + batch_size]
            # memmap fancy indexing returns a numpy array in RAM.
            batch = np.asarray(acts[idx], dtype=np.float32)
            yield torch.from_numpy(batch).to(device, non_blocking=True)


# ------------------------------------------------------------------------------
# Training loop. Runs for exactly TIME_BUDGET_SEC wall-clock, then evaluates.
# The agent is free to rip this up — the only invariant is that the eventual
# SAE module is passed to `evaluate_sae` and the summary block is printed.
# ------------------------------------------------------------------------------

def train():
    torch.manual_seed(SEED)
    device = get_device()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Hyperparameters — all fair game for the agent to tune.
    expansion = 8
    batch_size = 4096
    lr = 3e-4
    l1_coeff = 5e-4
    log_every = 200

    sae = SAE(d_model=D_MODEL, expansion=expansion).to(device)
    optim = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    loader = activation_loader(batch_size, device)

    t0 = time.time()
    step = 0
    last_log_time = t0
    total_tokens = 0

    while True:
        elapsed = time.time() - t0
        if elapsed >= TIME_BUDGET_SEC:
            break

        x = next(loader)
        decoder_norms = sae.W_dec.norm(dim=-1)
        recon, features = sae(x)
        loss, aux = sae_loss(x, recon, features, decoder_norms, l1_coeff)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        # Re-normalize decoder columns to unit norm after each step — standard
        # SAE bookkeeping that keeps the L1-weighting well-defined.
        with torch.no_grad():
            sae.W_dec.div_(sae.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        optim.step()

        total_tokens += x.shape[0]
        step += 1
        if step % log_every == 0:
            now = time.time()
            rate = log_every * batch_size / max(now - last_log_time, 1e-9)
            last_log_time = now
            print(
                f"step {step:6d} | t {elapsed:6.1f}s | loss {loss.item():.4f} | "
                f"mse {aux['mse']:.4f} | l1 {aux['l1']:.4f} | "
                f"{rate/1e6:.2f} Mtok/s",
                flush=True,
            )

    training_seconds = time.time() - t0

    # ---- Eval ---------------------------------------------------------------
    sae.eval()
    metrics = evaluate_sae(sae)
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    )
    peak_vram_gb = peak_vram_mb / 1024.0

    is_valid, invalid_reason = validate_constraints(metrics, peak_vram_gb)

    # ---- Summary block ------------------------------------------------------
    # Mirrors autoresearch: key:value lines, one per metric, grep-friendly.
    print("---")
    print(f"ce_loss_delta:    {metrics['ce_loss_delta']:.6f}")
    print(f"ce_clean:         {metrics['ce_clean']:.6f}")
    print(f"ce_patched:       {metrics['ce_patched']:.6f}")
    print(f"l0:               {metrics['l0']:.2f}")
    print(f"dead_fraction:    {metrics['dead_fraction']:.4f}")
    print(f"mse_normalized:   {metrics['mse_normalized']:.6f}")
    print(f"variance_explained: {metrics['variance_explained']:.4f}")
    print(f"n_features:       {metrics['n_features']}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"peak_vram_gb:     {peak_vram_gb:.2f}")
    print(f"num_steps:        {step}")
    print(f"total_tokens_M:   {total_tokens/1e6:.2f}")
    print(f"expansion:        {expansion}")
    print(f"batch_size:       {batch_size}")
    print(f"lr:               {lr:.2e}")
    print(f"l1_coeff:         {l1_coeff:.2e}")
    print(f"valid:            {int(is_valid)}")
    if not is_valid:
        print(f"invalid_reason:   {invalid_reason}")


if __name__ == "__main__":
    train()
