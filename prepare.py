"""prepare.py — fixed constants, base model loading, data prep, and eval harness.

This file is READ-ONLY from the agent's perspective. It defines the experimental
environment: the base model, the target layer, the activation cache, and the
ground-truth evaluation. The agent edits only `train_sae.py`.

Design note: mirrors the role of `prepare.py` in karpathy/autoresearch, but the
eval metric is `ce_loss_delta` (extra nats of cross-entropy when SAE reconstruction
is spliced in at the target layer) rather than `val_bpb`.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

# ------------------------------------------------------------------------------
# Fixed constants. Do NOT modify these from train_sae.py.
# ------------------------------------------------------------------------------

# Base model: Pythia-160M-deduped. Well-documented in SAE literature, small enough
# for fast iteration, large enough that findings plausibly transfer.
MODEL_NAME: str = "EleutherAI/pythia-160m-deduped"
D_MODEL: int = 768           # residual stream width for Pythia-160M
N_LAYERS: int = 12
LAYER: int = 8               # residual stream site the SAE is trained on
MAX_SEQ_LEN: int = 512       # context length used for activation extraction
VOCAB_SIZE: int = 50304      # Pythia vocab

# SAE constraints — enforced by the eval harness, violating any => run is `discard`.
L0_TARGET: int = 64          # soft sparsity target; L0 above this is invalid
DEAD_FRAC_MAX: float = 0.10
PEAK_VRAM_GB_MAX: float = 14.0

# Experiment budget (wall clock, excluding eval).
TIME_BUDGET_SEC: int = 300   # 5 minutes, matching autoresearch

# Data volumes for prepare-time caching.
TRAIN_TOKENS: int = 10_000_000
EVAL_TOKENS: int = 500_000

# Cache locations. Everything lives under ~/.cache/automechresearch so the repo
# stays clean and the cache survives `git clean -fdx`.
CACHE_DIR: Path = Path.home() / ".cache" / "automechresearch"
TOKENS_DIR: Path = CACHE_DIR / "tokens"
ACTS_DIR: Path = CACHE_DIR / "activations"
MODEL_CACHE_DIR: Path = CACHE_DIR / "models"

# Dtype for activation cache. fp16 halves disk vs fp32 with negligible quality loss
# for SAE training; we upcast to fp32 inside the SAE forward pass.
ACT_DTYPE: torch.dtype = torch.float16

# Reproducibility.
SEED: int = 0


# ------------------------------------------------------------------------------
# Device selection. The agent does not set this — it is fixed per machine.
# ------------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError(
        "automechresearch currently requires a CUDA GPU with >=16 GB VRAM. "
        "See README.md for the portability roadmap."
    )


# ------------------------------------------------------------------------------
# Base model loading. Kept as a module-level cache so repeat calls are cheap.
# ------------------------------------------------------------------------------

_MODEL = None
_TOKENIZER = None


def load_base_model():
    """Return (model, tokenizer) for the frozen base transformer.

    The model is returned in eval mode, on the GPU, in bf16. It is never trained
    or fine-tuned — only used to (1) dump activations during prepare and (2)
    splice SAE reconstructions back in during eval.
    """
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER

    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=str(MODEL_CACHE_DIR)
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=str(MODEL_CACHE_DIR),
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    model.to(get_device())
    for p in model.parameters():
        p.requires_grad_(False)

    _MODEL, _TOKENIZER = model, tokenizer
    return model, tokenizer


if __name__ == "__main__":
    # Smoke-check: resolve paths and announce the configuration. Real data prep
    # lands in the next commit.
    print(f"MODEL_NAME        = {MODEL_NAME}")
    print(f"D_MODEL           = {D_MODEL}")
    print(f"LAYER             = {LAYER}")
    print(f"MAX_SEQ_LEN       = {MAX_SEQ_LEN}")
    print(f"L0_TARGET         = {L0_TARGET}")
    print(f"TIME_BUDGET_SEC   = {TIME_BUDGET_SEC}")
    print(f"TRAIN_TOKENS      = {TRAIN_TOKENS:,}")
    print(f"EVAL_TOKENS       = {EVAL_TOKENS:,}")
    print(f"CACHE_DIR         = {CACHE_DIR}")
    print(f"device            = {get_device()}")
