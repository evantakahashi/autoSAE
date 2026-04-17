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
from typing import Iterator

import numpy as np
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
#
# With D_MODEL=768 and fp16 activations, each token costs 1.5 KB on disk.
# 4M tokens => ~6.0 GB; 500K eval => ~0.75 GB. This fits comfortably on any
# machine with ~10 GB free. At batch 4096, the 5-min training budget on a
# 4070 Ti Super will epoch over 4M tokens ~15x — plenty for SAE convergence.
TRAIN_TOKENS: int = 4_000_000
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
        dtype=torch.bfloat16,
    )
    model.eval()
    model.to(get_device())
    for p in model.parameters():
        p.requires_grad_(False)

    _MODEL, _TOKENIZER = model, tokenizer
    return model, tokenizer


# ------------------------------------------------------------------------------
# Tokenization + data shards.
#
# We use a streaming slice of OpenWebText, tokenized with the base model's
# tokenizer, packed into fixed-length sequences of MAX_SEQ_LEN, and saved as
# uint16 memmap shards (Pythia vocab fits in uint16). This is the same
# packing strategy nanochat / tinystories use.
# ------------------------------------------------------------------------------

TOKENS_DTYPE = np.uint16


def _tokens_path(split: str) -> Path:
    return TOKENS_DIR / f"{split}.bin"


def _packed_sequences(tokenizer, target_tokens: int) -> Iterator[list[int]]:
    """Yield lists of token IDs of length MAX_SEQ_LEN, packed from a text stream.

    Uses HuggingFace `datasets` in streaming mode so we never download the full
    corpus. We concatenate docs, separated by the eos token, and slice into
    fixed-length chunks.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "Skylion007/openwebtext",
        split="train",
        streaming=True,
    )

    eos = tokenizer.eos_token_id
    buf: list[int] = []
    emitted = 0
    for row in ds:
        ids = tokenizer.encode(row["text"])
        buf.extend(ids)
        buf.append(eos)
        while len(buf) >= MAX_SEQ_LEN:
            chunk = buf[:MAX_SEQ_LEN]
            buf = buf[MAX_SEQ_LEN:]
            yield chunk
            emitted += MAX_SEQ_LEN
            if emitted >= target_tokens:
                return


def _write_tokens_split(tokenizer, split: str, target_tokens: int) -> Path:
    from tqdm import tqdm

    os.makedirs(TOKENS_DIR, exist_ok=True)
    out = _tokens_path(split)
    if out.exists() and out.stat().st_size >= target_tokens * 2:
        return out  # already cached

    n_seqs = target_tokens // MAX_SEQ_LEN
    mm = np.memmap(out, dtype=TOKENS_DTYPE, mode="w+", shape=(n_seqs * MAX_SEQ_LEN,))

    cursor = 0
    pbar = tqdm(total=n_seqs, desc=f"tokenize/{split}")
    for chunk in _packed_sequences(tokenizer, target_tokens):
        mm[cursor : cursor + MAX_SEQ_LEN] = np.asarray(chunk, dtype=TOKENS_DTYPE)
        cursor += MAX_SEQ_LEN
        pbar.update(1)
        if cursor >= n_seqs * MAX_SEQ_LEN:
            break
    pbar.close()
    mm.flush()
    return out


def prepare_tokens() -> None:
    """One-time step: pack ~10M train tokens and ~500K eval tokens to disk."""
    _, tokenizer = load_base_model()
    _write_tokens_split(tokenizer, "train", TRAIN_TOKENS)
    _write_tokens_split(tokenizer, "eval", EVAL_TOKENS)


def load_tokens(split: str) -> np.ndarray:
    """Return a memmapped array of packed tokens shaped [n_seqs, MAX_SEQ_LEN]."""
    path = _tokens_path(split)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing tokenized shard at {path}. Run `uv run prepare.py` first."
        )
    arr = np.memmap(path, dtype=TOKENS_DTYPE, mode="r")
    n_seqs = arr.shape[0] // MAX_SEQ_LEN
    return arr[: n_seqs * MAX_SEQ_LEN].reshape(n_seqs, MAX_SEQ_LEN)


# ------------------------------------------------------------------------------
# Activation caching.
#
# We register a forward hook on the LAYER-th residual stream and dump activations
# to disk as an fp16 memmap. Shape: [n_tokens, D_MODEL]. This is the SAE training
# substrate — the agent's SAE only ever sees these cached vectors (plus the
# held-out eval set), never the base model itself.
# ------------------------------------------------------------------------------

def _acts_path(split: str) -> Path:
    return ACTS_DIR / f"{split}_layer{LAYER}.bin"


def _residual_hook_target(model):
    """Return the submodule whose output is the residual stream after LAYER."""
    # Pythia (GPTNeoX) layout: model.gpt_neox.layers[i] -> residual post-block.
    return model.gpt_neox.layers[LAYER]


@torch.no_grad()
def _extract_activations(split: str, target_tokens: int) -> Path:
    from tqdm import tqdm

    os.makedirs(ACTS_DIR, exist_ok=True)
    out = _acts_path(split)
    expected_bytes = target_tokens * D_MODEL * 2  # fp16
    if out.exists() and out.stat().st_size >= expected_bytes:
        return out  # already cached

    model, _ = load_base_model()
    device = get_device()
    tokens = load_tokens(split)  # [n_seqs, MAX_SEQ_LEN]

    mm = np.memmap(out, dtype=np.float16, mode="w+", shape=(target_tokens, D_MODEL))
    cursor = 0

    # Forward-hook capture.
    captured: list[torch.Tensor] = []
    def hook(_module, _inp, output):
        # GPTNeoX block output is a tuple (hidden_states, ...); hidden_states is [B, T, D].
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h)

    handle = _residual_hook_target(model).register_forward_hook(hook)

    # Choose a batch size that fits comfortably on a 16GB card.
    # Pythia-160M at seq 512 with bf16 is ~100MB/seq of activation memory,
    # so batch 8 stays well under budget.
    batch_size = 8

    try:
        pbar = tqdm(total=target_tokens, desc=f"activations/{split}", unit="tok")
        for start in range(0, tokens.shape[0], batch_size):
            if cursor >= target_tokens:
                break
            batch_np = tokens[start : start + batch_size]
            batch = torch.from_numpy(batch_np.astype(np.int64)).to(device)
            captured.clear()
            model(batch)
            h = captured[0]  # [B, T, D]
            flat = h.reshape(-1, D_MODEL).to(torch.float16).cpu().numpy()
            n = min(flat.shape[0], target_tokens - cursor)
            mm[cursor : cursor + n] = flat[:n]
            cursor += n
            pbar.update(n)
        pbar.close()
    finally:
        handle.remove()

    mm.flush()
    return out


def prepare_activations() -> None:
    """One-time step: dump residual-stream activations for train + eval splits."""
    _extract_activations("train", TRAIN_TOKENS)
    _extract_activations("eval", EVAL_TOKENS)


def load_activations(split: str) -> np.ndarray:
    """Return memmapped activations shaped [n_tokens, D_MODEL]."""
    path = _acts_path(split)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing activation cache at {path}. Run `uv run prepare.py` first."
        )
    n_tokens = path.stat().st_size // (D_MODEL * 2)
    return np.memmap(path, dtype=np.float16, mode="r", shape=(n_tokens, D_MODEL))


# ------------------------------------------------------------------------------
# Evaluation harness — the ground-truth metric.
#
# The agent's SAE is evaluated on three axes:
#
#   1. ce_loss_delta  — extra nats the base model incurs on held-out tokens when
#      the LAYER-th residual stream is replaced by the SAE reconstruction.
#      This is the PRIMARY metric. Lower is better.
#
#   2. l0             — mean number of non-zero features per token. Constraint:
#      must be <= L0_TARGET or the run is invalid.
#
#   3. dead_fraction  — fraction of SAE features that never fire over the eval
#      stream. Constraint: must be <= DEAD_FRAC_MAX or the run is invalid.
#
# Plus diagnostics: reconstruction MSE, variance explained.
#
# This function is the SAE's val_bpb — the single authoritative scoreboard.
# ------------------------------------------------------------------------------

EVAL_BATCH_SEQS: int = 4  # sequences per forward pass during eval


@torch.no_grad()
def evaluate_sae(sae: torch.nn.Module, eval_seqs: int = 64) -> dict:
    """Evaluate a trained SAE. Returns a dict of metrics.

    The SAE must implement `forward(x) -> (recon, features)` where x and recon
    are [..., D_MODEL] and features is [..., N_FEATURES].

    Args:
        sae: the trained SAE, in eval mode, on the same device as the base model.
        eval_seqs: number of held-out sequences to score. 64 * 512 = 32K tokens,
                   enough for a stable ce_loss_delta estimate.
    """
    model, _ = load_base_model()
    device = get_device()
    sae = sae.to(device).eval()

    tokens = load_tokens("eval")
    eval_seqs = min(eval_seqs, tokens.shape[0])
    tokens = tokens[:eval_seqs]

    # Accumulators.
    ce_clean_sum = 0.0
    ce_patched_sum = 0.0
    n_tokens_scored = 0

    mse_sum = 0.0
    var_sum = 0.0
    act_sq_sum = 0.0
    l0_sum = 0.0
    # Track whether each feature ever fired.
    fired: torch.Tensor | None = None

    # Hook: capture clean residual, replace with SAE reconstruction.
    layer = _residual_hook_target(model)
    mode = {"value": "capture"}  # "capture" | "patch"
    captured: list[torch.Tensor] = []

    def hook(_module, _inp, output):
        h = output[0] if isinstance(output, tuple) else output
        if mode["value"] == "capture":
            captured.append(h)
            return output
        # patch mode: replace h with sae(h).recon
        h32 = h.to(torch.float32)
        recon, feats = sae(h32)
        new_h = recon.to(h.dtype)
        if isinstance(output, tuple):
            return (new_h,) + output[1:]
        return new_h

    handle = layer.register_forward_hook(hook)

    try:
        for start in range(0, eval_seqs, EVAL_BATCH_SEQS):
            batch_np = tokens[start : start + EVAL_BATCH_SEQS]
            batch = torch.from_numpy(batch_np.astype(np.int64)).to(device)

            # ---- Pass 1: clean forward, capture activations, score CE.
            mode["value"] = "capture"
            captured.clear()
            out_clean = model(batch, labels=batch)
            h_clean = captured[0].to(torch.float32)  # [B, T, D]
            ce_clean = out_clean.loss.item()
            B, T, _ = h_clean.shape
            n = B * (T - 1)  # label-shifted CE is over T-1 positions per seq
            ce_clean_sum += ce_clean * n

            # SAE diagnostics on the clean activations.
            flat = h_clean.reshape(-1, D_MODEL)
            recon, feats = sae(flat)
            mse_sum += torch.nn.functional.mse_loss(recon, flat, reduction="sum").item()
            var_sum += (flat - flat.mean(dim=0, keepdim=True)).pow(2).sum().item()
            act_sq_sum += flat.pow(2).sum().item()
            l0_sum += (feats != 0).float().sum(dim=-1).sum().item()
            active = (feats != 0).any(dim=0).detach()  # [N_FEATURES]
            fired = active if fired is None else (fired | active)

            # ---- Pass 2: patched forward, score CE with SAE reconstruction.
            mode["value"] = "patch"
            out_patched = model(batch, labels=batch)
            ce_patched = out_patched.loss.item()
            ce_patched_sum += ce_patched * n
            n_tokens_scored += n
    finally:
        handle.remove()

    ce_clean = ce_clean_sum / max(n_tokens_scored, 1)
    ce_patched = ce_patched_sum / max(n_tokens_scored, 1)
    ce_loss_delta = ce_patched - ce_clean

    total_feats = feats.shape[-1]
    dead_fraction = 1.0 - (fired.float().mean().item() if fired is not None else 0.0)
    l0 = l0_sum / max(n_tokens_scored, 1)
    mse = mse_sum / max(act_sq_sum, 1e-9)  # normalized MSE
    variance_explained = 1.0 - (mse_sum / max(var_sum, 1e-9))

    return {
        "ce_clean": ce_clean,
        "ce_patched": ce_patched,
        "ce_loss_delta": ce_loss_delta,
        "l0": l0,
        "dead_fraction": dead_fraction,
        "mse_normalized": mse,
        "variance_explained": variance_explained,
        "n_features": total_feats,
        "n_tokens_scored": n_tokens_scored,
    }


def validate_constraints(metrics: dict, peak_vram_gb: float) -> tuple[bool, str]:
    """Check hard constraints. Returns (is_valid, reason_if_invalid)."""
    if metrics["l0"] > L0_TARGET:
        return False, f"L0={metrics['l0']:.1f} > L0_TARGET={L0_TARGET}"
    if metrics["dead_fraction"] > DEAD_FRAC_MAX:
        return False, f"dead_fraction={metrics['dead_fraction']:.3f} > {DEAD_FRAC_MAX}"
    if peak_vram_gb > PEAK_VRAM_GB_MAX:
        return False, f"peak_vram_gb={peak_vram_gb:.1f} > {PEAK_VRAM_GB_MAX}"
    return True, ""


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
    print("tokenizing...")
    prepare_tokens()
    print("token shards:")
    for split in ("train", "eval"):
        p = _tokens_path(split)
        print(f"  {split}: {p} ({p.stat().st_size / 1e6:.1f} MB)")
    print("caching activations...")
    prepare_activations()
    print("activation shards:")
    for split in ("train", "eval"):
        p = _acts_path(split)
        print(f"  {split}: {p} ({p.stat().st_size / 1e9:.2f} GB)")
