"""bench.compare — run a handful of SAE variants and dump bench/benchmark.tsv.

Each variant trains for TIME_BUDGET_SEC and is evaluated with the same
`prepare.evaluate_sae` harness as a normal agent run.

Usage:
    uv run python -m bench.compare
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
from bench.models import ReluSAE, TopKAuxSAE, TopKSAE


BENCH_DIR = Path(__file__).parent
TSV_PATH = BENCH_DIR / "benchmark.tsv"


def activation_loader(batch_size: int, device: torch.device, seed: int = SEED):
    acts = load_activations("train")
    n = acts.shape[0]
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        for start in range(0, n - batch_size, batch_size):
            idx = perm[start : start + batch_size]
            batch = np.asarray(acts[idx], dtype=np.float32)
            yield torch.from_numpy(batch).to(device, non_blocking=True)


@dataclass
class Variant:
    name: str
    make: Callable[[], nn.Module]
    loss_fn: Callable
    unit_norm_decoder: bool = True
    batch_size: int = 4096
    lr: float = 3e-4


def relu_loss(sae, x, recon, features, l1_coeff: float):
    mse = F.mse_loss(recon, x)
    dec_norms = sae.W_dec.norm(dim=-1)
    l1 = (features * dec_norms[None, :]).abs().sum(dim=-1).mean()
    return mse + l1_coeff * l1, {"mse": mse.item(), "l1": l1.item()}


def topk_loss(sae, x, recon, features, _):
    # TopK enforces sparsity structurally; just MSE.
    mse = F.mse_loss(recon, x)
    return mse, {"mse": mse.item(), "l1": 0.0}


VARIANTS: list[Variant] = [
    Variant(
        name="relu_l1",
        make=lambda: ReluSAE(D_MODEL, expansion=8),
        loss_fn=lambda sae, x, r, f: relu_loss(sae, x, r, f, l1_coeff=3e-3),
    ),
    Variant(
        name="topk_k32",
        make=lambda: TopKSAE(D_MODEL, expansion=8, k=32),
        loss_fn=lambda sae, x, r, f: topk_loss(sae, x, r, f, None),
    ),
    Variant(
        name="topk_k48",
        make=lambda: TopKSAE(D_MODEL, expansion=8, k=48),
        loss_fn=lambda sae, x, r, f: topk_loss(sae, x, r, f, None),
    ),
    Variant(
        name="topk_k24_aux",
        make=lambda: TopKAuxSAE(D_MODEL, expansion=8, k=24, k_aux=512),
        loss_fn=None,
    ),
    Variant(
        name="topk_k32_aux",
        make=lambda: TopKAuxSAE(D_MODEL, expansion=8, k=32, k_aux=512),
        loss_fn=None,
    ),
    Variant(
        name="topk_k48_aux",
        make=lambda: TopKAuxSAE(D_MODEL, expansion=8, k=48, k_aux=512),
        loss_fn=None,  # special-cased in run_variant
    ),
    Variant(
        name="topk_k64_aux",
        make=lambda: TopKAuxSAE(D_MODEL, expansion=8, k=64, k_aux=512),
        loss_fn=None,
    ),
    Variant(
        name="topk_k48_aux_exp16",
        make=lambda: TopKAuxSAE(D_MODEL, expansion=16, k=48, k_aux=1024),
        loss_fn=None,
    ),
]


def run_variant(v: Variant) -> dict:
    torch.manual_seed(SEED)
    device = get_device()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    sae = v.make().to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=v.lr, betas=(0.9, 0.999))
    loader = activation_loader(v.batch_size, device)
    use_aux = isinstance(sae, TopKAuxSAE)
    # Gao+2024 suggests aux_coef = 1/32 when K=32; scales roughly with k.
    aux_coef = 1.0 / 32.0

    t0 = time.time()
    step = 0
    last_log = t0
    last_aux_info = (0.0, 0)
    while time.time() - t0 < TIME_BUDGET_SEC:
        x = next(loader)
        if use_aux:
            recon, feat, aux_loss, n_dead = sae.forward_with_aux(x)
            mse = F.mse_loss(recon, x)
            loss = mse + aux_coef * aux_loss
            aux = {"mse": mse.item(), "l1": float(aux_loss.item())}
            last_aux_info = (float(aux_loss.item()), n_dead)
        else:
            recon, feat = sae(x)
            loss, aux = v.loss_fn(sae, x, recon, feat)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if v.unit_norm_decoder:
            with torch.no_grad():
                sae.W_dec.div_(sae.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        opt.step()
        step += 1
        if step % 500 == 0:
            now = time.time()
            rate = 500 * v.batch_size / max(now - last_log, 1e-9)
            last_log = now
            tail = ""
            if use_aux:
                tail = f" | aux {last_aux_info[0]:.4f} | dead {last_aux_info[1]}"
            print(
                f"  [{v.name}] step {step:6d} | t {now-t0:5.1f}s | "
                f"loss {loss.item():.4f} | mse {aux['mse']:.4f} | "
                f"{rate/1e6:.2f} Mtok/s{tail}",
                flush=True,
            )
    training_seconds = time.time() - t0

    sae.eval()
    metrics = evaluate_sae(sae)
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    )
    peak_vram_gb = peak_vram_mb / 1024.0
    is_valid, reason = validate_constraints(metrics, peak_vram_gb)

    row = {
        "variant": v.name,
        "ce_loss_delta": metrics["ce_loss_delta"],
        "ce_clean": metrics["ce_clean"],
        "ce_patched": metrics["ce_patched"],
        "l0": metrics["l0"],
        "dead_fraction": metrics["dead_fraction"],
        "mse_normalized": metrics["mse_normalized"],
        "variance_explained": metrics["variance_explained"],
        "peak_vram_gb": peak_vram_gb,
        "training_seconds": training_seconds,
        "steps": step,
        "valid": int(is_valid),
        "invalid_reason": reason or "",
    }
    return row


def _load_existing() -> list[dict]:
    if not TSV_PATH.exists():
        return []
    with open(TSV_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", help="Run only these variant names; keep others from tsv.")
    args = p.parse_args()

    existing = _load_existing() if args.only else []
    keep = {r["variant"]: r for r in existing if r["variant"] not in (args.only or [])}

    rows: list[dict] = list(keep.values())
    to_run = [v for v in VARIANTS if (not args.only or v.name in args.only)]
    for v in to_run:
        print(f"[bench] running variant: {v.name}")
        row = run_variant(v)
        print(
            f"[bench] {v.name}: ce_loss_delta={row['ce_loss_delta']:.4f} "
            f"L0={row['l0']:.1f} dead={row['dead_fraction']:.2%} "
            f"valid={row['valid']}"
        )
        rows.append(row)
        # Free between variants.
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Preserve VARIANTS declaration order in the tsv.
    order = {v.name: i for i, v in enumerate(VARIANTS)}
    rows.sort(key=lambda r: order.get(r["variant"], 1_000_000))

    if rows:
        fields = list(rows[0].keys())
        with open(TSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[bench] wrote {TSV_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
