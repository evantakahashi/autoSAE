"""viz.tour — generate tour.html, an interactive tour of the base model.

The tour is independent of the SAE training loop. It runs the frozen Pythia-160M
once and produces a self-contained HTML file with four sections:

  1. Induction-head scores      (Olsson+2022 prefix-matching test)
  2. Attention pattern viewer   (all heads, one example prompt)
  3. Logit lens                 (per-layer top-K next-token predictions)
  4. Residual stream norms      (layer-wise ||h|| on a handful of sentences)

Run:

    uv run python -m viz.tour

Output lands at `tour.html` in the repo root (gitignored).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from prepare import LAYER, MODEL_NAME
from viz.attention import DEFAULT_PROMPT as ATTN_PROMPT, capture_attention
from viz.induction import induction_scores
from viz.logit_lens import DEFAULT_PROMPT as LENS_PROMPT, logit_lens
from viz.render import render_tour
from viz.resid_norms import DEFAULT_TEXTS, residual_norms


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tour.html for the base model.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tour.html"),
        help="Output HTML path (default: tour.html)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=50,
        help="Length of the repeated random-token suffix for induction-head test.",
    )
    parser.add_argument(
        "--n-seqs",
        type=int,
        default=16,
        help="Number of random sequences for induction-head averaging.",
    )
    args = parser.parse_args()

    t0 = time.time()
    print(f"[tour] induction scores ({args.n_seqs} seqs x {args.seq_len} tokens)...")
    ind = induction_scores(n_seqs=args.n_seqs, seq_len=args.seq_len)

    print("[tour] capturing attention patterns...")
    attn, attn_tokens = capture_attention(ATTN_PROMPT)

    print("[tour] logit lens...")
    lens_tokens, lens = logit_lens(LENS_PROMPT)

    print("[tour] residual stream norms...")
    norms = residual_norms(DEFAULT_TEXTS)

    print(f"[tour] rendering -> {args.out}")
    render_tour(
        out_path=args.out,
        model_name=MODEL_NAME,
        sae_layer=LAYER,
        induction_scores=ind,
        induction_n_seqs=args.n_seqs,
        induction_seq_len=args.seq_len,
        attn_prompt=ATTN_PROMPT,
        attn=attn,
        attn_tokens=attn_tokens,
        lens_prompt=LENS_PROMPT,
        lens_tokens=lens_tokens,
        lens=lens,
        resid_norms=norms,
        resid_n_texts=len(DEFAULT_TEXTS),
    )
    print(f"[tour] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
