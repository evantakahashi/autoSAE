"""viz.tour — generate tour.html, an interactive tour of the base model.

The tour is independent of the SAE training loop. It runs the frozen Pythia-160M
once and produces a self-contained HTML file with four sections:

  1. Induction-head scores      (Olsson+2022 prefix-matching test)
  2. Attention pattern viewer   (all heads, one example prompt)
  3. Logit lens                 (per-layer top-K next-token predictions)
  4. Residual stream norms      (layer-wise ||h|| on the eval set)

Run:

    uv run python -m viz.tour

Output lands at `tour.html` in the repo root (gitignored).
"""

from __future__ import annotations

import argparse
from pathlib import Path


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

    print(f"[tour] writing to {args.out}")
    print("[tour] not yet implemented — sections will land in subsequent commits.")


if __name__ == "__main__":
    main()
