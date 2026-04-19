"""bench.plot — render bench/compare.png from bench/benchmark.tsv.

Produces a two-panel figure:
  - Left:  bar chart of ce_loss_delta per variant (lower = better)
  - Right: scatter of (L0, ce_loss_delta) — the Pareto view

Usage:
    uv run python -m bench.plot
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


BENCH_DIR = Path(__file__).parent
TSV_PATH = BENCH_DIR / "benchmark.tsv"
OUT_PATH = BENCH_DIR / "compare.png"


def load_rows() -> list[dict]:
    with open(TSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        r["ce_loss_delta"] = float(r["ce_loss_delta"])
        r["l0"] = float(r["l0"])
        r["dead_fraction"] = float(r["dead_fraction"])
        r["valid"] = int(r["valid"])
    return rows


def main() -> None:
    rows = load_rows()
    names = [r["variant"] for r in rows]
    deltas = [r["ce_loss_delta"] for r in rows]
    l0s = [r["l0"] for r in rows]
    valids = [bool(r["valid"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # --- Left: bar chart of ce_loss_delta (lower is better)
    colors = ["#2b6cb0" if v else "#aaaaaa" for v in valids]
    bars = ax1.bar(names, deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax1.tick_params(axis="x", labelrotation=35)
    for lbl in ax1.get_xticklabels():
        lbl.set_horizontalalignment("right")
    ax1.set_ylabel("ce_loss_delta (nats/token) — lower is better")
    ax1.set_title("Reconstruction fidelity per variant")
    ax1.grid(axis="y", alpha=0.3)
    for b, d in zip(bars, deltas):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{d:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    # Mark baseline with an annotation arrow to the best.
    if len(deltas) > 1:
        best_i = min(range(len(deltas)), key=lambda i: deltas[i] if valids[i] else float("inf"))
        if valids[best_i] and best_i != 0:
            improvement = deltas[0] - deltas[best_i]
            pct = 100 * improvement / deltas[0] if deltas[0] else 0
            ax1.annotate(
                f"-{improvement:.3f} ({pct:.0f}%)",
                xy=(best_i, deltas[best_i]),
                xytext=(best_i, deltas[0]),
                arrowprops=dict(arrowstyle="->", color="#c05621"),
                color="#c05621",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

    # --- Right: Pareto curve (AuxK variants connected by K order) + other variants scattered.
    aux_rows = [r for r in rows if r["variant"].endswith("_aux") and r["valid"]]
    aux_rows.sort(key=lambda r: r["l0"])
    if len(aux_rows) >= 2:
        ax2.plot(
            [r["l0"] for r in aux_rows],
            [r["ce_loss_delta"] for r in aux_rows],
            "-",
            color="#2b6cb0",
            linewidth=2,
            alpha=0.7,
            zorder=2,
            label="TopK+AuxK Pareto",
        )
    for n, l0, d, ok in zip(names, l0s, deltas, valids):
        ax2.scatter(
            l0,
            d,
            s=140,
            c="#2b6cb0" if ok else "#aaaaaa",
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        ax2.annotate(
            n,
            (l0, d),
            textcoords="offset points",
            xytext=(7, 6),
            fontsize=9,
        )
    ax2.axvline(64, color="#c05621", linestyle="--", alpha=0.7, label="L0 ceiling = 64")
    ax2.set_xlabel("L0 (mean active features / token)")
    ax2.set_ylabel("ce_loss_delta (nats/token)")
    ax2.set_title("Sparsity / fidelity Pareto")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="best")

    fig.suptitle(
        "automechresearch — SAE variant sweep on Pythia-160M (layer 8, 5-min budget)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
    print(f"[plot] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
