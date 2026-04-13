# automechresearch

> Autonomous overnight research loop for **mechanistic interpretability**, inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

The idea: give an AI agent a small, real mech-interp setup and let it experiment autonomously overnight. The agent modifies a single `train_sae.py`, trains a sparse autoencoder for 5 minutes, measures whether the reconstruction got better under sparsity/dead-feature constraints, and either keeps or discards the change. You wake up to a TSV of experiments, a `journal.md` writeup of what happened, and (hopefully) a better SAE.

You are not touching Python files like a normal researcher. You are programming `program.md` — the instructions that configure your autonomous research org.

## The big idea: "val_bpb" for SAEs

The autoresearch pattern only works if every experiment produces **one scalar, comparable across architectures**. The SAE analog:

- **Primary metric: `ce_loss_delta`** — the extra cross-entropy (in nats, on a fixed eval set) that the base model incurs when the residual-stream activation at the target layer is replaced with the SAE's reconstruction. Lower = better.
- **Hard constraints** (violation ⇒ `discard`):
  - `L0 ≤ L0_target` — no cheating by becoming dense
  - `dead_fraction ≤ 0.10` — no cheating by hiding features
  - `peak_vram_gb ≤ 14` — portability guardrail
  - Run completes within the time budget

Everything else — architecture, optimizer, loss, dict size, initialization — is fair game for the agent.

## Layout

```
prepare.py       — constants, model loading, activation caching, eval harness (do NOT modify)
train_sae.py     — SAE architecture + training loop (agent modifies this)
program.md       — agent instructions (human modifies this)
reading.md       — SAE literature pointers the agent should cite
hypotheses.md    — appended by the agent before each run
journal.md       — appended by the agent after each run
results.tsv      — one row per experiment (gitignored)
```

## Quickstart

See [program.md](program.md) for the agent-facing instructions. Full quickstart coming once `prepare.py` and `train_sae.py` land.
