# reading.md — SAE literature the agent should cite

Curated pointers for the agent. When proposing an experiment in `hypotheses.md`, cite the paper (by short tag, e.g. `Bricken+2023`) it draws from. For the human reader, this doubles as a guided tour of the modern SAE literature.

## Foundational

- **Bricken+2023 — "Towards Monosemanticity"** (Anthropic, transformer-circuits.pub).
  The paper that kicked off modern SAEs on LLMs. ReLU + L1, pre-encoder centering bias, unit-norm decoder columns. Our baseline is this, minus the extras.
- **Cunningham+2023 — "Sparse Autoencoders Find Highly Interpretable Features in Language Models"**.
  Concurrent work; introduced the SAE-on-Pythia recipe that a lot of later tooling (incl. SAELens) was built on.

## Architectural variants

- **Rajamanoharan+2024 — "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU SAEs"** (DeepMind).
  Replace ReLU with JumpReLU: a learnable per-feature threshold so features either fire at a meaningful magnitude or stay off. Better loss-recovered at fixed L0.
- **Makelov+2024 / Rajamanoharan+2024 — "Gated SAEs"** (DeepMind).
  Decouple the "does this feature fire?" gate from the "how much?" magnitude. Typically better Pareto curves than vanilla ReLU+L1.
- **Gao+2024 — "Scaling and evaluating sparse autoencoders"** (OpenAI).
  TopK SAEs: enforce exactly k active features per token, kill L1 entirely. Much easier to tune; directly targets L0. Also proposes the `auxk` auxiliary loss for dead-feature revival.
- **Templeton+2024 — "Scaling Monosemanticity"** (Anthropic).
  Scaling laws for SAEs. Mostly a qualitative read, but useful for thinking about dict-size tradeoffs.

## Training tricks

- **Jermyn+2024 — "Ghost grads"** (Anthropic update).
  Feeds a scaled-down gradient to dead features so they have a chance to revive. An alternative to `auxk`.
- **Anthropic "SAE training details" April 2024 update**.
  The activation-normalization trick (divide by running RMS norm of the input so loss is scale-free).
- **Bloom, SAELens**.
  Reference implementation (do not import — read for sanity checking). Has well-tuned defaults for Pythia/GPT-2.

## Evaluation and interpretation

- **Karvonen+2024 — "SAEBench"**.
  A battery of downstream evals. We only use one scalar (`ce_loss_delta`) in this repo to stay autoresearch-shaped, but if you ever relax that constraint, SAEBench is the frontier.
- **Bills+2023 — "Language models can explain neurons in language models"** (OpenAI).
  The auto-interp framework. Useful reading if we later extend AutoSAE to score feature interpretability as part of the loop.

## Related MI background (read if you have time)

- **Elhage+2022 — "Toy Models of Superposition"**.
  Why SAEs exist at all: features in superposition, and how linear dictionaries can in principle recover them.
- **Olah+2020 — "Zoom In: An Introduction to Circuits"**.
  The broader MI agenda SAEs feed into.

---

**How to cite in your experiments**: in `hypotheses.md`, use `**Inspired by**: Gao+2024 (TopK)` or `**Inspired by**: ablation of Bricken+2023 pre-decoder bias` etc. Short tags, no URLs — we're not writing a bibliography, we're leaving breadcrumbs.
