# automechresearch — agent program

This is an experiment to have you, the LLM, do your own mechanistic interpretability research. You will iterate on a **sparse autoencoder (SAE)** trained on the residual stream of Pythia-160M, layer 8.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr19`). The branch `autosae/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autosae/<tag>` from current `main`.
3. **Read the in-scope files**. The repo is small. Read all of these for full context:
   - `README.md` — repository context.
   - `prepare.py` — constants, model loading, activation caching, **eval harness**. Do not modify.
   - `train_sae.py` — SAE architecture, loss, training loop. **You modify this.**
   - `reading.md` — the SAE literature you should cite when proposing experiments.
4. **Verify the cache exists**: check that `~/.cache/automechresearch/` contains tokens and activations for both `train` and `eval`. If not, tell the user to run `uv run prepare.py`.
5. **Initialize `results.tsv`**: create it with just the header row. The baseline will be recorded after the first run.
6. **Initialize `hypotheses.md` and `journal.md`**: create both as empty files with a one-line header.
7. **Confirm and go**: confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment trains an SAE on a single GPU for a **fixed wall-clock budget of 5 minutes**, then evaluates. You launch it simply as:

```
uv run train_sae.py > run.log 2>&1
```

Do NOT use `tee` or otherwise let the log flood your context.

**What you CAN do:**
- Modify `train_sae.py`. Everything is fair game: the SAE architecture (ReLU, TopK, JumpReLU, Gated, anything new from the literature), the loss (auxiliary losses, ghost grads, dead-neuron revival schemes), the optimizer, the schedule, the batch size, the dict size / expansion factor, the initialization.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed tokenizer, data, activation cache, time budget, and — most importantly — the `evaluate_sae` harness that is the ground truth.
- Install new packages. Only what is already in `pyproject.toml`.
- Modify the evaluation metric or the hard constraints. The `evaluate_sae` function and `validate_constraints` in `prepare.py` are final.

## The metric and the constraints

**Primary metric: `ce_loss_delta`** — the extra cross-entropy (nats per token) the base model incurs when the residual stream at layer 8 is replaced by your SAE's reconstruction, measured on a fixed held-out eval set. **Lower is better.**

**Hard constraints** (violating any → the run is `invalid`, same as `discard`):
- `l0 ≤ 64` (the `L0_TARGET`) — no cheating by becoming dense
- `dead_fraction ≤ 0.10` — no cheating by hiding features
- `peak_vram_gb ≤ 14` — portability guardrail
- Run completes without crashing, within the time budget

**Simplicity criterion**: All else being equal, simpler is better. A small `ce_loss_delta` improvement that adds 50 lines of hacky code is probably not worth it. A `ce_loss_delta` improvement from *deleting* code is a strong win. Keep `train_sae.py` readable.

## Output format

The script prints a summary block at the end like:

```
---
ce_loss_delta:    0.042310
ce_clean:         3.214500
ce_patched:       3.256810
l0:               57.30
dead_fraction:    0.0412
mse_normalized:   0.101200
variance_explained: 0.8991
n_features:       6144
training_seconds: 300.2
peak_vram_mb:     7120.4
peak_vram_gb:     6.95
num_steps:        4800
total_tokens_M:   19.7
expansion:        8
batch_size:       4096
lr:               3.00e-04
l1_coeff:         5.00e-04
valid:            1
```

Extract the key numbers:

```
grep "^ce_loss_delta:\|^l0:\|^dead_fraction:\|^peak_vram_gb:\|^valid:" run.log
```

## Hypothesis-first logging (important!)

**Before every run, you append a short entry to `hypotheses.md`.** Format:

```markdown
## <commit-hash-placeholder> — <one-line summary>

**Hypothesis**: <what you are trying, in plain English, 1-3 sentences>
**Expected**: <what you expect to happen to ce_loss_delta, l0, dead_fraction>
**Falsified if**: <what result would make you abandon this direction>
**Inspired by**: <paper from reading.md, or "ablation of prior run X">
```

After the run, update the first line to include the actual commit hash and the actual outcome.

## Post-run journal

**After every run, you append a one-paragraph writeup to `journal.md`.** Format:

```markdown
## <commit-hash> — <status>

<1-3 sentences: what you changed, what the numbers were, what you think it means. Cite the paper/idea this came from. End with one sentence on what you plan to try next.>
```

This is the morning-reading artifact — write it for a human who will read all the night's entries over coffee.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

Columns (7):

```
commit	ce_loss_delta	l0	dead_frac	vram_gb	status	description
```

- `commit` — short (7 char) hash
- `ce_loss_delta` — 6 decimal places; use `0.000000` for crashes
- `l0` — 1 decimal place
- `dead_frac` — 3 decimal places
- `vram_gb` — 1 decimal place
- `status` — one of `keep`, `discard`, `invalid`, `crash`
- `description` — one-line summary of the experiment

Example:

```
commit	ce_loss_delta	l0	dead_frac	vram_gb	status	description
a1b2c3d	0.042310	57.3	0.041	7.0	keep	baseline relu + l1
b2c3d4e	0.038100	61.2	0.055	7.1	keep	increase expansion 8->16
c3d4e5f	0.091200	10.4	0.003	7.0	discard	l1_coeff 5e-4 -> 5e-3 (too sparse)
d4e5f6g	0.040500	70.1	0.042	7.1	invalid	topk k=80 (l0 > 64)
e5f6g7h	0.000000	0.0	0.000	0.0	crash	OOM at expansion 64
```

## The experiment loop

The loop runs on a dedicated branch (e.g. `autosae/apr19`).

LOOP FOREVER:

1. Look at the git state: the current branch and commit.
2. **Write your hypothesis** to `hypotheses.md` (before touching code).
3. Tune `train_sae.py` with the experimental idea by directly editing the code.
4. `git commit -am "<short description>"`
5. Run the experiment: `uv run train_sae.py > run.log 2>&1`
6. Read out the results:
   `grep "^ce_loss_delta:\|^l0:\|^dead_fraction:\|^peak_vram_gb:\|^valid:" run.log`
7. If the grep is empty, the run crashed — run `tail -n 50 run.log`, decide if the fix is trivial (typo, import, OOM-avoidable), fix and retry once. Otherwise mark `crash`, move on.
8. **Write your journal entry** to `journal.md`.
9. Record the row in `results.tsv` (leave it untracked).
10. Decision rule:
    - If `valid == 1` AND `ce_loss_delta` strictly decreased vs. the current branch best → `keep`: leave the commit in place; this is the new best.
    - If `valid == 1` AND `ce_loss_delta` equal or worse → `discard`: `git reset --hard HEAD~1`.
    - If `valid == 0` → `invalid`: `git reset --hard HEAD~1`.
    - If crashed and unrecoverable → `crash`: `git reset --hard HEAD~1`.
11. Continue.

**Timeout**: each experiment should take ~5 min training + ~1 min eval. If a run exceeds 10 minutes, kill it and treat it as `crash`.

**Crashes**: OOMs and simple bugs are worth one retry. Fundamentally broken ideas just get marked `crash` and skipped.

**NEVER STOP**: once the experiment loop has begun, do NOT pause to ask the user if you should continue. Do NOT ask "should I keep going?" The user is likely asleep and expects you to run autonomously until they manually stop you. If you run out of ideas, **re-read `reading.md`**, re-read the summary blocks of your own best runs, try combining two near-misses, try a more radical architectural swap. The loop runs until the human interrupts, period.

## Ideas to explore (non-exhaustive, not an ordered list)

- Sparsity-regime changes: TopK (fix exactly k=32 active), JumpReLU, Gated.
- Auxiliary losses: ghost grads / auxiliary reconstruction on dead features (Jermyn+2024).
- Initialization: scale of b_dec to the activation mean; decoder init direction.
- Optimizer: AdamW vs. Adam, betas, weight decay, LR warmup/cosine.
- Dict size / expansion factor: 4, 8, 16, 32. More features is not always better — at fixed L0 you trade coverage for specificity.
- Batch size: larger batches reduce gradient noise; smaller batches often help SAE training.
- Activation normalization: divide input by running-mean L2 norm, as in Anthropic's scaling paper.
- Decoder constraints: unit-norm columns (already in baseline) vs. no normalization vs. weight decay.

When proposing any of these, cite the paper or idea from `reading.md`.

As an example use case: the user leaves you running overnight. Each experiment takes ~6 min total, so ~10/hour → ~80 overnight. They wake up to a TSV of 80 experiments, a curated journal, and the best branch-tip pointing at an SAE measurably better than the baseline.
