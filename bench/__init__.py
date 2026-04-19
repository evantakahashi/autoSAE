"""bench — scripted demo sweep for the README.

This is deliberately *separate* from `train_sae.py`. The agent's workflow
touches `train_sae.py`; this package runs multiple variants in one invocation
so we can generate a comparison chart. Each variant uses the same
TIME_BUDGET_SEC and evaluate_sae harness as a normal agent run — the only
thing that differs is the SAE architecture.
"""
