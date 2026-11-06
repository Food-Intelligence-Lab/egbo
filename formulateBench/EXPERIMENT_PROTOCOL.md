# Frozen FormulateBench experimental protocol

This file records the protocol to use for the final FormulateBench rerun. It
is intended to prevent configuration drift between the paper, runners, and
released results.

## Tasks and objective

- 24 product categories: 14 meat and 10 dairy.
- A candidate is a nonnegative mixture over the eligible ingredient pool;
  mixture fractions sum to one.
- The hidden loss is z-score-normalized RMSE to the category's animal-product
  target over the reported subset of eight nutrients. Lower is better.
- Total budget: 20 objective evaluations; sequential batch size: 1.

## Common ingredient pools

Rows that are animal-derived, ambiguous animal fat, or directly reveal a
target category are removed before *every* LLM call and optimization run.

- Meat exclusions: `egg white`, `egg whites`, `fat`, `milk`, `milk proteins`.
- Dairy exclusions: `mozzarella`.

After filtering, every method for a domain receives the same eligible pool.
Full-space methods may use every eligible ingredient; an expert-selected
support is an EGBO intervention, not a constraint imposed on full-space
baselines.

## LLM information and shared first evaluation

- Model: Claude Opus 4.7; temperature 0.
- The LLM sees the target product category, eligible ingredient names, the
  instruction to optimize overall similarity, and the evaluation budget.
- It does not see nutrient profiles, the target nutrient vector, or the form
  of the hidden objective.
- After evaluation, iterative LLM methods see evaluated recipes and only the
  scalar target-dissimilarity score.
- The LLM's first recipe uses at most 9 ingredients and counts as evaluation
  1. One-shot EGBO, adaptive EGBO, fixed-support iterative LLM, and
  unrestricted iterative LLM share this exact first recipe.

## Initialization and evaluation accounting

- Conventional BO methods use 5 globally generated initial design points.
- EGBO uses the shared LLM recipe plus 4 seeded local logit-normal
  perturbations, for `n_init=5`; BO acquisition begins at evaluation 6.
- Iterative LLM baselines begin proposing feedback-conditioned recipes at
  evaluation 2.
- Every trajectory contains exactly 20 evaluated recipes. The initial recipe
  is never treated as a free evaluation.

## EGBO variants

- **EGBO, default BO:** the expert fixes the support through its first recipe;
  concentration optimization uses the native BoTorch 0.18.1
  `SingleTaskGP` defaults with qLogNEI.
- **EGBO, LLM-configured BO:** support and initialization are identical, but a
  single global BO configuration elicited once from Claude Opus 4.7 is used
  for every category: Matérn-5/2, default lengthscale prior, default/learned
  noise, qLogNEI, and ARD.
- **Adaptive EGBO (Q=2):** uses the LLM-configured BO setup. After evaluation
  10, the expert sees evaluated recipes and scalar feedback and may add at
  most 2 ingredients. The active set grows monotonically; old recipes have
  zero concentration on newly added ingredients.

## Comparison methods

- Random search.
- Full-space vanilla BO.
- Random-subset BO with a fixed random support of size 9.
- REMBO with latent dimension 9.
- TuRBO, SAASBO, and SEBO.
- Fixed-support iterative LLM: may change concentrations but not the initial
  ingredient set.
- Unrestricted iterative LLM: may change both ingredients and concentrations.
- Constrained NNLS oracle on the expert-selected support and constrained NNLS
  oracle on the full eligible pool. These are diagnostic floors, not methods
  receiving the 20-evaluation budget.

## Replication and aggregation

- EGBO and conventional stochastic optimizers: 10 seeds per category.
- Temperature-zero iterative LLM methods: one cached trajectory per category.
- For cross-category aggregation, each raw trajectory is divided by that
  category's mean random-search loss at evaluation 1.
- Main curves report category-level mean best-so-far normalized loss.
- Shaded regions are 95% confidence intervals across category-level mean
  trajectories, not across pooled seed-by-category observations.

## Status

The protocol is frozen and the final rerun is complete. The release uses only
the `final_protocol_*` result groups plus the frozen semantic LLM trajectories
in `meat_semantic_all14_v1` and `dairy_semantic_all10_v3`. Exploratory and
partial checkpoints are not paper results.
