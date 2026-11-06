"""Hartmann6 embedded-in-N-dimensions: simulated EGBO vs baselines.

Hartmann6 (d=6, known maximum f* = 3.32237) is embedded in N >> 6 ambient
dimensions by appending N - 6 inert variables. The simulated expert
selects variables for the active set independently across coordinates,
each relevant variable with probability p (recall) and each irrelevant
variable with probability q (false-positive rate).

This reproduces the synthetic results in the paper (Fig 1, one-shot
Q=1; Fig 6, adaptive Q=T).

Methods compared:
    random        - random search in [0, 1]^N
    vanilla       - BO over the full N-dim domain
    oracle        - BO restricted to the 6 relevant variables (p=1, q=0)
    rembo         - random embedding to k=6 (Wang et al. 2016)
    random_subset - BO on a random fixed 6-d subset, inactive dims at 0
    turbo         - trust-region BO over the full N-dim domain
    saasbo        - SAAS prior over the full N-dim domain
    sebo          - SAAS + smooth-L0 homotopy over the full N-dim domain
    egbo_strong   - simulated expert with p=0.9, q=0.01
    egbo_weak     - simulated expert with p=0.2, q=0.10

Each EGBO method runs in either one-shot (Q=1, expert queried before BO
starts) or adaptive mode, selectable via --expert-mode.

NOTE on "Q=T". Two adaptive schedules exist and they are NOT identical:
  * --expert-mode adaptive (n_rounds unset): the expert is queried before
    every one of the T evaluations, including the n_init Sobol points
    (T = 20 queries). Used for Fig 1(b) and the (p, q) heatmap (Fig 6).
  * --n-rounds Q with Q >= T - n_init + 1: the expert is queried once before
    the first evaluation and then before each BO step (T - n_init + 1 = 16
    queries at T=20, n_init=5). Used for the Q-sweep (Figs 7-8, Q=16 and Q=20).
With identical seeds these give different results, so report which one a
figure uses.

Usage:
    python run_hartmann6.py --expert-mode oneshot --name release_oneshot
    python run_hartmann6.py --expert-mode adaptive --name release_adaptive
    # smoke test
    python run_hartmann6.py --budget 6 --seeds 2 --n-init 2 \
        --methods random vanilla oracle egbo_strong --name smoke
"""

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction, qLogNoisyExpectedImprovement
from botorch.fit import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
try:
    from botorch.models import SaasFullyBayesianSingleTaskGP, SingleTaskGP
except ImportError:  # botorch >= 0.12 relocated the fully-Bayesian model
    from botorch.models import SingleTaskGP
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine


# ---------------------------------------------------------------------------
# Hartmann6 (the target function)
# ---------------------------------------------------------------------------

D_RELEVANT = 6
HARTMANN6_GLOBAL_MAX = 3.32237  # achieved at x ≈ (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)

_HARTMANN6_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_HARTMANN6_A = np.array([
    [10.0,  3.0, 17.0,  3.5,  1.7,  8.0],
    [ 0.05, 10.0, 17.0,  0.1,  8.0, 14.0],
    [ 3.0,  3.5,  1.7, 10.0, 17.0,  8.0],
    [17.0,  8.0,  0.05, 10.0,  0.1, 14.0],
])
_HARTMANN6_P = 1e-4 * np.array([
    [1312, 1696, 5569,  124, 8283, 5886],
    [2329, 4135, 8307, 3736, 1004, 9991],
    [2348, 1451, 3522, 2883, 3047, 6650],
    [4047, 8828, 8732, 5743, 1091,  381],
])


def hartmann6(x: np.ndarray) -> float:
    """Hartmann6 on [0, 1]^6, oriented so larger is better (max ≈ 3.3224)."""
    delta = x[None, :] - _HARTMANN6_P
    inner = np.sum(_HARTMANN6_A * delta ** 2, axis=1)
    return float(np.sum(_HARTMANN6_ALPHA * np.exp(-inner)))


def make_embedded_objective(d_total: int) -> Callable[[np.ndarray], float]:
    """Return f(x_full) = hartmann6(x_full[:6]) for x_full in [0, 1]^d_total."""
    def f(x_full: np.ndarray) -> float:
        return hartmann6(x_full[:D_RELEVANT])
    return f


# ---------------------------------------------------------------------------
# Simulated expert
# ---------------------------------------------------------------------------

@dataclass
class ExpertConfig:
    name: str
    p: float  # per-iteration recall: P(add a relevant variable not yet active)
    q: float  # per-iteration false-positive rate: P(add an irrelevant variable)
    # Coordinates are proposed in contiguous blocks that share a single
    # Bernoulli draw (all-or-none inclusion). Block size 1 (the default)
    # reproduces the original independent per-coordinate behavior exactly.
    # Block size > 1 injects positive within-block correlation while leaving
    # each coordinate's marginal inclusion probability unchanged (matched
    # marginals), so E[|S|] = kappa*p + (N-kappa)*q is invariant to it.
    rel_block_size: int = 1
    irr_block_size: int = 1


EXPERT_PRESETS: Dict[str, ExpertConfig] = {
    "egbo_strong": ExpertConfig("egbo_strong", p=0.9,  q=0.01),
    "egbo_weak":   ExpertConfig("egbo_weak",   p=0.2,  q=0.10),
    # Correlated-expert robustness (Experiment B): same marginals (p, q) as
    # above, but relevant coordinates are added in all-or-none blocks of 3
    # (models e.g. an emulsifier pulling in its complementary stabilizer).
    "egbo_strong_blocked": ExpertConfig(
        "egbo_strong_blocked", p=0.9, q=0.01, rel_block_size=3, irr_block_size=1),
    "egbo_weak_blocked": ExpertConfig(
        "egbo_weak_blocked", p=0.2, q=0.10, rel_block_size=3, irr_block_size=1),
}

# `egbo_custom` is a sentinel method that uses CLI-provided (p, q) instead of
# a fixed preset. Created at runtime in run_method() if --expert-p/--expert-q
# are set; not stored in EXPERT_PRESETS so it doesn't appear in default sweeps.
_EGBO_CUSTOM_KEY = "egbo_custom"


def _contiguous_blocks(lo: int, hi: int, block_size: int) -> List[List[int]]:
    """Partition [lo, hi) into consecutive blocks of `block_size` (last may be smaller)."""
    bs = max(1, int(block_size))
    return [list(range(s, min(s + bs, hi))) for s in range(lo, hi, bs)]


def expert_step(
    active: set, d_total: int, config: ExpertConfig, rng: np.random.RandomState,
) -> set:
    """One expert query: propose missing relevant/irrelevant variables in blocks.

    Each contiguous block shares a single Bernoulli draw (prob p for relevant
    blocks, q for irrelevant), so its coordinates are added together or not at
    all. With block size 1 (the default) each coordinate is drawn independently,
    reproducing the original per-coordinate behavior bit-for-bit (same number of
    draws, same order). Block size > 1 injects positive within-block correlation
    at matched marginals: every coordinate is still included with probability p
    (resp. q), so only the dependence structure changes.
    """
    new_active = set(active)
    for block in _contiguous_blocks(0, D_RELEVANT, config.rel_block_size):
        missing = [v for v in block if v not in new_active]
        if missing and rng.random() < config.p:
            new_active.update(missing)
    for block in _contiguous_blocks(D_RELEVANT, d_total, config.irr_block_size):
        missing = [v for v in block if v not in new_active]
        if missing and rng.random() < config.q:
            new_active.update(missing)
    return new_active


# ---------------------------------------------------------------------------
# BO building blocks
# ---------------------------------------------------------------------------

def _fit_with_warning(mll, context: str) -> bool:
    try:
        fit_gpytorch_mll(mll)
        return True
    except Exception as exc:
        warnings.warn(
            f"{context}: GP fit failed ({type(exc).__name__}: {exc}). "
            "Falling back to a random candidate.",
            RuntimeWarning, stacklevel=2,
        )
        return False


def _suggest_qlnei(gp: SingleTaskGP, train_X_norm: torch.Tensor, d: int) -> torch.Tensor:
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
    acq = qLogNoisyExpectedImprovement(model=gp, X_baseline=train_X_norm, sampler=sampler)
    cand, _ = optimize_acqf(
        acq_function=acq,
        bounds=torch.stack([torch.zeros(d), torch.ones(d)]).double(),
        q=1, num_restarts=10, raw_samples=512,
    )
    return cand.squeeze(0)


# ---------------------------------------------------------------------------
# Method implementations — each returns {best_values, all_y, final_best, ...}
# ---------------------------------------------------------------------------

def run_random(eval_fn, d_total: int, budget: int, n_init: int, seed: int) -> Dict:
    rng = np.random.RandomState(seed)
    all_y, best_values = [], []
    for _ in range(budget):
        x = rng.random(d_total)
        y = eval_fn(x)
        all_y.append(y)
        best_values.append(max(all_y))
    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def run_vanilla_bo(eval_fn, d_total: int, budget: int, n_init: int, seed: int) -> Dict:
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    d = d_total
    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    x_init = sobol.draw(n_init).double().numpy()

    all_x, all_y, best_values = [], [], []
    for i in range(n_init):
        x = x_init[i]
        y = eval_fn(x)
        all_x.append(x)
        all_y.append(y)
        best_values.append(max(all_y))

    for step in range(budget - n_init):
        train_X = torch.tensor(np.array(all_x), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        if not _fit_with_warning(mll, f"vanilla_bo(seed={seed}, step={step})"):
            x_new = rng.random(d)
        else:
            x_new = _suggest_qlnei(gp, train_X, d).detach().numpy()
        y_new = eval_fn(x_new)
        all_x.append(x_new)
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def run_subspace_bo(
    eval_fn, d_total: int, budget: int, n_init: int, seed: int,
    active_indices: Sequence[int],
) -> Dict:
    """BO on a fixed subset of variables, inactive dims fixed at 0."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    active = sorted(active_indices)
    k = len(active)
    if k == 0:
        return run_random(eval_fn, d_total, budget, n_init, seed)

    def lift(x_sub: np.ndarray) -> np.ndarray:
        # Inactive coordinates pinned at 0 to match the supp(x) ⊆ S
        # convention used in the paper's Assumption 1 / Theorem 1.
        x_full = np.zeros(d_total)
        x_full[active] = x_sub
        return x_full

    # One scrambled Sobol stream per init iter (seed + i). Matches run_egbo's
    # init scheme so Oracle and EGBO are apples-to-apples comparable.
    all_x_sub, all_y, best_values = [], [], []
    for i in range(n_init):
        sobol_i = SobolEngine(dimension=k, scramble=True, seed=seed + i)
        x_sub = sobol_i.draw(1).double().squeeze(0).numpy()
        y = eval_fn(lift(x_sub))
        all_x_sub.append(x_sub)
        all_y.append(y)
        best_values.append(max(all_y))

    for step in range(budget - n_init):
        train_X = torch.tensor(np.array(all_x_sub), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        if not _fit_with_warning(mll, f"subspace_bo(seed={seed}, k={k}, step={step})"):
            x_sub = rng.random(k)
        else:
            x_sub = _suggest_qlnei(gp, train_X, k).detach().numpy()
        y_new = eval_fn(lift(x_sub))
        all_x_sub.append(x_sub)
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {
        "best_values": best_values,
        "all_y": all_y,
        "final_best": max(all_y),
        "active_indices": list(active),
    }


def run_oracle(eval_fn, d_total: int, budget: int, n_init: int, seed: int) -> Dict:
    return run_subspace_bo(
        eval_fn, d_total, budget, n_init, seed,
        active_indices=list(range(D_RELEVANT)),
    )


def run_random_subset(eval_fn, d_total: int, budget: int, n_init: int, seed: int,
                       k: int = D_RELEVANT) -> Dict:
    rng = np.random.RandomState(seed + 10_000)
    active = sorted(rng.choice(d_total, size=k, replace=False).tolist())
    res = run_subspace_bo(eval_fn, d_total, budget, n_init, seed, active_indices=active)
    res["n_relevant_in_subset"] = int(len(set(range(D_RELEVANT)).intersection(active)))
    return res


def run_rembo(eval_fn, d_total: int, budget: int, n_init: int, seed: int,
               target_dim: int = D_RELEVANT) -> Dict:
    torch.manual_seed(seed)
    k = target_dim
    rng = np.random.RandomState(seed)
    A = torch.tensor(rng.randn(d_total, k) / np.sqrt(k), dtype=torch.double)
    x_center = torch.full((d_total,), 0.5, dtype=torch.double)

    sobol = SobolEngine(dimension=d_total, scramble=True, seed=seed)
    x_samp = sobol.draw(4096).double()
    z_samp = (x_samp - x_center) @ A
    z_min = z_samp.min(dim=0).values
    z_max = z_samp.max(dim=0).values
    z_bounds = torch.stack([z_min, z_max])

    def z_to_x(z: torch.Tensor) -> np.ndarray:
        x = x_center + A @ z
        return torch.clamp(x, 0.0, 1.0).detach().numpy()

    sobol_z = SobolEngine(dimension=k, scramble=True, seed=seed)
    z_init = z_min + sobol_z.draw(n_init).double() * (z_max - z_min)

    all_z, all_y, best_values = [], [], []
    for i in range(n_init):
        y = eval_fn(z_to_x(z_init[i]))
        all_z.append(z_init[i].numpy())
        all_y.append(y)
        best_values.append(max(all_y))

    for step in range(budget - n_init):
        train_Z = torch.tensor(np.array(all_z), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        train_Z_norm = normalize(train_Z, z_bounds)
        gp = SingleTaskGP(train_Z_norm, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        if not _fit_with_warning(mll, f"rembo(seed={seed}, step={step})"):
            z_new_arr = rng.uniform(low=z_min.numpy(), high=z_max.numpy())
            z_new = torch.tensor(z_new_arr, dtype=torch.double)
        else:
            cand = _suggest_qlnei(gp, train_Z_norm, k)
            z_new = unnormalize(cand, z_bounds)
        y_new = eval_fn(z_to_x(z_new))
        all_z.append(z_new.detach().numpy())
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def run_turbo(eval_fn, d_total: int, budget: int, n_init: int, seed: int) -> Dict:
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    d = d_total
    length, length_min, length_max = 0.8, 0.5 ** 7, 1.6
    # TuRBO defaults (Eriksson et al. 2019, §3): failure tolerance scales
    # with effective dimension as max(4, ceil(d_e / batch_size)). With q=1,
    # this reduces to max(4, d). The previous hardcoded value of 4 caused
    # the trust region to collapse far too quickly in high dimension.
    success_tol, failure_tol = 3, max(4, d)
    n_succ, n_fail = 0, 0

    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    x_init = sobol.draw(n_init).double().numpy()
    all_x, all_y, best_values = [], [], []
    for i in range(n_init):
        x = x_init[i]
        y = eval_fn(x)
        all_x.append(x)
        all_y.append(y)
        best_values.append(max(all_y))
    best_y = max(all_y)

    for step in range(budget - n_init):
        train_X = torch.tensor(np.array(all_x), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        if not _fit_with_warning(mll, f"turbo(seed={seed}, step={step})"):
            x_new = rng.random(d)
            y_new = eval_fn(x_new)
            all_x.append(x_new)
            all_y.append(y_new)
            best_values.append(max(all_y))
            continue

        best_idx = int(np.argmax(all_y))
        x_center = train_X[best_idx]
        covar = gp.covar_module
        ls = (covar.base_kernel.lengthscale if hasattr(covar, "base_kernel")
              else covar.lengthscale).detach().squeeze().clamp_min(1e-12)
        weights = ls.mean() / ls
        weights = weights / torch.prod(weights).pow(1.0 / d)
        tr_lb = torch.clamp(x_center - weights * length / 2, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * length / 2, 0.0, 1.0)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        acq = qLogNoisyExpectedImprovement(model=gp, X_baseline=train_X, sampler=sampler)
        cand, _ = optimize_acqf(
            acq_function=acq, bounds=torch.stack([tr_lb, tr_ub]).double(),
            q=1, num_restarts=10, raw_samples=512,
        )
        x_new = cand.squeeze(0).detach().numpy()
        y_new = eval_fn(x_new)
        all_x.append(x_new)
        all_y.append(y_new)
        best_values.append(max(all_y))

        if y_new > best_y:
            n_succ, n_fail, best_y = n_succ + 1, 0, y_new
        else:
            n_succ, n_fail = 0, n_fail + 1
        if n_succ >= success_tol:
            length, n_succ = min(2.0 * length, length_max), 0
        elif n_fail >= failure_tol:
            length, n_fail = length / 2.0, 0
        if length < length_min:
            length, n_succ, n_fail = 0.8, 0, 0

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def run_saasbo(
    eval_fn, d_total: int, budget: int, n_init: int, seed: int,
    warmup_steps: int = 512, num_samples: int = 256,
) -> Dict:
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    d = d_total

    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    x_init = sobol.draw(n_init).double().numpy()
    all_x, all_y, best_values = [], [], []
    for i in range(n_init):
        x = x_init[i]
        y = eval_fn(x)
        all_x.append(x)
        all_y.append(y)
        best_values.append(max(all_y))

    for step in range(budget - n_init):
        train_X = torch.tensor(np.array(all_x), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        gp = SaasFullyBayesianSingleTaskGP(train_X=train_X, train_Y=train_Y)
        try:
            fit_fully_bayesian_model_nuts(
                gp, warmup_steps=warmup_steps, num_samples=num_samples, disable_progbar=True,
            )
            fit_ok = True
        except Exception as exc:
            warnings.warn(
                f"saasbo(seed={seed}): NUTS fit failed "
                f"({type(exc).__name__}: {exc}). Falling back to SingleTaskGP.",
                RuntimeWarning, stacklevel=2,
            )
            gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_ok = _fit_with_warning(mll, f"saasbo(seed={seed}) fallback")
        if not fit_ok:
            x_new = rng.random(d)
        else:
            x_new = _suggest_qlnei(gp, train_X, d).detach().numpy()
        y_new = eval_fn(x_new)
        all_x.append(x_new)
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


class _PenalizedAcq(AcquisitionFunction):
    def __init__(self, base_acq, penalty_func, lambda_t: float):
        super().__init__(model=base_acq.model)
        self.base_acq = base_acq
        self.penalty_func = penalty_func
        self.lambda_t = float(lambda_t)

    def forward(self, X):
        return self.base_acq(X) - self.lambda_t * self.penalty_func(X)


def run_sebo(
    eval_fn, d_total: int, budget: int, n_init: int, seed: int,
    lambda_max: float = 1.0, l0_a: float = 0.05,
    warmup_steps: int = 512, num_samples: int = 256,
) -> Dict:
    """SEBO (Liu et al. 2023): SAAS prior + smooth-L0 penalty against x_ref=0."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    d = d_total
    a_squared = float(l0_a) ** 2
    # Smooth-L0 penalty centered at 0 to match the paper's supp(x) ⊆ S
    # convention (sparsity = "few coords differ from 0").
    x_ref = torch.zeros(d, dtype=torch.double)

    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    x_init = sobol.draw(n_init).double().numpy()
    all_x, all_y, best_values = [], [], []
    for i in range(n_init):
        x = x_init[i]
        y = eval_fn(x)
        all_x.append(x)
        all_y.append(y)
        best_values.append(max(all_y))

    def sparsity_penalty(X: torch.Tensor) -> torch.Tensor:
        diff = X - x_ref
        active = 1.0 - torch.exp(-diff.pow(2) / a_squared)
        return active.sum(dim=(-1, -2))

    n_bo_steps = budget - n_init
    for step in range(n_bo_steps):
        train_X = torch.tensor(np.array(all_x), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        gp = SaasFullyBayesianSingleTaskGP(train_X=train_X, train_Y=train_Y)
        try:
            fit_fully_bayesian_model_nuts(
                gp, warmup_steps=warmup_steps, num_samples=num_samples, disable_progbar=True,
            )
            fit_ok = True
        except Exception as exc:
            warnings.warn(
                f"sebo(seed={seed}): NUTS fit failed "
                f"({type(exc).__name__}: {exc}). Falling back to SingleTaskGP.",
                RuntimeWarning, stacklevel=2,
            )
            gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_ok = _fit_with_warning(mll, f"sebo(seed={seed}) fallback")
        if not fit_ok:
            x_new = rng.random(d)
            y_new = eval_fn(x_new)
            all_x.append(x_new)
            all_y.append(y_new)
            best_values.append(max(all_y))
            continue

        progress = (step + 1) / max(1, n_bo_steps)
        lambda_t = lambda_max * progress
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        base_acq = qLogNoisyExpectedImprovement(
            model=gp, X_baseline=train_X, sampler=sampler,
        )
        penalized = _PenalizedAcq(base_acq, sparsity_penalty, lambda_t)
        cand, _ = optimize_acqf(
            acq_function=penalized,
            bounds=torch.stack([torch.zeros(d), torch.ones(d)]).double(),
            q=1, num_restarts=10, raw_samples=512,
        )
        x_new = cand.squeeze(0).detach().numpy()
        y_new = eval_fn(x_new)
        all_x.append(x_new)
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def run_egbo(
    eval_fn, d_total: int, budget: int, n_init: int, seed: int,
    config: ExpertConfig, adaptive: bool, n_rounds: "int | None" = None,
) -> Dict:
    """Expert-guided BO.

    one-shot (Q=1):  expert queried once before any evaluation (adaptive=False).
    adaptive (Q=T):  expert queried before each of the T evaluations,
                     including the n_init Sobol points; active set grows
                     monotonically (adaptive=True). Differs from
                     n_rounds >= T - n_init + 1, which skips the init points.
    Q=N (N>1):       n_rounds parameter overrides the adaptive boolean. The
                     budget is split into N rounds; round 0 holds the n_init
                     Sobol points plus the first chunk of BO steps, subsequent
                     rounds hold equal BO budget. Expert is queried at the
                     start of each round; active set grows monotonically.
    Inactive variables are fixed at 0 throughout (supp(x) ⊆ S convention).
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    active: set = set()

    # Compute the iterations at which we query the expert.
    if n_rounds is not None and n_rounds >= 1:
        if n_rounds == 1:
            query_iters = {0}
        else:
            bo_total = max(0, budget - n_init)
            if n_rounds >= bo_total + 1:
                # More query rounds than BO steps fit at distinct iterations:
                # query at round 0 and at every BO iteration thereafter.
                # Equivalent to Q=T behavior; capped at one query per iteration.
                query_iters = {0} | set(range(n_init, budget))
            else:
                # Float-based even spacing so positions are distinct for any
                # n_rounds <= bo_total. (Integer floor division can collapse
                # later rounds to the same iteration when n_rounds is large.)
                query_iters = {0}
                for r in range(1, n_rounds):
                    query_iters.add(n_init + int(round(r * bo_total / n_rounds)))
    elif adaptive:
        query_iters = set(range(budget))  # Q=T
    else:
        query_iters = {0}                  # one-shot
    if 0 in query_iters:
        active = expert_step(active, d_total, config, rng)
        query_iters = query_iters - {0}

    all_x_full: List[np.ndarray] = []
    all_y, best_values = [], []
    active_size_trace: List[int] = []
    n_relevant_trace: List[int] = []

    for t in range(budget):
        if t in query_iters:
            active = expert_step(active, d_total, config, rng)
        active_list = sorted(active)
        d_active = len(active_list)
        active_size_trace.append(d_active)
        n_relevant_trace.append(len(set(range(D_RELEVANT)) & active))

        if d_active == 0:
            x_full = rng.random(d_total)
            y = eval_fn(x_full)
            all_x_full.append(x_full)
            all_y.append(y)
            best_values.append(max(all_y))
            continue

        if t < n_init or len(all_y) < 2:
            sobol = SobolEngine(dimension=d_active, scramble=True, seed=seed + t)
            x_sub = sobol.draw(1).double().squeeze(0).numpy()
        else:
            X_active_hist = np.array([x[active_list] for x in all_x_full])
            train_X = torch.tensor(X_active_hist, dtype=torch.double)
            train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
            gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            if not _fit_with_warning(mll, f"egbo(seed={seed}, k={d_active}, t={t})"):
                x_sub = rng.random(d_active)
            else:
                x_sub = _suggest_qlnei(gp, train_X, d_active).detach().numpy()

        # Inactive coordinates pinned at 0 to match the supp(x) ⊆ S
        # convention used in the paper's Assumption 1 / Theorem 1.
        x_full = np.zeros(d_total)
        x_full[active_list] = x_sub
        y = eval_fn(x_full)
        all_x_full.append(x_full)
        all_y.append(y)
        best_values.append(max(all_y))

    return {
        "best_values": best_values,
        "all_y": all_y,
        "final_best": max(all_y),
        "active_set_sizes": active_size_trace,
        "n_relevant_selected": n_relevant_trace,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_METHODS = [
    "random", "vanilla", "oracle", "rembo", "random_subset",
    "turbo", "saasbo", "sebo",
    "egbo_strong", "egbo_weak",
    "egbo_strong_blocked", "egbo_weak_blocked",
    _EGBO_CUSTOM_KEY,
]


def run_method(
    method: str, eval_fn, d_total: int, budget: int, n_init: int, seed: int,
    expert_mode: str, n_rounds: "int | None" = None,
    expert_p: "float | None" = None, expert_q: "float | None" = None,
    expert_rel_block: int = 1, expert_irr_block: int = 1,
) -> Dict:
    if method == "random":
        return run_random(eval_fn, d_total, budget, n_init, seed)
    if method == "vanilla":
        return run_vanilla_bo(eval_fn, d_total, budget, n_init, seed)
    if method == "oracle":
        return run_oracle(eval_fn, d_total, budget, n_init, seed)
    if method == "rembo":
        return run_rembo(eval_fn, d_total, budget, n_init, seed)
    if method == "random_subset":
        return run_random_subset(eval_fn, d_total, budget, n_init, seed)
    if method == "turbo":
        return run_turbo(eval_fn, d_total, budget, n_init, seed)
    if method == "saasbo":
        return run_saasbo(eval_fn, d_total, budget, n_init, seed)
    if method == "sebo":
        return run_sebo(eval_fn, d_total, budget, n_init, seed)
    if method == _EGBO_CUSTOM_KEY:
        if expert_p is None or expert_q is None:
            raise ValueError(
                "egbo_custom requires --expert-p and --expert-q to be set."
            )
        cfg = ExpertConfig(
            name=f"egbo_custom_p{expert_p:g}_q{expert_q:g}",
            p=float(expert_p), q=float(expert_q),
            rel_block_size=int(expert_rel_block), irr_block_size=int(expert_irr_block),
        )
        return run_egbo(
            eval_fn, d_total, budget, n_init, seed,
            config=cfg,
            adaptive=(expert_mode == "adaptive"),
            n_rounds=n_rounds,
        )
    if method in EXPERT_PRESETS:
        return run_egbo(
            eval_fn, d_total, budget, n_init, seed,
            config=EXPERT_PRESETS[method],
            adaptive=(expert_mode == "adaptive"),
            n_rounds=n_rounds,
        )
    raise ValueError(f"Unknown method: {method!r}")


def run_simulation(
    d_total: int, budget: int, n_init: int, n_seeds: int,
    expert_mode: str, methods: List[str], seed_start: int = 0,
    n_rounds: "int | None" = None,
    expert_p: "float | None" = None, expert_q: "float | None" = None,
    expert_rel_block: int = 1, expert_irr_block: int = 1,
) -> Dict[str, Dict]:
    eval_fn = make_embedded_objective(d_total)
    results: Dict[str, Dict] = {}

    for method in methods:
        print(f"  running {method}...", flush=True)
        traces, active_traces, relevant_traces = [], [], []
        for seed in range(seed_start, seed_start + n_seeds):
            r = run_method(
                method, eval_fn, d_total, budget, n_init, seed, expert_mode,
                n_rounds=n_rounds,
                expert_p=expert_p, expert_q=expert_q,
                expert_rel_block=expert_rel_block, expert_irr_block=expert_irr_block,
            )
            traces.append(r["best_values"])
            if "active_set_sizes" in r:
                active_traces.append(r["active_set_sizes"])
                relevant_traces.append(r["n_relevant_selected"])
            print(f"    seed {seed}: final_best = {r['final_best']:.4f}")
        traces_arr = np.array(traces)
        method_result = {
            "all_traces": traces_arr.tolist(),
            "mean_trace": traces_arr.mean(axis=0).tolist(),
            "std_trace": traces_arr.std(axis=0).tolist(),
            "mean_final": float(traces_arr[:, -1].mean()),
            "std_final": float(traces_arr[:, -1].std()),
        }
        if active_traces:
            method_result["active_set_all_traces"] = active_traces
            method_result["relevant_all_traces"] = relevant_traces
        results[method] = method_result
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METHOD_STYLES = {
    "random":         {"color": "#8c564b",  "ls": ":",  "label": "Random search"},
    "vanilla":        {"color": "#7f7f7f",  "ls": "--", "label": "Vanilla BO"},
    "oracle":         {"color": "black",    "ls": "-",  "label": "Oracle (p=1, q=0)"},
    "rembo":          {"color": "#1f77b4",  "ls": "-.", "label": "REMBO (k=6)"},
    "random_subset":  {"color": "#6a3d9a",  "ls": "--", "label": "Random subset + BO (k=6)"},
    "turbo":          {"color": "tab:cyan", "ls": "-.", "label": "TuRBO"},
    "saasbo":         {"color": "tab:purple","ls": "-.","label": "SAASBO"},
    "sebo":           {"color": "#f1c40f",  "ls": "-.", "label": "SEBO"},
    "egbo_strong":    {"color": "#1b9e77",  "ls": "-",  "label": "EGBO (p=0.9, q=0.01)"},
    "egbo_weak":      {"color": "#d95f02",  "ls": "-",  "label": "EGBO (p=0.2, q=0.10)"},
    "egbo_strong_blocked": {"color": "#1b9e77", "ls": "--", "label": "EGBO blocked (p=0.9, q=0.01)"},
    "egbo_weak_blocked":   {"color": "#d95f02", "ls": "--", "label": "EGBO blocked (p=0.2, q=0.10)"},
}

_DIM_TAGGED = {"vanilla", "turbo", "saasbo", "sebo"}


def _label(method: str, d_total: int) -> str:
    base = METHOD_STYLES.get(method, {}).get("label", method)
    if method in _DIM_TAGGED:
        base = f"{base} (d={d_total})"
    if method == "rembo" or method == "random_subset":
        # already say k=6 in the label
        return base
    return base


def plot_convergence(
    results: Dict, d_total: int, budget: int, n_seeds: int,
    expert_mode: str, save_path: str,
):
    fig, ax = plt.subplots(1, 1, figsize=(11, 6.3))
    iters = np.arange(1, budget + 1)
    for method, data in results.items():
        style = METHOD_STYLES.get(method, {"color": "grey", "ls": "-", "label": method})
        mean = np.array(data["mean_trace"])
        std = np.array(data["std_trace"])
        sem = std / np.sqrt(n_seeds)
        ax.plot(iters, mean, color=style["color"], ls=style["ls"],
                label=_label(method, d_total), linewidth=2)
        ax.fill_between(iters, mean - 1.96 * sem, mean + 1.96 * sem,
                        color=style["color"], alpha=0.12)
    ax.axhline(HARTMANN6_GLOBAL_MAX, color="gold", ls=":", lw=1.5,
               label=f"Global opt. ({HARTMANN6_GLOBAL_MAX:.4f})")
    ax.set_xlabel("Evaluation", fontsize=13)
    ax.set_ylabel("Best Hartmann6 Value Found", fontsize=13)
    suffix = "one-shot (Q=1)" if expert_mode == "oneshot" else "adaptive (Q=T)"
    ax.set_title(f"EGBO vs Baselines on Hartmann6 (d=200), {suffix}", fontsize=14)
    ax.legend(loc="lower right", fontsize=9, ncol=2, frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=170)
    plt.close(fig)
    print(f"  saved {save_path}")


def print_results(results: Dict, d_total: int):
    print("\n" + "=" * 72)
    print(f"FINAL HARTMANN6 RESULTS (d_total={d_total}, max=3.3224)")
    print("=" * 72)
    print(f"{'Method':<32} {'Final Best (mean ± std)':<28} {'Gap'}")
    print("-" * 72)
    for method, data in sorted(results.items(), key=lambda kv: -kv[1]["mean_final"]):
        label = _label(method, d_total)
        perf = f"{data['mean_final']:.4f} ± {data['std_final']:.4f}"
        gap = HARTMANN6_GLOBAL_MAX - data["mean_final"]
        print(f"{label:<32} {perf:<28} {gap:+.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hartmann6 embedded-in-N-dim simulation for EGBO vs baselines",
    )
    parser.add_argument("--dim", type=int, default=200, help="Total ambient dim N")
    parser.add_argument("--budget", type=int, default=20, help="Total evaluations T")
    parser.add_argument("--n-init", type=int, default=5, help="Initial Sobol points")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds")
    parser.add_argument(
        "--seed-start", type=int, default=0,
        help="First seed index (default: 0). With --seeds N --seed-start S, seeds S..S+N-1 are run.",
    )
    parser.add_argument(
        "--expert-mode", choices=["oneshot", "adaptive"], default="oneshot",
        help="Q=1 (one-shot) or Q=T (adaptive) expert queries",
    )
    parser.add_argument(
        "--n-rounds", type=int, default=None,
        help="Number of expert query rounds Q (overrides --expert-mode if set). "
             "Q=1 = one-shot, Q=2 = lightweight adaptive, Q=T (=budget) = full adaptive.",
    )
    parser.add_argument(
        "--expert-p", type=float, default=None,
        help="Per-iteration recall (Pr add a relevant variable not yet active). "
             "Required when 'egbo_custom' is in --methods. Overrides preset for it.",
    )
    parser.add_argument(
        "--expert-q", type=float, default=None,
        help="Per-iteration false-positive rate (Pr add an irrelevant variable). "
             "Required when 'egbo_custom' is in --methods. Overrides preset for it.",
    )
    parser.add_argument(
        "--rel-block-size", type=int, default=1,
        help="egbo_custom only: add relevant coords in all-or-none blocks of this "
             "size (matched marginals; 1 = independent). See Experiment B.",
    )
    parser.add_argument(
        "--irr-block-size", type=int, default=1,
        help="egbo_custom only: add irrelevant coords in all-or-none blocks of this "
             "size (matched marginals; 1 = independent).",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None, choices=ALL_METHODS,
        help=f"Methods to run (default: all). Choices: {', '.join(ALL_METHODS)}",
    )
    parser.add_argument("--name", type=str, default=None,
                        help="Run name. Outputs go to <output-root>/<name>/. Defaults to a timestamp.")
    parser.add_argument(
        "--output-root", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
        help="Root directory for outputs (default: results/ next to this script)",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    # When --methods is omitted, default to all methods EXCEPT egbo_custom
    # (egbo_custom requires --expert-p / --expert-q and must be opted into).
    methods = args.methods or [m for m in ALL_METHODS if m != _EGBO_CUSTOM_KEY]
    run_id = args.name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    def _atomic_write_json(path: str, obj) -> None:
        # Defensive against networked / synced filesystems where mkdir and a
        # subsequent open() can race. Re-create the parent and write via tmp.
        parent = os.path.dirname(path)
        os.makedirs(parent, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)

    print(f"Hartmann6 in d={args.dim}, T={args.budget}, n_init={args.n_init}, "
          f"seeds={args.seeds} starting from {args.seed_start}, "
          f"expert_mode={args.expert_mode}")
    print(f"Methods: {methods}")
    print(f"Run folder: {run_dir}\n")

    if _EGBO_CUSTOM_KEY in methods and (args.expert_p is None or args.expert_q is None):
        raise SystemExit(
            f"--methods includes {_EGBO_CUSTOM_KEY} but --expert-p / --expert-q not set."
        )

    results = run_simulation(
        d_total=args.dim, budget=args.budget, n_init=args.n_init,
        n_seeds=args.seeds, expert_mode=args.expert_mode, methods=methods,
        seed_start=args.seed_start, n_rounds=args.n_rounds,
        expert_p=args.expert_p, expert_q=args.expert_q,
        expert_rel_block=args.rel_block_size, expert_irr_block=args.irr_block_size,
    )
    print_results(results, args.dim)

    raw_path = os.path.join(run_dir, "raw.json")
    payload = {
        "config": {
            "function": "hartmann6",
            "d_relevant": D_RELEVANT,
            "d_total": int(args.dim),
            "budget": int(args.budget),
            "n_init": int(args.n_init),
            "n_seeds": int(args.seeds),
            "seed_start": int(args.seed_start),
            "expert_mode": args.expert_mode,
            "n_rounds": (None if args.n_rounds is None else int(args.n_rounds)),
            "expert_p": (None if args.expert_p is None else float(args.expert_p)),
            "expert_q": (None if args.expert_q is None else float(args.expert_q)),
            "rel_block_size": int(args.rel_block_size),
            "irr_block_size": int(args.irr_block_size),
            "methods": methods,
            "global_max": float(HARTMANN6_GLOBAL_MAX),
        },
        "results": results,
    }
    _atomic_write_json(raw_path, payload)
    print(f"  saved {raw_path}")

    if not args.no_plot:
        plot_convergence(
            results, d_total=args.dim, budget=args.budget, n_seeds=args.seeds,
            expert_mode=args.expert_mode,
            save_path=os.path.join(run_dir, "convergence.png"),
        )


if __name__ == "__main__":
    main()
