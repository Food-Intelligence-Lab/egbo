"""Expert-set BO configuration arm for FormulateBench (EGBO Role 3).

Motivation
----------
On FormulateBench the paper finds the *optimization gap* dominates the
*selection gap*: the LLM's k=9 ingredient set already spans a near-optimal
NNLS support, so the residual comes from BO failing to converge inside the
chosen subspace, not from the subspace being wrong. Yet every method in
``run_formulatebench.py`` uses the identical BoTorch-default GP:

    SingleTaskGP(train_Z_norm, train_Y, outcome_transform=Standardize(m=1))
    + qLogNoisyExpectedImprovement

This module lets the *expert* also configure the optimizer (kernel,
lengthscale prior, observation noise, acquisition), not just the subspace.
It is a drop-in alternative to ``reduced_bo`` that takes a validated
``BOConfig``, plus an LLM prompt that returns such a config, plus fixed
"oracle" configs for the ablation that separates two questions:

    (a) does expert-set BO config help at all?          -> egbo_cfg_oracle
    (b) can the LLM pick a good config from the problem? -> egbo_cfg_llm

Important parametrization note (read before interpreting results)
-----------------------------------------------------------------
BO runs in logit space z in [-12, 0]^k, mapped to the simplex by softmax:
x = softmax(z). The nutrition loss is linear in x, but softmax is nonlinear,
so y(z) = -sqrt(mean((z_profile(softmax(z)) - z_target)^2)) is a smooth,
saturating, *nonlinear* function of z. A linear kernel is therefore NOT
obviously matched to the geometry; the defensible expert levers here are:
  * observation noise: the in-silico objective is deterministic, but the
    default SingleTaskGP infers a noise level and a homoskedastic prior that
    assumes nontrivial noise. Fixing a tiny noise is well-justified and, at
    ~30 evaluations, often the single highest-value change.
  * lengthscale prior: softmax compresses logits, so the effective function
    in normalized-z space is smooth -> a longer-lengthscale prior can beat
    BoTorch's default GammaPrior(3, 6) (mode ~ 1/12 of the unit range).
  * acquisition: qLogNEI marginalizes observation noise; on a deterministic
    objective qLogEI (noiseless) is the matched choice.
  * kernel family: smoothness assumption (Matern-5/2 default vs RBF vs a
    degree-2 polynomial that can capture the quadratic-form curvature).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Sequence

import numpy as np
import torch
from botorch.acquisition import (
    qLogNoisyExpectedImprovement,
    qLogExpectedImprovement,
)
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.kernels import (
    LinearKernel,
    MaternKernel,
    PolynomialKernel,
    RBFKernel,
    ScaleKernel,
)
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior


# ----------------------------------------------------------------------------
# Config schema: a small, whitelisted space the expert may set.
# ----------------------------------------------------------------------------

_KERNELS = {"matern52", "matern32", "rbf", "linear", "poly2"}
_LENGTHSCALE = {"default", "long", "short"}     # ignored for linear/poly2
_NOISE = {"default", "low", "fixed_tiny"}
_ACQ = {"qlognei", "qlogei", "qucb"}
_ARD = {"ard", "isotropic"}

# The single fixed config used for the "config helps?" oracle arm. Chosen a
# priori from the parametrization note above (deterministic + smooth-under-
# softmax + tiny budget), NOT tuned per category, so it is a fair oracle.
ORACLE_CONFIG_SPEC = {
    "kernel": "matern52",
    "lengthscale_prior": "long",
    "noise": "fixed_tiny",
    "acquisition": "qlogei",
    "ard": "isotropic",
}

# BoTorch-default equivalent -> reproduces the existing `llm` (EGBO) arm.
# Use as the control so the only thing that changes across arms is the config.
DEFAULT_CONFIG_SPEC = {
    "kernel": "matern52",
    "lengthscale_prior": "default",
    "noise": "default",
    "acquisition": "qlognei",
    "ard": "ard",
}


@dataclass
class BOConfig:
    kernel: str = "matern52"
    lengthscale_prior: str = "default"
    noise: str = "default"
    acquisition: str = "qucb" if False else "qlognei"
    ard: str = "ard"
    beta: float = 0.2                       # only used by qucb
    known_noise_var: Optional[float] = None  # if set: fixed observation-noise variance (raw-Y units)
    _fallbacks: List[str] = field(default_factory=list)

    @classmethod
    def from_spec(cls, spec: Dict, *, context: str = "") -> "BOConfig":
        """Validate a raw dict into a BOConfig, coercing bad fields to the
        default and recording what was coerced (so misgenerated LLM configs
        degrade gracefully to the control rather than crashing a 24-category
        sweep)."""
        fallbacks: List[str] = []

        def pick(key, allowed, default):
            v = str(spec.get(key, default)).strip().lower()
            if v not in allowed:
                fallbacks.append(f"{key}={spec.get(key)!r}->default({default})")
                return default
            return v

        cfg = cls(
            kernel=pick("kernel", _KERNELS, "matern52"),
            lengthscale_prior=pick("lengthscale_prior", _LENGTHSCALE, "default"),
            noise=pick("noise", _NOISE, "default"),
            acquisition=pick("acquisition", _ACQ, "qlognei"),
            ard=pick("ard", _ARD, "ard"),
        )
        try:
            cfg.beta = float(spec.get("beta", 0.2))
        except (TypeError, ValueError):
            cfg.beta = 0.2
        cfg._fallbacks = fallbacks
        if fallbacks and context:
            warnings.warn(
                f"BOConfig.from_spec[{context}] coerced: {'; '.join(fallbacks)}",
                RuntimeWarning, stacklevel=2,
            )
        return cfg

    def label(self) -> str:
        return (f"{self.kernel}/{self.lengthscale_prior}/{self.noise}/"
                f"{self.acquisition}/{self.ard}")


# ----------------------------------------------------------------------------
# GP + acquisition factories (the only place BO hyperparameters are set).
# ----------------------------------------------------------------------------

# Lengthscale presets as GammaPrior(concentration, rate). BoTorch default for
# MaternKernel is GammaPrior(3.0, 6.0) (mode ~= 0.33 in *raw* units; ARD acts
# on normalized [0,1] inputs). "long" pushes toward smoother fits; "short"
# toward wigglier. Concentrations kept >1 so the mode is well-defined.
_LS_PRIORS = {
    "default": (3.0, 6.0),
    "long":    (3.0, 1.0),      # mode = 2.0 -> smooth, good for ~30 evals
    "short":   (3.0, 12.0),     # mode = 1/6 -> wigglier
}


def _base_kernel(cfg: BOConfig, ard_num_dims: int):
    if cfg.kernel in ("matern52", "matern32"):
        nu = 2.5 if cfg.kernel == "matern52" else 1.5
        conc, rate = _LS_PRIORS[cfg.lengthscale_prior]
        return MaternKernel(
            nu=nu, ard_num_dims=ard_num_dims,
            lengthscale_prior=GammaPrior(conc, rate),
        )
    if cfg.kernel == "rbf":
        conc, rate = _LS_PRIORS[cfg.lengthscale_prior]
        return RBFKernel(
            ard_num_dims=ard_num_dims,
            lengthscale_prior=GammaPrior(conc, rate),
        )
    if cfg.kernel == "linear":
        return LinearKernel()
    if cfg.kernel == "poly2":
        return PolynomialKernel(power=2)
    raise ValueError(f"unknown kernel {cfg.kernel!r}")


def build_gp(train_Z_norm: torch.Tensor, train_Y: torch.Tensor, cfg: BOConfig):
    """SingleTaskGP whose covariance/noise are set by `cfg`. Inputs are the
    already-normalized logits; Standardize handles the outputs, matching the
    default arms exactly except for the configured pieces."""
    k = train_Z_norm.shape[-1]
    ard_num_dims = k if cfg.ard == "ard" else None
    covar = ScaleKernel(_base_kernel(cfg, ard_num_dims=ard_num_dims))

    train_Yvar = None
    if cfg.known_noise_var is not None:
        # Known observation noise (e.g. a characterized assay): fix the GP's
        # noise variance instead of learning it. Off by default.
        train_Yvar = torch.full_like(train_Y, float(cfg.known_noise_var))
    elif cfg.noise == "fixed_tiny":
        # Deterministic objective: pin near-zero noise (in raw-Y units; the
        # outcome transform rescales it). 1e-6 relative to observed spread.
        yvar = float(max(1e-8, 1e-6 * (train_Y.var().item() + 1e-12)))
        train_Yvar = torch.full_like(train_Y, yvar)

    if train_Yvar is not None:
        gp = SingleTaskGP(
            train_Z_norm, train_Y, train_Yvar=train_Yvar,
            covar_module=covar, outcome_transform=Standardize(m=1),
        )
    else:
        gp = SingleTaskGP(
            train_Z_norm, train_Y,
            covar_module=covar, outcome_transform=Standardize(m=1),
        )
        if cfg.noise == "low":
            # Bias the learned noise downward without fully fixing it.
            gp.likelihood.noise_covar.register_prior(
                "noise_prior", GammaPrior(1.1, 50.0), "raw_noise",
            )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    return gp, mll


def build_acqf(gp, train_Z_norm: torch.Tensor, cfg: BOConfig):
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
    if cfg.acquisition == "qlognei":
        return qLogNoisyExpectedImprovement(
            model=gp, X_baseline=train_Z_norm, sampler=sampler)
    if cfg.acquisition == "qlogei":
        best_f = gp.train_targets.max().item()
        return qLogExpectedImprovement(model=gp, best_f=best_f, sampler=sampler)
    if cfg.acquisition == "qucb":
        return qUpperConfidenceBound(model=gp, beta=cfg.beta, sampler=sampler)
    raise ValueError(f"unknown acquisition {cfg.acquisition!r}")


# ----------------------------------------------------------------------------
# The optimizer: reduced_bo, but with an expert-set config.
# Signature mirrors run_formulatebench.reduced_bo so it slots into the
# method_registry lambdas unchanged.
# ----------------------------------------------------------------------------

def reduced_bo_cfg(
    objective, d_full: int, selected_indices: List[int],
    budget: int, n_init: int, seed: int, cfg: BOConfig,
    fit_mll,          # pass run_formulatebench._fit_gpytorch_mll_with_warning
    softmax,          # pass run_formulatebench._softmax
    init_logits,      # pass run_formulatebench._simplex_initial_logits
    initial_x=None,
    initial_design: str = "global",
    local_sigma: float = 0.35,
) -> Dict:
    """EGBO subspace BO with a configurable GP/acquisition. Identical to
    ``reduced_bo`` when cfg == DEFAULT_CONFIG_SPEC, so it is a clean control."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    k = len(selected_indices)
    z_lo, z_hi = -12.0, 0.0
    bounds = torch.tensor([[z_lo] * k, [z_hi] * k], dtype=torch.double)

    def z_to_x(z):
        fracs = softmax(z)
        x = np.zeros(d_full)
        for i, idx in enumerate(selected_indices):
            x[idx] = fracs[i]
        return x

    if initial_design not in {"global", "local"}:
        raise ValueError("initial_design must be 'global' or 'local'")

    all_z, all_y, best_values = [], [], []
    if initial_x is None:
        z_init = init_logits(k, n_init, seed)
    else:
        x0_full = np.asarray(initial_x, dtype=float)
        if (x0_full.shape != (d_full,) or np.any(x0_full < 0)
                or not np.isfinite(x0_full).all() or x0_full.sum() <= 0):
            raise ValueError(
                "initial_x must be a finite nonnegative vector of length d_full"
            )
        outside = np.delete(x0_full, selected_indices)
        if np.any(outside > 1e-12):
            raise ValueError("initial_x has positive mass outside selected_indices")
        x0 = x0_full[selected_indices]
        x0 = x0 / x0.sum()
        z0 = np.log(np.clip(x0, np.exp(z_lo), 1.0))
        z_init = [z0]
        if initial_design == "local":
            for _ in range(max(0, n_init - 1)):
                # Logistic-normal perturbations preserve the mixture geometry
                # and keep the initial design near the scientist's recipe.
                z = np.clip(z0 + rng.normal(0.0, local_sigma, size=k), z_lo, z_hi)
                z_init.append(z)
        else:
            z_init.extend(init_logits(k, max(0, n_init - 1), seed + 1))

    for i, z in enumerate(z_init):
        # Count the scientist's recipe itself as evaluation 1. For the
        # full-catalog arm, converting exact zeros through finite logits
        # otherwise adds tiny mass to every inactive ingredient and makes the
        # nominally shared starting evaluation differ across methods.
        if i == 0 and initial_x is not None:
            y = objective(x0_full / x0_full.sum())
        else:
            y = objective(z_to_x(z))
        all_z.append(z)
        all_y.append(y)
        best_values.append(max(all_y))

    for _ in range(budget - n_init):
        train_Z = torch.tensor(np.array(all_z), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        train_Z_norm = normalize(train_Z, bounds)

        try:
            gp, mll = build_gp(train_Z_norm, train_Y, cfg)
            fit_ok = fit_mll(mll, f"reduced_bo_cfg(seed={seed}, k={k}, {cfg.label()})")
        except Exception as exc:                      # noqa: BLE001
            warnings.warn(
                f"reduced_bo_cfg build/fit failed ({type(exc).__name__}: {exc}); "
                f"random step.", RuntimeWarning, stacklevel=2)
            fit_ok = False

        if not fit_ok:
            z_new = rng.uniform(z_lo, z_hi, size=k)
            y_new = objective(z_to_x(z_new))
            all_z.append(z_new); all_y.append(y_new); best_values.append(max(all_y))
            continue

        acq = build_acqf(gp, train_Z_norm, cfg)
        cand, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack([torch.zeros(k), torch.ones(k)]).double(),
            q=1, num_restarts=10, raw_samples=512,
        )
        z_new = unnormalize(cand.squeeze(0), bounds).detach().numpy()
        y_new = objective(z_to_x(z_new))
        all_z.append(z_new); all_y.append(y_new); best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y),
            "bo_config": cfg.label(), "config_fallbacks": cfg._fallbacks,
            "initial_design": initial_design, "local_sigma": local_sigma}


# ----------------------------------------------------------------------------
# Expert query: LLM returns a BO config for this category.
# ----------------------------------------------------------------------------

_CONFIG_SYSTEM_PROMPT = (
    "You are an expert in Bayesian optimization configuring a Gaussian-process "
    "optimizer for a specific black-box problem."
)


def _extract_json(raw: str) -> Dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object in response: {raw[:200]!r}")
    return json.loads(raw[start : end + 1])


def llm_select_bo_config(
    client, category: str, k: int, budget: int, target_block: str,
    *, temperature: float = 0.0,
) -> BOConfig:
    """Round-0 expert query for the BO configuration (Role 3). Falls back to
    DEFAULT_CONFIG_SPEC on any failure so a 24-category sweep never crashes."""
    user = (
        f"We are running Bayesian optimization to formulate a plant-based "
        f"{category} that matches a nutrition target ({target_block}).\n\n"
        "Setup you are configuring:\n"
        f"  * {k} ingredients; the decision variable is their mixing "
        "proportions.\n"
        "  * Internally, BO searches logit vectors z and maps them to "
        "proportions via softmax (x = softmax(z)), so the search space is a "
        "smooth reparametrization of the simplex.\n"
        "  * The objective is the negative per-nutrient z-score RMSD between "
        "the mixture's computed nutrition profile and the target. It is a "
        "DETERMINISTIC, noiseless function (nutrition is computed from a "
        "database, not measured).\n"
        f"  * Total budget is only {budget} evaluations, a handful of which "
        "are the initial design. Sample efficiency is everything.\n\n"
        "Choose GP/acquisition settings. Respond with ONLY a JSON object:\n"
        '{"kernel": <one of ["matern52","matern32","rbf","linear","poly2"]>, '
        '"lengthscale_prior": <one of ["default","long","short"]>, '
        '"noise": <one of ["default","low","fixed_tiny"]>, '
        '"acquisition": <one of ["qlognei","qlogei","qucb"]>}\n\n'
        "Reason about: how smooth the objective is under the softmax "
        "reparametrization; whether observation noise should be modeled given "
        "the objective is deterministic; and which acquisition suits a "
        "noiseless objective under a tiny budget. Output the JSON only."
    )
    try:
        raw = client.chat(_CONFIG_SYSTEM_PROMPT, user, temperature=temperature)
        spec = _extract_json(raw)
        return BOConfig.from_spec(spec, context=f"{category}")
    except Exception as exc:                          # noqa: BLE001
        warnings.warn(
            f"llm_select_bo_config({category!r}) failed "
            f"({type(exc).__name__}: {exc}); using DEFAULT_CONFIG_SPEC.",
            RuntimeWarning, stacklevel=2)
        return BOConfig.from_spec(DEFAULT_CONFIG_SPEC, context=f"{category}:fallback")
