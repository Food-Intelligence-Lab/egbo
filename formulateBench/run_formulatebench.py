"""FormulateBench: nutrition-matching experiments for plant-based meat and dairy.

For each animal-based reference category, find a simplex-constrained mixture
of plant-based ingredients whose nutrient profile is closest to the target
in z-score space, across the up to 8 nutrients (energy, protein, fat,
saturated fat, carbs, fiber, sugars, sodium) for which the target value is
reported in animal_products_nutrition.csv. Each nutrient is normalized
using the mean and standard deviation of the domain's ingredient ground
set, so all measured dimensions enter the loss with equal weight, and the
loss is the per-dim RMSD so categories with different numbers of measured
nutrients are comparable.

Methods compared on each category:
    llm           - LLM-selected k=9 ingredients, then BO
    vanilla       - BO over the full d-dim simplex
    random        - random simplex search
    turbo         - trust-region BO (Eriksson et al. 2019)
    saasbo        - sparse axis-aligned subspace BO (Eriksson & Jankowiak 2021)
    sebo          - sparsity-exploiting BO (Liu et al. 2023)
    rembo         - random embedding BO with k=9 (Wang et al. 2016)
    random_subset - random k=9 ingredient subset, then BO
    llm_optimizer - sequential LLM recipe proposals with objective feedback

The live LLM methods support two information regimes. ``full`` reproduces
the nutrition-aware prompts, while ``semantic-only`` reveals only the target
product category, ingredient names, and scalar scores from past evaluations.
The evaluated objective remains exactly the same nutrition-matching loss.

Targets come from data/animal_products_nutrition.csv. Ingredients come from
data/usda_nutrition_data.csv, filtered by the domain (in_meat/in_dairy).

Usage:
    python run_formulatebench.py --domain meat --name release_meat
    python run_formulatebench.py --domain dairy --categories yogurt --budget 5 --seeds 2
"""

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
import warnings
from datetime import datetime
from typing import Dict, List, Sequence, Tuple


def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
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


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(_HERE, "formulatebench", "data")

NUTRIENT_COLUMNS = [
    "energy_kcal_per_100g",
    "protein_g_per_100g",
    "fat_g_per_100g",
    "saturated_fat_g_per_100g",
    "carbs_g_per_100g",
    "fiber_g_per_100g",
    "sugars_g_per_100g",
    "sodium_mg_per_100g",
]

NUTRIENT_LABELS = {
    "energy_kcal_per_100g": "energy",
    "protein_g_per_100g": "protein",
    "fat_g_per_100g": "fat",
    "saturated_fat_g_per_100g": "sat_fat",
    "carbs_g_per_100g": "carbs",
    "fiber_g_per_100g": "fiber",
    "sugars_g_per_100g": "sugars",
    "sodium_mg_per_100g": "sodium",
}

DOMAIN_CONFIG = {
    "meat":  {"in_column": "in_meat",  "target_filter": "meat"},
    "dairy": {"in_column": "in_dairy", "target_filter": "dairy"},
}

# Claude Opus 4.7 plant-based-food-scientist ingredient selections (k=9 per
# category, k chosen via Proposition 1: m=8 nutrient targets + simplex
# constraint => sparse optimum exists with at most m+1=9 nonzeros).
LLM_SELECTIONS: Dict[str, Dict[str, List[str]]] = {
    "meat": {
        "bacon": ["coconut oil", "canola oil", "soy protein isolate", "textured vegetable protein", "mycoprotein", "methylcellulose", "liquid smoke", "soy sauce", "yeast extract"],
        "bratwurst": ["coconut oil", "canola oil", "soy protein isolate", "pea protein", "textured vegetable protein", "potato protein", "methylcellulose", "yeast extract", "mustard"],
        "breaded chicken fillets": ["soy protein isolate", "textured vegetable protein", "wheat gluten", "wheat flour", "corn starch", "methylcellulose", "canola oil", "mycoprotein", "yeast extract"],
        "breakfast sausage patties": ["coconut oil", "canola oil", "soy protein isolate", "textured vegetable protein", "pea protein", "methylcellulose", "yeast extract", "soy sauce", "mustard"],
        "burgers": ["soy protein isolate", "textured vegetable protein", "pea protein", "mycoprotein", "coconut oil", "canola oil", "methylcellulose", "yeast extract", "onion"],
        "chicken nuggets": ["soy protein isolate", "textured vegetable protein", "mycoprotein", "wheat gluten", "wheat flour", "corn starch", "methylcellulose", "canola oil", "yeast extract"],
        "deli slices (ham)": ["soy protein isolate", "wheat gluten", "potato protein", "mycoprotein", "methylcellulose", "soy sauce", "yeast extract", "liquid smoke", "onion"],
        "deli slices (turkey)": ["soy protein isolate", "wheat gluten", "potato protein", "mycoprotein", "methylcellulose", "soy sauce", "yeast extract", "liquid smoke", "onion"],
        "hot dogs": ["coconut oil", "canola oil", "soy protein isolate", "textured vegetable protein", "pea protein", "potato protein", "methylcellulose", "soy sauce", "yeast extract"],
        "meatballs": ["soy protein isolate", "textured vegetable protein", "coconut oil", "canola oil", "wheat flour", "methylcellulose", "onion", "garlic", "yeast extract"],
        "pulled pork": ["jackfruit", "soy protein isolate", "textured vegetable protein", "mycoprotein", "coconut oil", "canola oil", "liquid smoke", "methylcellulose", "yeast extract"],
        "steak fillets": ["wheat gluten", "textured vegetable protein", "mycoprotein", "mushroom mycelium", "pea protein", "canola oil", "coconut oil", "methylcellulose", "onion"],
        "unbreaded chicken fillet": ["wheat gluten", "textured vegetable protein", "mycoprotein", "mushroom mycelium", "pea protein", "potato protein", "canola oil", "methylcellulose", "onion"],
        "unbreaded chicken strips and chunks": ["soy protein isolate", "wheat gluten", "textured vegetable protein", "mycoprotein", "mushroom mycelium", "potato protein", "canola oil", "methylcellulose", "yeast extract"],
    },
    "dairy": {
        "barista milk": ["oat milk", "soymilk", "almond milk", "pea milk", "canola oil", "coconut oil", "pea protein", "gellan gum", "corn syrup solids"],
        "butter": ["coconut oil", "cocoa butter", "palm oil", "fully hydrogenated vegetable oils", "canola oil", "soy lecithin", "soy protein isolate", "carrageenan", "gellan gum"],
        "cheddar": ["coconut oil", "cocoa butter", "soy protein isolate", "soy protein concentrate", "pea protein", "cashews", "canola oil", "carrageenan", "gellan gum"],
        "coffee creamer": ["coconut cream", "oat cream", "coconut oil", "soymilk", "almond milk", "allulose", "corn syrup solids", "pea protein", "gellan gum"],
        "cream cheese": ["coconut oil", "cashew cream", "cocoa butter", "soy protein isolate", "potato protein", "canola oil", "allulose", "carrageenan", "gellan gum"],
        "ice cream": ["coconut cream", "oat cream", "coconut oil", "soymilk", "allulose", "corn syrup solids", "pea protein", "carrageenan", "gellan gum"],
        "milk": ["oat milk", "soymilk", "almond milk", "pea milk", "coconut oil", "canola oil", "pea protein", "corn syrup solids", "gellan gum"],
        "mozzarella": ["coconut oil", "cocoa butter", "soy protein isolate", "soy protein concentrate", "pea protein", "cashews", "canola oil", "carrageenan", "gellan gum"],
        "sour cream": ["coconut cream", "cashew cream", "coconut oil", "soymilk", "pea protein", "lemon juice", "allulose", "carrageenan", "gellan gum"],
        "yogurt": ["oat milk", "soymilk", "pea milk", "pea protein", "coconut oil", "inulin", "lemon juice", "carrageenan", "gellan gum"],
    },
}

ALL_METHOD_KEYS = ["llm", "vanilla", "random", "turbo", "saasbo", "sebo", "rembo", "random_subset", "adaptive_llm", "llm_optimizer"]
# Methods run by default when --methods is omitted. Live LLM methods require an
# OpenRouter key, so they are opt-in only.
_LIVE_LLM_METHOD_KEYS = {"adaptive_llm", "llm_optimizer"}
_DEFAULT_METHOD_KEYS = [k for k in ALL_METHOD_KEYS if k not in _LIVE_LLM_METHOD_KEYS]


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def load_ingredients(domain: str, data_dir: str = _DEFAULT_DATA_DIR) -> pd.DataFrame:
    in_column = DOMAIN_CONFIG[domain]["in_column"]
    df = pd.read_csv(os.path.join(data_dir, "usda_nutrition_data.csv"))
    df = df[df[in_column] == "Y"].copy()
    for col in NUTRIENT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["ingredient"] = df["ingredient"].fillna("").astype(str)
    df.reset_index(drop=True, inplace=True)
    return df


def load_targets(
    domain: str, data_dir: str = _DEFAULT_DATA_DIR,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Read animal_products_nutrition.csv and return {category: (target_vec, mask)}.

    `target_vec` has length 8 (one per nutrient). `mask` is a boolean array
    of the same length: True where the CSV has a value, False where it is
    NaN. The objective ignores nutrients with mask == False, so each
    category is matched only on the dimensions that were actually reported.

    NaN values in target_vec are replaced with 0 for safe array math; the
    mask is what controls inclusion in the loss.
    """
    target_filter = DOMAIN_CONFIG[domain]["target_filter"]
    df = pd.read_csv(os.path.join(data_dir, "animal_products_nutrition.csv"))
    df = df[df["category"] == target_filter].copy()
    for col in NUTRIENT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    targets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for _, row in df.iterrows():
        name = str(row["ingredient"])
        raw = np.array([row[col] for col in NUTRIENT_COLUMNS], dtype=float)
        mask = ~np.isnan(raw)
        if not mask.any():
            warnings.warn(
                f"target '{name}' skipped: no measured nutrients in CSV.",
                RuntimeWarning, stacklevel=2,
            )
            continue
        vec = np.where(mask, raw, 0.0)
        targets[name] = (vec, mask)
    return targets


def get_nutrition_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[NUTRIENT_COLUMNS].to_numpy(dtype=float)


def resolve_selections(
    selection_names: Sequence[str], df: pd.DataFrame
) -> List[int]:
    """Map ingredient names to row positions in the filtered+reset DataFrame."""
    name_to_idx = {str(name): i for i, name in enumerate(df["ingredient"])}
    missing = [n for n in selection_names if n not in name_to_idx]
    if missing:
        raise KeyError(f"Ingredients not found in domain: {missing}")
    return [name_to_idx[str(n)] for n in selection_names]


# ----------------------------------------------------------------------------
# Objective: z-score nutrition matching
# ----------------------------------------------------------------------------

def make_objective(
    nutrition_matrix: np.ndarray, target: np.ndarray, mask: np.ndarray,
):
    """Build the negative per-dim z-score RMSD across the masked nutrients.

    Statistics are computed over the domain's ingredient ground set so that
    target and any candidate formulation are normalized identically. Only
    nutrients with mask==True contribute to the loss, so categories whose
    CSV row is missing some nutrients are matched only on the dimensions
    that were actually reported.

    Returning RMSD per measured dimension (rather than raw L2) makes the
    objective comparable across categories with different mask sizes.
    """
    mu = nutrition_matrix.mean(axis=0)
    sigma = nutrition_matrix.std(axis=0)
    sigma = np.where(sigma > 0, sigma, 1.0)
    z_target = (target - mu) / sigma
    m = int(mask.sum())
    assert m > 0, "objective requires at least one measured nutrient"

    def objective(x: np.ndarray) -> float:
        profile = x @ nutrition_matrix
        z_profile = (profile - mu) / sigma
        diff = (z_profile - z_target)[mask]
        return -float(np.sqrt(np.mean(diff ** 2)))

    return objective


def nnls_simplex_oracle(
    nutrition_matrix: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    selected_indices: Sequence[int] | None = None,
) -> Tuple[np.ndarray, float]:
    """Closed-form floor for the FormulateBench objective.

    Solves
        min_x  sqrt(mean_M( (z_profile(x) - z_target)^2 ))
        s.t.   sum_i x_i = 1,  x_i >= 0

    where z_* are computed against the domain ingredient ground set, and
    M selects the measured nutrient dimensions for this category. Optionally
    restricts the support to `selected_indices` (e.g., the LLM's k=9 set).

    Returns (x_optimal, oracle_loss). The oracle_loss is in the same units
    as the BO trace (per-dim z-score RMSD), so it can be plotted as a
    horizontal floor on the convergence plot.
    """
    sigma = nutrition_matrix.std(axis=0)
    sigma = np.where(sigma > 0, sigma, 1.0)

    if selected_indices is None:
        active = np.arange(nutrition_matrix.shape[0])
    else:
        active = np.asarray(selected_indices, dtype=int)

    # The z-score loss is sqrt(mean_M(((profile - target)/sigma)^2)); the
    # mean mu cancels in the difference. So we minimize ||x@A - b||^2 with
    #   A = (nutrition_matrix[active] / sigma)[:, mask]
    #   b = target / sigma             (not (target - mu) / sigma)
    A = (nutrition_matrix[active] / sigma)[:, mask]   # (k, m_active)
    b = (target / sigma)[mask]                         # (m_active,)
    k = A.shape[0]
    m_active = A.shape[1]

    AtA = A @ A.T                # (k, k)
    Atb = A @ b                  # (k,)

    def f_and_grad(x):
        diff = x @ A - b                            # (m_active,)
        val = 0.5 * float(diff @ diff)
        grad = AtA @ x - Atb                        # (k,)
        return val, grad

    constraints = [{
        "type": "eq",
        "fun":  lambda x: np.sum(x) - 1.0,
        "jac":  lambda x: np.ones(k),
    }]
    bounds = [(0.0, None)] * k
    x0 = np.ones(k) / k

    res = minimize(
        f_and_grad, x0, jac=True, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    x_opt = np.clip(res.x, 0.0, None)
    s = x_opt.sum()
    if s > 0:
        x_opt = x_opt / s

    diff = x_opt @ A - b
    rmsd = float(np.sqrt(np.mean(diff ** 2))) if m_active > 0 else 0.0

    if selected_indices is None:
        x_full = x_opt
    else:
        x_full = np.zeros(nutrition_matrix.shape[0])
        x_full[active] = x_opt
    return x_full, rmsd


def best_subset_sample(
    nutrition_matrix: np.ndarray,
    selected: List[int],
    target: np.ndarray,
    mask: np.ndarray,
    n_samples: int = 4096,
) -> Tuple[np.ndarray, float]:
    """Best random Dirichlet draw over the selected subset, for diagnostics."""
    mu = nutrition_matrix.mean(axis=0)
    sigma = nutrition_matrix.std(axis=0)
    sigma = np.where(sigma > 0, sigma, 1.0)
    z_target = (target - mu) / sigma

    rng = np.random.RandomState(123)
    draws = rng.dirichlet(np.ones(len(selected)), size=n_samples)
    profiles = draws @ nutrition_matrix[selected]
    z_profiles = (profiles - mu) / sigma
    diffs = (z_profiles - z_target)[:, mask]
    dists = np.sqrt(np.mean(diffs ** 2, axis=1))
    best = int(np.argmin(dists))
    return profiles[best], float(dists[best])


# ----------------------------------------------------------------------------
# Optimizer building blocks
# ----------------------------------------------------------------------------

def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def _simplex_initial_logits(dim: int, n_init: int, seed: int) -> np.ndarray:
    """Uniform mixture + Dirichlet draws on the simplex, mapped back to logits."""
    rng = np.random.RandomState(seed)
    log_floor = np.exp(-12.0)
    simplex_points = [np.full(dim, 1.0 / dim)]
    for _ in range(max(0, n_init - 1)):
        simplex_points.append(rng.dirichlet(np.ones(dim)))
    simplex = np.array(simplex_points[:n_init], dtype=float)
    simplex = np.clip(simplex, log_floor, 1.0)
    simplex = simplex / simplex.sum(axis=1, keepdims=True)
    return np.log(simplex)


def _fit_gpytorch_mll_with_warning(mll, context: str) -> bool:
    try:
        fit_gpytorch_mll(mll)
        return True
    except Exception as exc:
        warnings.warn(
            f"{context}: GP fit failed ({type(exc).__name__}: {exc}). "
            "Falling back to a random candidate.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False


# ----------------------------------------------------------------------------
# OpenRouter expert (for adaptive EGBO)
# ----------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = "You are an expert plant-based food scientist."


def _throttle(model: str) -> None:
    """Optional cross-process rate limit for live API calls (cache hits skip it).

    Enabled only when the environment variable OPENROUTER_MAX_RPM is set (e.g.
    18 for new OpenRouter accounts, which allow 20 requests/minute per model).
    Uses a lock file per model so all worker processes share one budget.
    """
    rpm = os.environ.get("OPENROUTER_MAX_RPM")
    if not rpm:
        return
    import fcntl
    import tempfile
    import time
    interval = 60.0 / float(rpm)
    path = os.path.join(tempfile.gettempdir(), "openrouter_rl_" + re.sub(r"[^A-Za-z0-9]", "_", model))
    with open(path, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        try:
            last = float(fh.read().strip() or 0.0)
        except ValueError:
            last = 0.0
        wait = last + interval - time.time()
        if wait > 0:
            time.sleep(wait)
        fh.seek(0); fh.truncate(); fh.write(str(time.time())); fh.flush()
        fcntl.flock(fh, fcntl.LOCK_UN)


class OpenRouterClient:
    """Minimal HTTPS client for OpenRouter chat completions, with optional
    SHA-256-keyed disk cache for reproducibility across runs."""

    def __init__(
        self, api_key: str, model: str, timeout: float = 120.0,
        cache_dir: "str | None" = None, max_tokens: "int | None" = None,
    ):
        if not api_key:
            raise ValueError("OpenRouter API key is required.")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.cache_dir = cache_dir
        self.max_tokens = max_tokens
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_key(self, system: str, user: str, temperature: float) -> str:
        h = hashlib.sha256()
        h.update(self.model.encode("utf-8"))
        h.update(b"\0")
        h.update(system.encode("utf-8"))
        h.update(b"\0")
        h.update(user.encode("utf-8"))
        h.update(b"\0")
        h.update(f"{temperature:.6f}".encode("utf-8"))
        if self.max_tokens is not None:
            h.update(b"\0")
            h.update(f"max_tokens={self.max_tokens}".encode("utf-8"))
        return h.hexdigest()

    def _cache_path(self, key: str) -> str:
        # Two-level fanout to keep any single dir small if the cache grows.
        return os.path.join(self.cache_dir, key[:2], key + ".json")

    def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        if self.cache_dir is not None:
            key = self._cache_key(system, user, temperature)
            path = self._cache_path(key)
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        cached = json.load(f).get("response")
                    return cached if cached is not None else ""
                except Exception as exc:
                    warnings.warn(
                        f"OpenRouter cache read failed ({type(exc).__name__}: {exc}); "
                        "re-issuing live call.",
                        RuntimeWarning, stacklevel=2,
                    )

        _throttle(self.model)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        body_dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if self.max_tokens is not None:
            body_dict["max_tokens"] = int(self.max_tokens)
        body = json.dumps(body_dict).encode("utf-8")

        # Retry-with-backoff: rate limits / 5xx / network blips. Up to 5 tries
        # at 2, 4, 8, 16, 32 seconds. 4xx other than 429 fail immediately.
        last_exc = None
        for attempt in range(5):
            req = urllib.request.Request(
                self.url, data=body, method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.load(resp)
                break
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code == 429 or 500 <= exc.code < 600:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                time.sleep(2 ** (attempt + 1))
                continue
        else:
            raise RuntimeError(
                f"OpenRouterClient.chat: 5 consecutive failures; last={last_exc!r}"
            )
        response = payload["choices"][0]["message"].get("content")
        if response is None:
            # Reasoning models can return null content if the reasoning
            # budget consumed all output tokens. Treat as empty for
            # downstream parsers; cache the null so we don't re-spend.
            response = ""

        if self.cache_dir is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({
                    "model": self.model,
                    "system": system,
                    "user": user,
                    "temperature": temperature,
                    "max_tokens": self.max_tokens,
                    "response": response,
                }, f)
            os.replace(tmp, path)

        return response


def _ingredients_block(df: pd.DataFrame) -> str:
    lines = []
    for _, row in df.iterrows():
        parts = [
            f"{NUTRIENT_LABELS[col]}={float(row[col]):.1f}"
            for col in NUTRIENT_COLUMNS
        ]
        lines.append(f"- {row['ingredient']}: " + ", ".join(parts))
    return "\n".join(lines)


def _ingredient_names_block(df: pd.DataFrame) -> str:
    """Ingredient catalog with no numerical or descriptive side information."""
    return "\n".join(f"- {name}" for name in df["ingredient"])


def _target_block(target: np.ndarray, mask: np.ndarray) -> str:
    return ", ".join(
        f"{NUTRIENT_LABELS[col]}={float(target[j]):.1f}"
        for j, col in enumerate(NUTRIENT_COLUMNS) if mask[j]
    )


def _parse_ingredient_list(raw: str, df: pd.DataFrame, key: str) -> List[str]:
    """Pull a list of ingredient names from an LLM response, case-insensitively
    matched against `df['ingredient']`. Unknown names are dropped with a
    warning. Accepts either {"<key>": [...]} or a bare JSON array, optionally
    wrapped in markdown code fences or surrounded by prose."""
    items: List[str] = []
    # Strip markdown code fences if present (```json ... ``` or ``` ... ```).
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = cleaned.replace("```", "")
    # Try every {...} candidate from longest to shortest; the LLM may have
    # multiple braces (nested objects, embedded JSON-in-prose, etc.).
    obj_candidates = sorted(
        set(re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned, re.DOTALL)),
        key=len, reverse=True,
    )
    for cand in obj_candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        v = data.get(key, [])
        if isinstance(v, list) and v:
            items = [s for s in v if isinstance(s, str)]
            if items:
                break
    if not items:
        # Fallback: bare JSON array, longest-first.
        for cand in sorted(set(re.findall(r"\[[^\[\]]*\]", cleaned, re.DOTALL)),
                           key=len, reverse=True):
            try:
                v = json.loads(cand)
            except json.JSONDecodeError:
                continue
            if isinstance(v, list) and v:
                items = [s for s in v if isinstance(s, str)]
                if items:
                    break
    if not items:
        raise ValueError(f"Could not parse ingredient list from LLM response: {raw[:300]}")

    name_lookup = {str(n).lower(): str(n) for n in df["ingredient"]}
    resolved, missing = [], []
    for s in items:
        canonical = name_lookup.get(s.strip().lower())
        if canonical and canonical not in resolved:
            resolved.append(canonical)
        else:
            missing.append(s)
    if missing:
        warnings.warn(
            f"LLM returned {len(missing)} unknown ingredient(s); dropped: {missing[:5]}",
            RuntimeWarning, stacklevel=2,
        )
    return resolved


def _parse_recipe(
    raw: str, df: pd.DataFrame, max_ingredients: int = 9,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Parse and normalize an LLM-proposed formulation.

    The expected shape is ``{"recipe": [{"ingredient": "...",
    "fraction": 0.2}, ...]}``. Positive percentages or gram amounts are also
    accepted because quantities are normalized to sum to one.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "")
    data = None
    try:
        parsed = json.loads(cleaned.strip())
        if isinstance(parsed, dict):
            data = parsed.get("recipe")
    except json.JSONDecodeError:
        pass
    if not isinstance(data, list):
        # Greedy match is intentional: recipe entries contain nested objects.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    data = parsed.get("recipe")
            except json.JSONDecodeError:
                pass
    if not isinstance(data, list):
        raise ValueError("response did not contain a JSON recipe list")

    lookup = {str(name).strip().casefold(): i for i, name in enumerate(df["ingredient"])}
    amounts: Dict[int, float] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        idx = lookup.get(str(item.get("ingredient", "")).strip().casefold())
        if idx is None:
            continue
        raw_amount = item.get("fraction", item.get("amount", item.get("percentage")))
        try:
            amount = float(raw_amount)
        except (TypeError, ValueError):
            continue
        if np.isfinite(amount) and amount > 0:
            amounts[idx] = amounts.get(idx, 0.0) + amount

    if not amounts:
        raise ValueError("recipe contained no recognized positive ingredient amounts")
    if len(amounts) > max_ingredients:
        raise ValueError(
            f"recipe used {len(amounts)} ingredients; maximum is {max_ingredients}"
        )

    total = float(sum(amounts.values()))
    x = np.zeros(len(df), dtype=float)
    recipe: List[Dict[str, float]] = []
    for idx, amount in sorted(amounts.items(), key=lambda pair: -pair[1]):
        fraction = amount / total
        x[idx] = fraction
        recipe.append({
            "ingredient": str(df["ingredient"].iloc[idx]),
            "fraction": float(fraction),
        })
    return x, recipe


def iterative_llm_optimizer(
    objective, df: pd.DataFrame, category: str,
    target: np.ndarray, mask: np.ndarray, nutrition_matrix: np.ndarray,
    budget: int, llm_client: OpenRouterClient,
    max_ingredients: int = 9, temperature: float = 0.0,
    information_regime: str = "full",
    initial_x: "np.ndarray | None" = None,
    initial_recipe: "List[Dict[str, float]] | None" = None,
    feedback_note: "str | None" = None,
) -> Dict:
    """Sequential LLM-only optimizer with the same evaluation budget as BO.

    ``feedback_note`` (semantic-only regime) is appended after the feedback
    block, e.g. to tell the model that scores are noisy. ``None`` leaves the
    prompt byte-identical to the paper's protocol."""
    if llm_client is None:
        raise ValueError("iterative_llm_optimizer requires an OpenRouterClient.")
    if information_regime not in {"full", "semantic-only"}:
        raise ValueError(f"Unknown LLM information regime: {information_regime}")

    history: List[Dict[str, object]] = []
    all_y: List[float] = []
    best_values: List[float] = []
    ingredient_block = (
        _ingredients_block(df)
        if information_regime == "full"
        else _ingredient_names_block(df)
    )
    target_block = _target_block(target, mask)

    if initial_x is not None:
        initial_x = np.asarray(initial_x, dtype=float)
        if initial_x.shape != (len(df),):
            raise ValueError(
                f"initial_x has shape {initial_x.shape}; expected {(len(df),)}"
            )
        if np.any(initial_x < 0) or not np.isfinite(initial_x).all():
            raise ValueError("initial_x must be finite and nonnegative")
        total = float(initial_x.sum())
        if total <= 0:
            raise ValueError("initial_x must have positive mass")
        initial_x = initial_x / total
        if initial_recipe is None:
            initial_recipe = [
                {
                    "ingredient": str(df["ingredient"].iloc[i]),
                    "fraction": float(initial_x[i]),
                }
                for i in np.flatnonzero(initial_x > 0)
            ]
        y0 = float(objective(initial_x))
        profile0 = initial_x @ nutrition_matrix
        first_record: Dict[str, object] = {
            "iteration": 1,
            "recipe": initial_recipe,
            "loss": -y0,
        }
        if information_regime == "full":
            first_record["profile"] = {
                NUTRIENT_LABELS[col]: float(profile0[j])
                for j, col in enumerate(NUTRIENT_COLUMNS) if mask[j]
            }
        history.append(first_record)
        all_y.append(y0)
        best_values.append(y0)
        print(
            f"  [llm_optimizer] shared initialization 1/{budget}: "
            f"z-rmsd={-y0:.4f}", flush=True,
        )

    for step in range(len(history), budget):
        if information_regime == "full":
            if history:
                feedback = (
                    "Previous evaluated recipes follow. Loss is z-score RMSD, so "
                    "lower is better. Use these outcomes to propose a new recipe "
                    "that improves the best loss.\n" +
                    json.dumps(history, separators=(",", ":"))
                )
            else:
                feedback = (
                    "No recipes have been evaluated yet. Propose the strongest "
                    "initial recipe using food-science and nutrition knowledge."
                )
            prompt = f"""You are optimizing a plant-based formulation that matches the nutrition of animal-based {category}.

Target measured nutrients: {target_block}

Available plant-based ingredients and nutrient profiles per 100 g:
{ingredient_block}

{feedback}

Propose exactly one complete formulation. Use at most {max_ingredients} available ingredients, nonnegative quantities, and no ingredient outside the catalog. Return only JSON in this exact shape:
{{"recipe":[{{"ingredient":"exact catalog name","fraction":0.25}}]}}
Fractions must sum to 1. Do not include commentary."""
        else:
            if history:
                scalar_history = [
                    {
                        "iteration": item["iteration"],
                        "recipe": item["recipe"],
                        "target_dissimilarity": item["loss"],
                    }
                    for item in history
                ]
                feedback = (
                    "Previous evaluated formulations follow. "
                    "Target dissimilarity is the evaluator's scalar output; "
                    "lower is better. Use only these observations to propose "
                    "the next formulation.\n" +
                    json.dumps(scalar_history, separators=(",", ":"))
                )
            else:
                feedback = (
                    "No formulations have been evaluated yet. Propose the "
                    "strongest initial formulation using your prior knowledge."
                )
            if feedback_note:
                feedback = f"{feedback}\n\n{feedback_note}"
            prompt = f"""You are optimizing the overall similarity of a plant-based formulation to animal-based {category}. The evaluator is a black box.

Available plant-based ingredients:
{ingredient_block}

{feedback}

Propose exactly one complete formulation. Use at most {max_ingredients} available ingredients, nonnegative quantities, and no ingredient outside the catalog. Return only JSON in this exact shape:
{{"recipe":[{{"ingredient":"exact catalog name","fraction":0.25}}]}}
Fractions must sum to 1. Do not include commentary."""

        raw = llm_client.chat(_LLM_SYSTEM_PROMPT, prompt, temperature=temperature)
        parse_error = None
        for repair_attempt in range(3):
            try:
                x, recipe = _parse_recipe(
                    raw, df, max_ingredients=max_ingredients,
                )
                break
            except ValueError as exc:
                parse_error = exc
                if repair_attempt == 2:
                    raise
                repair_prompt = f"""Your proposed recipe was invalid: {exc}.

Invalid response:
{raw}

Correct it without changing the optimization goal. Use at most {max_ingredients} ingredients from the catalog, with nonnegative fractions summing to 1. Return only:
{{"recipe":[{{"ingredient":"exact catalog name","fraction":0.25}}]}}"""
                raw = llm_client.chat(
                    _LLM_SYSTEM_PROMPT, repair_prompt, temperature=temperature,
                )
        else:  # pragma: no cover - loop either breaks or raises
            raise parse_error or ValueError("could not parse recipe")
        y = float(objective(x))
        profile = x @ nutrition_matrix
        record = {
            "iteration": step + 1,
            "recipe": recipe,
            "loss": -y,
        }
        if information_regime == "full":
            record["profile"] = {
                NUTRIENT_LABELS[col]: float(profile[j])
                for j, col in enumerate(NUTRIENT_COLUMNS) if mask[j]
            }
        history.append(record)
        all_y.append(y)
        best_values.append(max(all_y))
        print(
            f"  [llm_optimizer] iteration {step + 1}/{budget}: "
            f"z-rmsd={-y:.4f}, best={-max(all_y):.4f}", flush=True,
        )

    return {
        "best_values": best_values,
        "all_y": all_y,
        "final_best": max(all_y),
        "recipe_history": history,
    }


def _llm_select_initial(
    client: OpenRouterClient, category: str, df: pd.DataFrame,
    target: np.ndarray, mask: np.ndarray, k: int,
    information_regime: str = "full",
) -> List[str]:
    """Round-0 expert query, matching the prompt in EGBO Appendix C."""
    if information_regime == "full":
        user = (
            "You have available to you the following set of ingredients:\n"
            f"{_ingredients_block(df)}\n\n"
            f"Your goal is to match the nutritional properties of animal-based "
            f"{category}: {_target_block(target, mask)}.\n\n"
            f"Choose a subset of at most {k} ingredients, on which we will run "
            "Bayesian optimization to iteratively determine ingredient "
            "concentrations.\n\n"
            "Respond with a JSON object: "
            "{\"ingredients\": [\"name1\", \"name2\", ...]}. "
            "Use only ingredient names from the list above, exactly as written."
        )
    elif information_regime == "semantic-only":
        user = (
            "You are optimizing the overall similarity of a plant-based "
            f"formulation to animal-based {category}.\n\n"
            "Available plant-based ingredients:\n"
            f"{_ingredient_names_block(df)}\n\n"
            f"Choose a subset of at most {k} ingredients, on which we will run "
            "Bayesian optimization to determine ingredient concentrations.\n\n"
            "Respond with a JSON object: "
            "{\"ingredients\": [\"name1\", \"name2\", ...]}. "
            "Use only ingredient names from the list above, exactly as written."
        )
    else:
        raise ValueError(f"Unknown LLM information regime: {information_regime}")
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            return _parse_ingredient_list(
                client.chat(_LLM_SYSTEM_PROMPT, user), df, key="ingredients",
            )
        except ValueError as exc:
            last_exc = exc
            warnings.warn(
                f"_llm_select_initial({category!r}, attempt={attempt+1}/3) "
                f"parse failed: {exc}; retrying.",
                RuntimeWarning, stacklevel=2,
            )
    assert last_exc is not None
    raise last_exc


def _llm_suggest_additions(
    client: OpenRouterClient, category: str, df: pd.DataFrame,
    target: np.ndarray, mask: np.ndarray,
    nutrition_matrix: np.ndarray,
    selected_names: List[str], selected_indices: List[int],
    all_z: List[np.ndarray], all_y: List[float],
    k_add: int,
) -> List[str]:
    """Round-q (q>0) expert query: additions A_{q+1} subset of [N] minus S_q.

    The prompt includes the full BO trajectory so far (composition + achieved
    nutrient profile + per-nutrient gap to target + z-RMSD for every
    evaluation), so the LLM can reason about which nutrients the active set
    cannot reach and propose ingredients that fill those gaps.
    """
    available_df = df[~df["ingredient"].isin(selected_names)]
    d_full = len(df)
    ingredient_names = df["ingredient"].tolist()

    def _z_to_x(z: np.ndarray) -> np.ndarray:
        fracs = _softmax(np.asarray(z, dtype=float))
        x = np.zeros(d_full)
        # Active subset at the time of this evaluation = selected_indices[:len(z)]
        sel_at_t = selected_indices[: len(fracs)]
        for i, idx in enumerate(sel_at_t):
            x[idx] = fracs[i]
        return x

    nutrient_short = {
        "energy": "en", "protein": "pr", "fat": "fa", "sat_fat": "sf",
        "carbs": "ca", "fiber": "fi", "sugars": "su", "sodium": "so",
    }
    short_names = [nutrient_short[NUTRIENT_LABELS[c]] for c in NUTRIENT_COLUMNS]

    # Build trajectory: every evaluation as one row.
    trajectory_lines = []
    best_idx = int(np.argmax(all_y))
    for i, (z, y) in enumerate(zip(all_z, all_y)):
        x = _z_to_x(z)
        profile = x @ nutrition_matrix
        gap = target - profile
        rmsd = -float(y)
        # Composition: non-zero, sorted by weight, top 8.
        nonzero = [(j, float(x[j])) for j in range(d_full) if x[j] > 1e-3]
        nonzero.sort(key=lambda kv: -kv[1])
        comp = ", ".join(f"{ingredient_names[j]}={w:.2f}" for j, w in nonzero[:8])
        prof_str = ", ".join(f"{short_names[k]}={profile[k]:.0f}" for k in range(len(NUTRIENT_COLUMNS)))
        gap_str = ", ".join(
            f"{short_names[k]}={gap[k]:+.0f}"
            for k in range(len(NUTRIENT_COLUMNS)) if mask[k]
        )
        marker = "*" if i == best_idx else " "
        trajectory_lines.append(
            f"  {marker} iter {i+1:2d} | rmsd={rmsd:.4f} | comp: {comp} | "
            f"profile: {prof_str} | gap (tgt−achieved): {gap_str}"
        )
    trajectory = "\n".join(trajectory_lines)

    # Best-so-far details
    best_z = all_z[best_idx]
    best_x = _z_to_x(best_z)
    best_profile = best_x @ nutrition_matrix
    best_gap = target - best_profile
    best_rmsd = -float(all_y[best_idx])
    best_comp = ", ".join(
        f"{ingredient_names[j]}={float(best_x[j]):.3f}"
        for j in np.argsort(-best_x)[:10] if best_x[j] > 1e-3
    )
    best_gap_str = ", ".join(
        f"{NUTRIENT_LABELS[c]}={best_gap[k]:+.1f}"
        for k, c in enumerate(NUTRIENT_COLUMNS) if mask[k]
    )

    user = (
        f"You are running Bayesian optimization (BO) for an animal-based "
        f"{category} target ({_target_block(target, mask)}).\n\n"
        f"Current active ingredient set ({len(selected_names)}): "
        f"{selected_names}.\n\n"
        f"Best mixture found so far on this set has z-score nutrition RMSD = "
        f"{best_rmsd:.4f}, with composition: {best_comp}.\n"
        f"Per-nutrient gap (target − achieved) for the best-so-far: "
        f"{best_gap_str}.\n\n"
        f"Optimization trajectory (* marks current best):\n{trajectory}\n\n"
        f"Available ingredients to add (excluding the active set):\n"
        f"{_ingredients_block(available_df)}\n\n"
        f"Suggest up to {k_add} additional ingredients to expand the active "
        f"set so the BO can find a closer match. Past observations remain "
        f"valid (the new ingredients had zero concentration in those "
        f"mixtures), so only add ingredients that meaningfully extend what "
        f"the active set can express.\n\n"
        f"Respond with a JSON object: {{\"additions\": [\"name1\", ...]}}. "
        f"Use only ingredient names from the available list, exactly as "
        f"written. You may return an empty list if the active set is "
        f"already sufficient."
    )
    return _parse_ingredient_list(
        client.chat(_LLM_SYSTEM_PROMPT, user), df, key="additions",
    )


# ----------------------------------------------------------------------------
# Optimizers
# ----------------------------------------------------------------------------

def random_search(objective, d: int, budget: int, seed: int) -> Dict:
    rng = np.random.RandomState(seed)
    best_values, all_y = [], []
    for _ in range(budget):
        x = rng.dirichlet(np.ones(d))
        y = objective(x)
        all_y.append(y)
        best_values.append(max(all_y))
    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def vanilla_bo(
    objective, d: int, budget: int, n_init: int, seed: int,
    initial_x: "np.ndarray | None" = None,
) -> Dict:
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    z_lo, z_hi = -12.0, 0.0
    bounds = torch.tensor([[z_lo] * d, [z_hi] * d], dtype=torch.double)

    all_z, all_y, best_values = [], [], []

    if initial_x is None:
        initial_points = [_softmax(z) for z in _simplex_initial_logits(d, n_init, seed)]
    else:
        x0 = np.asarray(initial_x, dtype=float)
        if x0.shape != (d,) or np.any(x0 < 0) or not np.isfinite(x0).all():
            raise ValueError("initial_x must be a finite nonnegative vector of length d")
        x0 = x0 / x0.sum()
        initial_points = [x0]
        initial_points.extend(
            rng.dirichlet(np.ones(d)) for _ in range(max(0, n_init - 1))
        )
    for x in initial_points:
        x_for_logits = np.clip(x, np.exp(-12.0), 1.0)
        x_for_logits = x_for_logits / x_for_logits.sum()
        z = np.log(x_for_logits)
        y = objective(x)
        all_z.append(z)
        all_y.append(y)
        best_values.append(max(all_y))

    for _ in range(budget - n_init):
        train_Z = torch.tensor(np.array(all_z), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        train_Z_norm = normalize(train_Z, bounds)

        gp = SingleTaskGP(train_Z_norm, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        if not _fit_gpytorch_mll_with_warning(mll, f"vanilla_bo(seed={seed})"):
            z_new = rng.uniform(z_lo, z_hi, size=d)
            y_new = objective(_softmax(z_new))
            all_z.append(z_new)
            all_y.append(y_new)
            best_values.append(max(all_y))
            continue

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        acq = qLogNoisyExpectedImprovement(model=gp, X_baseline=train_Z_norm, sampler=sampler)
        cand, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack([torch.zeros(d), torch.ones(d)]).double(),
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        z_new = unnormalize(cand.squeeze(0), bounds).detach().numpy()
        y_new = objective(_softmax(z_new))
        all_z.append(z_new)
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def reduced_bo(
    objective, d_full: int, selected_indices: List[int],
    budget: int, n_init: int, seed: int,
    initial_x: "np.ndarray | None" = None,
    initial_design: str = "global",
    local_sigma: float = 0.35,
) -> Dict:
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    k = len(selected_indices)
    z_lo, z_hi = -12.0, 0.0
    bounds = torch.tensor([[z_lo] * k, [z_hi] * k], dtype=torch.double)

    def z_to_x(z):
        fracs = _softmax(z)
        x = np.zeros(d_full)
        for i, idx in enumerate(selected_indices):
            x[idx] = fracs[i]
        return x

    all_z, all_y, best_values = [], [], []

    if initial_design not in {"global", "local"}:
        raise ValueError("initial_design must be 'global' or 'local'")

    if initial_x is None:
        initial_points = [_softmax(z) for z in _simplex_initial_logits(k, n_init, seed)]
    else:
        x0_full = np.asarray(initial_x, dtype=float)
        if x0_full.shape != (d_full,) or np.any(x0_full < 0) or not np.isfinite(x0_full).all():
            raise ValueError(
                "initial_x must be a finite nonnegative vector of length d_full"
            )
        outside = np.delete(x0_full, selected_indices)
        if np.any(outside > 1e-12):
            raise ValueError("initial_x has positive mass outside selected_indices")
        x0 = x0_full[selected_indices]
        x0 = x0 / x0.sum()
        initial_points = [x0]
        if initial_design == "local":
            x0_clip = np.clip(x0, np.exp(-12.0), 1.0)
            x0_clip /= x0_clip.sum()
            z0 = np.log(x0_clip)
            for _ in range(max(0, n_init - 1)):
                z = np.clip(
                    z0 + rng.normal(0.0, local_sigma, size=k), z_lo, z_hi,
                )
                initial_points.append(_softmax(z))
        else:
            initial_points.extend(
                rng.dirichlet(np.ones(k)) for _ in range(max(0, n_init - 1))
            )
    for fracs in initial_points:
        fracs_for_logits = np.clip(fracs, np.exp(-12.0), 1.0)
        fracs_for_logits = fracs_for_logits / fracs_for_logits.sum()
        z = np.log(fracs_for_logits)
        x = np.zeros(d_full)
        x[selected_indices] = fracs
        y = objective(x)
        all_z.append(z)
        all_y.append(y)
        best_values.append(max(all_y))

    for _ in range(budget - n_init):
        train_Z = torch.tensor(np.array(all_z), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        train_Z_norm = normalize(train_Z, bounds)

        gp = SingleTaskGP(train_Z_norm, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        if not _fit_gpytorch_mll_with_warning(mll, f"reduced_bo(seed={seed}, k={k})"):
            z_new = rng.uniform(z_lo, z_hi, size=k)
            y_new = objective(z_to_x(z_new))
            all_z.append(z_new)
            all_y.append(y_new)
            best_values.append(max(all_y))
            continue

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        acq = qLogNoisyExpectedImprovement(model=gp, X_baseline=train_Z_norm, sampler=sampler)
        cand, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack([torch.zeros(k), torch.ones(k)]).double(),
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        z_new = unnormalize(cand.squeeze(0), bounds).detach().numpy()
        y_new = objective(z_to_x(z_new))
        all_z.append(z_new)
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def random_subset_bo(
    objective, d_full: int, k: int, budget: int, n_init: int, seed: int,
) -> Dict:
    subset_rng = np.random.RandomState(seed + 10_000)
    selected = sorted(subset_rng.choice(d_full, size=k, replace=False).tolist())
    return reduced_bo(objective, d_full, selected, budget=budget, n_init=n_init, seed=seed)


def rembo_bo(
    objective, d: int, budget: int, n_init: int, seed: int, target_dim: int = 9,
) -> Dict:
    torch.manual_seed(seed)
    k = target_dim
    z_lo, z_hi = -3.0, 3.0
    bounds_full = torch.tensor([[z_lo] * d, [z_hi] * d], dtype=torch.double)

    rng = np.random.RandomState(seed)
    A = torch.tensor(rng.randn(d, k) / np.sqrt(k), dtype=torch.double)
    x_center = torch.zeros(d, dtype=torch.double)
    z_radius = float(np.sqrt(k))
    z_bounds = torch.tensor([[-z_radius] * k, [z_radius] * k], dtype=torch.double)

    def z_to_x(z):
        x_unc = x_center + A @ z
        x_unc = torch.clamp(x_unc, bounds_full[0, 0], bounds_full[1, 0])
        return _softmax(x_unc.detach().numpy())

    all_z, all_y, best_values = [], [], []

    sobol_z = SobolEngine(dimension=k, scramble=True, seed=seed)
    z_init = unnormalize(sobol_z.draw(n_init).double(), z_bounds)
    for i in range(n_init):
        y = objective(z_to_x(z_init[i]))
        all_z.append(z_init[i].numpy())
        all_y.append(y)
        best_values.append(max(all_y))

    for _ in range(budget - n_init):
        train_Z = torch.tensor(np.array(all_z), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        train_Z_norm = normalize(train_Z, z_bounds)

        gp = SingleTaskGP(train_Z_norm, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        if not _fit_gpytorch_mll_with_warning(mll, f"rembo_bo(seed={seed}, k={k})"):
            z_new = rng.uniform(-z_radius, z_radius, size=k)
            y_new = objective(z_to_x(torch.tensor(z_new, dtype=torch.double)))
            all_z.append(np.array(z_new, dtype=float))
            all_y.append(y_new)
            best_values.append(max(all_y))
            continue

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        acq = qLogNoisyExpectedImprovement(model=gp, X_baseline=train_Z_norm, sampler=sampler)
        cand, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack([torch.zeros(k), torch.ones(k)]).double(),
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        z_new = unnormalize(cand.squeeze(0), z_bounds)
        y_new = objective(z_to_x(z_new))
        all_z.append(z_new.detach().numpy())
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def turbo_bo(objective, d: int, budget: int, n_init: int, seed: int) -> Dict:
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    z_lo, z_hi = -12.0, 0.0
    bounds = torch.tensor([[z_lo] * d, [z_hi] * d], dtype=torch.double)

    length_init, length_min, length_max = 0.8, 0.5 ** 7, 1.6
    # TuRBO defaults (Eriksson et al. 2019, §3): failure tolerance scales
    # with effective dimension as max(4, ceil(d_e / batch_size)). With q=1,
    # this reduces to max(4, d). The previous hardcoded value of 4 caused
    # the trust region to collapse far too quickly in high dimension.
    success_tolerance, failure_tolerance = 3, max(4, d)
    length, n_successes, n_failures = length_init, 0, 0

    all_z, all_y, best_values = [], [], []

    z_init = _simplex_initial_logits(d, n_init, seed)
    for i in range(n_init):
        z = z_init[i]
        y = objective(_softmax(z))
        all_z.append(z)
        all_y.append(y)
        best_values.append(max(all_y))

    best_y = max(all_y)

    for _ in range(budget - n_init):
        train_Z = torch.tensor(np.array(all_z), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        train_Z_norm = normalize(train_Z, bounds)

        gp = SingleTaskGP(train_Z_norm, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        if not _fit_gpytorch_mll_with_warning(mll, f"turbo_bo(seed={seed})"):
            z_new = rng.uniform(z_lo, z_hi, size=d)
            y_new = objective(_softmax(z_new))
            all_z.append(z_new)
            all_y.append(y_new)
            best_values.append(max(all_y))
            continue

        best_idx = int(np.argmax(all_y))
        z_center_norm = train_Z_norm[best_idx]

        covar = gp.covar_module
        if hasattr(covar, "base_kernel"):
            lengthscales = covar.base_kernel.lengthscale.detach().squeeze()
        else:
            lengthscales = covar.lengthscale.detach().squeeze()
        lengthscales = lengthscales.clamp_min(1e-12)
        weights = lengthscales.mean() / lengthscales
        weights = weights / torch.prod(weights).pow(1.0 / d)

        tr_lb = torch.clamp(z_center_norm - weights * length / 2, 0.0, 1.0)
        tr_ub = torch.clamp(z_center_norm + weights * length / 2, 0.0, 1.0)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        acq = qLogNoisyExpectedImprovement(model=gp, X_baseline=train_Z_norm, sampler=sampler)
        cand, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack([tr_lb, tr_ub]).double(),
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        z_new = unnormalize(cand.squeeze(0), bounds).detach().numpy()
        y_new = objective(_softmax(z_new))
        all_z.append(z_new)
        all_y.append(y_new)
        best_values.append(max(all_y))

        if y_new > best_y:
            n_successes += 1
            n_failures = 0
            best_y = y_new
        else:
            n_successes = 0
            n_failures += 1

        if n_successes >= success_tolerance:
            length = min(2.0 * length, length_max)
            n_successes = 0
        elif n_failures >= failure_tolerance:
            length = length / 2.0
            n_failures = 0

        if length < length_min:
            length = length_init
            n_successes = 0
            n_failures = 0

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def saasbo_bo(
    objective, d: int, budget: int, n_init: int, seed: int,
    warmup_steps: int = 512, num_samples: int = 256,
) -> Dict:
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    z_lo, z_hi = -12.0, 0.0
    bounds = torch.tensor([[z_lo] * d, [z_hi] * d], dtype=torch.double)

    all_z, all_y, best_values = [], [], []

    z_init = _simplex_initial_logits(d, n_init, seed)
    for i in range(n_init):
        z = z_init[i]
        y = objective(_softmax(z))
        all_z.append(z)
        all_y.append(y)
        best_values.append(max(all_y))

    for _ in range(budget - n_init):
        train_Z = torch.tensor(np.array(all_z), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        train_Z_norm = normalize(train_Z, bounds)

        gp = SaasFullyBayesianSingleTaskGP(train_X=train_Z_norm, train_Y=train_Y)
        try:
            fit_fully_bayesian_model_nuts(
                gp, warmup_steps=warmup_steps, num_samples=num_samples, disable_progbar=True,
            )
            fit_ok = True
        except Exception as exc:
            warnings.warn(
                f"saasbo_bo(seed={seed}): NUTS fit failed "
                f"({type(exc).__name__}: {exc}). Falling back to SingleTaskGP.",
                RuntimeWarning,
                stacklevel=2,
            )
            gp = SingleTaskGP(train_Z_norm, train_Y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_ok = _fit_gpytorch_mll_with_warning(mll, f"saasbo_bo(seed={seed}) fallback")

        if not fit_ok:
            z_new = rng.uniform(z_lo, z_hi, size=d)
            y_new = objective(_softmax(z_new))
            all_z.append(z_new)
            all_y.append(y_new)
            best_values.append(max(all_y))
            continue

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        acq = qLogNoisyExpectedImprovement(model=gp, X_baseline=train_Z_norm, sampler=sampler)
        cand, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack([torch.zeros(d), torch.ones(d)]).double(),
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        z_new = unnormalize(cand.squeeze(0), bounds).detach().numpy()
        y_new = objective(_softmax(z_new))
        all_z.append(z_new)
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


def sebo_bo(
    objective, d: int, budget: int, n_init: int, seed: int,
    lambda_max: float = 1.0, l0_a: float = 0.05,
    warmup_steps: int = 512, num_samples: int = 256,
) -> Dict:
    """SEBO (Liu et al. 2023): SAAS prior + smooth-L0 penalty with linear homotopy."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    z_lo, z_hi = -12.0, 0.0
    bounds = torch.tensor([[z_lo] * d, [z_hi] * d], dtype=torch.double)

    all_z, all_y, best_values = [], [], []

    z_init = _simplex_initial_logits(d, n_init, seed)
    for i in range(n_init):
        z = z_init[i]
        y = objective(_softmax(z))
        all_z.append(z)
        all_y.append(y)
        best_values.append(max(all_y))

    a_squared = float(l0_a) ** 2

    def sparsity_penalty(X: torch.Tensor) -> torch.Tensor:
        z = X * (z_hi - z_lo) + z_lo
        x = torch.softmax(z, dim=-1)
        active = 1.0 - torch.exp(-x.pow(2) / a_squared)
        return active.sum(dim=(-1, -2))

    n_bo_steps = budget - n_init
    for step in range(n_bo_steps):
        train_Z = torch.tensor(np.array(all_z), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        train_Z_norm = normalize(train_Z, bounds)

        gp = SaasFullyBayesianSingleTaskGP(train_X=train_Z_norm, train_Y=train_Y)
        try:
            fit_fully_bayesian_model_nuts(
                gp, warmup_steps=warmup_steps, num_samples=num_samples, disable_progbar=True,
            )
            fit_ok = True
        except Exception as exc:
            warnings.warn(
                f"sebo_bo(seed={seed}): NUTS fit failed "
                f"({type(exc).__name__}: {exc}). Falling back to SingleTaskGP.",
                RuntimeWarning,
                stacklevel=2,
            )
            gp = SingleTaskGP(train_Z_norm, train_Y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_ok = _fit_gpytorch_mll_with_warning(mll, f"sebo_bo(seed={seed}) fallback")

        if not fit_ok:
            z_new = rng.uniform(z_lo, z_hi, size=d)
            y_new = objective(_softmax(z_new))
            all_z.append(z_new)
            all_y.append(y_new)
            best_values.append(max(all_y))
            continue

        progress = (step + 1) / max(1, n_bo_steps)
        lambda_t = lambda_max * progress

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        base_acq = qLogNoisyExpectedImprovement(
            model=gp, X_baseline=train_Z_norm, sampler=sampler,
        )
        penalized_acq = _PenalizedAcq(base_acq, sparsity_penalty, lambda_t)
        cand, _ = optimize_acqf(
            acq_function=penalized_acq,
            bounds=torch.stack([torch.zeros(d), torch.ones(d)]).double(),
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        z_new = unnormalize(cand.squeeze(0), bounds).detach().numpy()
        y_new = objective(_softmax(z_new))
        all_z.append(z_new)
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def adaptive_llm_bo(
    objective, df: pd.DataFrame, category: str,
    target: np.ndarray, mask: np.ndarray,
    budget: int, n_init: int, seed: int,
    llm_client: OpenRouterClient,
    n_rounds: int = 3, k_init: int = 5, k_add: int = 2,
    initial_selection: "List[str] | None" = None,
    nutrition_matrix: "np.ndarray | None" = None,
) -> Dict:
    """Adaptive EGBO (Algorithm 1, Q>1).

    Q rounds of expert query with monotonic active-set growth: round 0 asks
    the LLM for an initial subset S_0, then for each q in 0..Q-1 we run
    tau = (budget-n_init)/Q BO steps in span(S_q), and (for q < Q-1) query
    the LLM for additions A_{q+1} given the current best mixture. Old
    observations remain valid because newly added ingredients had zero
    concentration in earlier formulations -- in logits we represent that as
    z = z_lo (softmax floor).
    """
    if llm_client is None:
        raise ValueError("adaptive_llm_bo requires an OpenRouterClient.")
    if nutrition_matrix is None:
        nutrition_matrix = get_nutrition_matrix(df)
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    d_full = len(df)
    z_lo, z_hi = -12.0, 0.0

    if initial_selection is not None:
        # Round-0 picks fixed (typically: the cached LLM_SELECTIONS used by
        # the static `llm` method, so adaptive's round-0 is identical to
        # one-shot through the first BO round).
        selected_names = list(initial_selection)[:k_init]
    else:
        selected_names = _llm_select_initial(llm_client, category, df, target, mask, k=k_init)
    if not selected_names:
        raise RuntimeError(
            f"adaptive_llm_bo(seed={seed}): LLM returned no ingredients for {category!r}."
        )
    selected = resolve_selections(selected_names, df)
    print(
        f"  [adaptive_llm seed={seed}] round 0 |S|={len(selected_names)}: "
        f"{selected_names}",
        flush=True,
    )

    def _z_to_x(z_arr: np.ndarray, sel: List[int]) -> np.ndarray:
        fracs = _softmax(np.asarray(z_arr, dtype=float))
        x = np.zeros(d_full)
        for i, idx in enumerate(sel):
            x[idx] = fracs[i]
        return x

    all_z: List[np.ndarray] = []
    all_y: List[float] = []
    best_values: List[float] = []

    z_init = _simplex_initial_logits(len(selected), n_init, seed)
    for i in range(n_init):
        z = np.asarray(z_init[i], dtype=float)
        y = objective(_z_to_x(z, selected))
        all_z.append(z)
        all_y.append(y)
        best_values.append(max(all_y))

    bo_left = max(0, budget - n_init)
    # Clamp n_rounds at bo_left so each round runs >= 1 BO step. Without this,
    # n_rounds > bo_left causes consecutive expert queries with zero BO
    # feedback between them (active set grows on stale trajectory). Q=T is
    # naturally bo_left (one query per BO step).
    n_rounds = max(1, min(int(n_rounds), bo_left)) if bo_left > 0 else 1
    base = bo_left // n_rounds
    rounds_budget = [base] * n_rounds
    rounds_budget[-1] += bo_left - base * n_rounds

    for q, n_steps in enumerate(rounds_budget):
        for _ in range(n_steps):
            k_now = len(selected)
            padded = np.full((len(all_z), k_now), z_lo, dtype=float)
            for r, z in enumerate(all_z):
                padded[r, : len(z)] = z
            train_Z = torch.tensor(padded, dtype=torch.double)
            train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
            bounds = torch.tensor([[z_lo] * k_now, [z_hi] * k_now], dtype=torch.double)
            train_Z_norm = normalize(train_Z, bounds)

            gp = SingleTaskGP(train_Z_norm, train_Y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            if not _fit_gpytorch_mll_with_warning(
                mll, f"adaptive_llm_bo(seed={seed}, k={k_now}, q={q})"
            ):
                z_new = rng.uniform(z_lo, z_hi, size=k_now)
                y_new = objective(_z_to_x(z_new, selected))
                all_z.append(z_new)
                all_y.append(y_new)
                best_values.append(max(all_y))
                continue

            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
            acq = qLogNoisyExpectedImprovement(
                model=gp, X_baseline=train_Z_norm, sampler=sampler,
            )
            cand, _ = optimize_acqf(
                acq_function=acq,
                bounds=torch.stack([torch.zeros(k_now), torch.ones(k_now)]).double(),
                q=1, num_restarts=10, raw_samples=512,
            )
            z_new = unnormalize(cand.squeeze(0), bounds).detach().numpy()
            y_new = objective(_z_to_x(z_new, selected))
            all_z.append(z_new)
            all_y.append(y_new)
            best_values.append(max(all_y))

        if q < n_rounds - 1:
            try:
                add_names = _llm_suggest_additions(
                    llm_client, category, df, target, mask,
                    nutrition_matrix=nutrition_matrix,
                    selected_names=selected_names,
                    selected_indices=list(selected),
                    all_z=list(all_z), all_y=list(all_y),
                    k_add=k_add,
                )
            except Exception as exc:
                warnings.warn(
                    f"adaptive_llm_bo(seed={seed}, round={q}): expert query "
                    f"failed ({type(exc).__name__}: {exc}); active set unchanged.",
                    RuntimeWarning, stacklevel=2,
                )
                add_names = []

            add_names = [n for n in add_names if n not in selected_names]
            if add_names:
                add_indices = resolve_selections(add_names, df)
                selected_names.extend(add_names)
                selected.extend(add_indices)
            print(
                f"  [adaptive_llm seed={seed}] round {q+1} added "
                f"{len(add_names)}: {add_names} -> |S|={len(selected_names)}",
                flush=True,
            )

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


# ----------------------------------------------------------------------------
# Per-category run
# ----------------------------------------------------------------------------

def run_single_category(
    domain: str,
    category: str,
    df: pd.DataFrame,
    nutrition_matrix: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    budget: int,
    n_seeds: int,
    n_init: int,
    methods: List[str],
    on_seed_complete: "Optional[callable]" = None,
    llm_client: "OpenRouterClient | None" = None,
    llm_rounds: int = 3,
    llm_k_init: int = 5,
    llm_k_add: int = 2,
    llm_information: str = "full",
) -> Dict[str, object]:
    d = len(df)
    objective = make_objective(nutrition_matrix, target, mask)

    if llm_information == "semantic-only" and "adaptive_llm" in methods:
        raise ValueError(
            "adaptive_llm is not yet defined for semantic-only feedback; "
            "use one-shot EGBO ('llm') or 'llm_optimizer'."
        )
    if llm_information == "semantic-only" and "llm" in methods:
        if llm_client is None:
            raise ValueError("semantic-only EGBO requires an OpenRouterClient")
        selection_names = _llm_select_initial(
            llm_client, category, df, target, mask, k=9,
            information_regime="semantic-only",
        )
    else:
        selection_names = LLM_SELECTIONS[domain][category]
    selected = resolve_selections(selection_names, df)
    approx_profile, approx_dist = best_subset_sample(nutrition_matrix, selected, target, mask)
    k = len(selected)

    _, oracle_full = nnls_simplex_oracle(nutrition_matrix, target, mask)
    _, oracle_llm = nnls_simplex_oracle(nutrition_matrix, target, mask, selected_indices=selected)

    method_registry = {
        "llm":           (
            (f"Semantic-only EGBO (k={k})" if llm_information == "semantic-only" else f"EGBO (k={k})"),
            lambda s: reduced_bo(objective, d, selected, budget=budget, n_init=n_init, seed=s),
        ),
        "vanilla":       (f"Vanilla BO (d={d})",        lambda s: vanilla_bo(objective, d, budget=budget, n_init=n_init, seed=s)),
        "random":        (f"Random Search (d={d})",     lambda s: random_search(objective, d, budget=budget, seed=s)),
        "turbo":         (f"TuRBO (d={d})",             lambda s: turbo_bo(objective, d, budget=budget, n_init=n_init, seed=s)),
        "saasbo":        (f"SAASBO (d={d})",            lambda s: saasbo_bo(objective, d, budget=budget, n_init=n_init, seed=s)),
        "sebo":          (f"SEBO (d={d})",              lambda s: sebo_bo(objective, d, budget=budget, n_init=n_init, seed=s)),
        "rembo":         (f"REMBO (d={d}->k={k})",      lambda s: rembo_bo(objective, d, budget=budget, n_init=n_init, seed=s, target_dim=k)),
        "random_subset": (f"Random Subset + BO (k={k})", lambda s: random_subset_bo(objective, d, k=k, budget=budget, n_init=n_init, seed=s)),
        "llm_optimizer": (
            ("Semantic-only iterative LLM" if llm_information == "semantic-only" else "Iterative LLM"),
            lambda s: iterative_llm_optimizer(
                objective, df, category, target, mask, nutrition_matrix,
                budget=budget, llm_client=llm_client,
                max_ingredients=9, temperature=0.0,
                information_regime=llm_information,
            ),
        ),
        "adaptive_llm":  (
            f"Adaptive EGBO (Q={llm_rounds})",
            lambda s: adaptive_llm_bo(
                objective, df, category, target, mask,
                budget=budget, n_init=n_init, seed=s,
                llm_client=llm_client,
                n_rounds=llm_rounds, k_init=llm_k_init, k_add=llm_k_add,
                # Use the cached LLM_SELECTIONS for round-0 when k_init matches
                # the cache size, so adaptive's first round is *identical* to
                # the static `llm` method's expert prior. Eliminates LLM-call
                # variance asymmetry between the two methods.
                initial_selection=(
                    selection_names if llm_k_init == len(selection_names) else None
                ),
            ),
        ),
    }
    active = [(key, method_registry[key]) for key in methods]

    method_names = [name for _, (name, _) in active]
    all_traces: Dict[str, List[List[float]]] = {name: [] for name in method_names}
    seed_records = []

    print(f"\n{'=' * 72}")
    print(f"  {category.upper()}")
    target_str = ", ".join(
        f"{NUTRIENT_LABELS[col]}={target[j]:.1f}" if mask[j]
        else f"{NUTRIENT_LABELS[col]}=NA"
        for j, col in enumerate(NUTRIENT_COLUMNS)
    )
    print(f"  target ({int(mask.sum())} measured dims): {target_str}")
    print(f"  selected: {[df['ingredient'].iloc[i] for i in selected]}")
    approx_str = ", ".join(
        f"{NUTRIENT_LABELS[col]}={approx_profile[j]:.1f}"
        for j, col in enumerate(NUTRIENT_COLUMNS)
    )
    print(f"  best sampled subset fit: z-rmsd={approx_dist:.3f} ({approx_str})")
    print(f"  oracle (NNLS, full d={d}): z-rmsd={oracle_full:.4f}")
    print(f"  oracle (NNLS, LLM k={k}):  z-rmsd={oracle_llm:.4f}")
    print(f"{'=' * 72}")

    def _build_result(n_completed: int) -> Dict[str, object]:
        """Snapshot of category results after `n_completed` seeds."""
        return {
            "traces": {name: list(all_traces[name]) for name in all_traces},
            "selected_indices": [int(i) for i in selected],
            "selected_ingredients": [str(df["ingredient"].iloc[i]) for i in selected],
            "target": {
                NUTRIENT_LABELS[col]: (float(target[j]) if mask[j] else None)
                for j, col in enumerate(NUTRIENT_COLUMNS)
            },
            "measured_dims": [
                NUTRIENT_LABELS[col] for j, col in enumerate(NUTRIENT_COLUMNS) if mask[j]
            ],
            "approx_profile": {
                NUTRIENT_LABELS[col]: float(approx_profile[j])
                for j, col in enumerate(NUTRIENT_COLUMNS)
            },
            "approx_distance": float(approx_dist),
            "oracle_full": float(oracle_full),
            "oracle_llm_subset": float(oracle_llm),
            "n_seeds_completed": int(n_completed),
            "n_seeds_planned": int(n_seeds),
            "seed_records": list(seed_records),
        }

    for seed in range(n_seeds):
        seed_start = time.time()
        seed_record = {"seed": int(seed), "methods": {}}
        per_method_summary = []

        for _, (name, run_fn) in active:
            m_start = time.time()
            res = run_fn(seed)
            m_elapsed = time.time() - m_start
            all_traces[name].append(res["best_values"])
            seed_record["methods"][name] = {
                "best_values": [float(v) for v in res["best_values"]],
                "all_y": [float(v) for v in res["all_y"]],
                "final_best": float(res["final_best"]),
            }
            if "recipe_history" in res:
                seed_record["methods"][name]["recipe_history"] = res["recipe_history"]
            z_rmsd = -float(res["final_best"])
            per_method_summary.append(f"{name}={z_rmsd:.4f}({_fmt_elapsed(m_elapsed)})")
            print(
                f"  [{category}] seed {seed + 1}/{n_seeds} {name}: "
                f"z-rmsd={z_rmsd:.4f} ({_fmt_elapsed(m_elapsed)})",
                flush=True,
            )

        seed_records.append(seed_record)
        seed_elapsed = time.time() - seed_start
        print(
            f"  [{category}] seed {seed + 1}/{n_seeds} done in "
            f"{_fmt_elapsed(seed_elapsed)} | "
            + " ".join(per_method_summary),
            flush=True,
        )

        if on_seed_complete is not None:
            try:
                on_seed_complete(_build_result(seed + 1))
            except Exception as exc:
                warnings.warn(
                    f"[{category}] checkpoint after seed {seed + 1} failed: "
                    f"{type(exc).__name__}: {exc}",
                    RuntimeWarning, stacklevel=2,
                )

    return _build_result(n_seeds)


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

_METHOD_STYLE = {
    "EGBO": ("#2ecc71", "-", 2.5),
    "Iterative LLM": ("#c039d3", "--", 2.2),
    "Vanilla": ("#e74c3c", "-.", 2.0),
    "Random Subset": ("#d4ac0d", "--", 2.0),
    "Random": ("#999999", "--", 1.5),
    "TuRBO": ("#9b59b6", "-.", 2.0),
    "SAASBO": ("#e67e22", ":", 2.0),
    "SEBO": ("#16a085", ":", 2.0),
    "REMBO": ("#3498db", "--", 2.0),
}


def _style(name: str):
    for key, val in _METHOD_STYLE.items():
        if key in name:
            return val
    return ("#333333", "-", 1.5)


def _safe_filename(s: str) -> str:
    return s.replace(" ", "_").replace("&", "and").replace("(", "").replace(")", "")


def plot_category_convergence(
    category: str, all_traces: Dict[str, list], budget: int, save_dir: str,
    oracle_full: float | None = None, oracle_llm: float | None = None,
):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    iters = np.arange(1, budget + 1)
    for name, traces in all_traces.items():
        arr = np.array(traces)
        mean_dist = -arr.mean(axis=0)
        std_dist = arr.std(axis=0)
        color, ls, lw = _style(name)
        ax.plot(iters, mean_dist, label=name, color=color, linestyle=ls, linewidth=lw)
        ax.fill_between(iters, mean_dist - std_dist, mean_dist + std_dist, alpha=0.15, color=color)
    if oracle_llm is not None:
        ax.axhline(oracle_llm, color="#2c3e50", linestyle=(0, (1, 1)), linewidth=1.5,
                   label=f"Oracle (EGBO k=9) = {oracle_llm:.3f}")
    if oracle_full is not None:
        ax.axhline(oracle_full, color="#000000", linestyle=(0, (3, 3)), linewidth=1.5,
                   label=f"Oracle (full d) = {oracle_full:.3f}")
    ax.set_xlabel("Evaluation", fontsize=12)
    ax.set_ylabel("Z-Score Nutrition RMSD (lower = better)", fontsize=12)
    ax.set_title(f"Plant-Based {category.title()}: Multi-Nutrient Matching", fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, budget)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{_safe_filename(category)}_convergence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def plot_summary_heatmap(all_results: Dict[str, Dict[str, list]], save_dir: str, domain: str):
    categories = list(all_results.keys())
    method_names = list(all_results[categories[0]].keys())
    mat = np.zeros((len(categories), len(method_names)))
    for i, cat in enumerate(categories):
        for j, meth in enumerate(method_names):
            finals = [-t[-1] for t in all_results[cat][meth]]
            mat[i, j] = np.mean(finals)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels([m.replace(" ", "\n") for m in method_names], fontsize=9)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([c.title() for c in categories], fontsize=10)
    for i in range(len(categories)):
        for j in range(len(method_names)):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > mat.mean() else "black")
    ax.set_title("Final Z-Score Nutrition RMSD by Category & Method", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{domain}_summary_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------

def _atomic_write_json(path: str, obj) -> None:
    """Write JSON to `path` atomically.

    Networked / synced filesystems can race between `mkdir` and a
    subsequent `open()`: the directory exists in the in-process view but
    isn't yet visible to syscalls. This helper re-creates the parent on
    every call and writes via a tmp file + rename so partial writes never
    linger.
    """
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def run_all(
    domain: str,
    categories: List[str] | None,
    budget: int,
    n_seeds: int,
    n_init: int,
    methods: List[str],
    plot: bool,
    name: str | None,
    output_root: str,
    data_dir: str = _DEFAULT_DATA_DIR,
    llm_client: "OpenRouterClient | None" = None,
    llm_rounds: int = 3,
    llm_k_init: int = 5,
    llm_k_add: int = 2,
    llm_information: str = "full",
    exclude_ingredients: "List[str] | None" = None,
):
    df = load_ingredients(domain, data_dir=data_dir)
    excluded = {
        str(name).strip().casefold() for name in (exclude_ingredients or [])
    }
    if excluded:
        df = df[
            ~df["ingredient"].astype(str).str.strip().str.casefold().isin(excluded)
        ].reset_index(drop=True)
    nutrition_matrix = get_nutrition_matrix(df)
    targets = load_targets(domain, data_dir=data_dir)
    d = len(df)

    available = list(LLM_SELECTIONS[domain].keys())
    if categories is None:
        categories = available
    else:
        unknown = [c for c in categories if c not in available]
        if unknown:
            raise ValueError(f"Unknown {domain} categories: {unknown}. Available: {available}")
    missing_targets = [c for c in categories if c not in targets]
    if missing_targets:
        raise ValueError(
            f"Categories missing from animal_products_nutrition.csv: {missing_targets}"
        )

    print(f"Domain: {domain}, Ingredients: d={d}, Budget: {budget}, Seeds: {n_seeds}, n_init: {n_init}")
    print(f"Nutrients: {[NUTRIENT_LABELS[c] for c in NUTRIENT_COLUMNS]}")
    print(f"Categories ({len(categories)}): {categories}")
    print(f"Methods: {methods}\n")

    run_id = name if name else datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, run_id, domain)
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    summary_path = os.path.join(run_dir, "summary.json")
    raw_path = os.path.join(run_dir, "raw.json")
    print(f"Run folder: {run_dir}")

    all_results: Dict[str, Dict[str, list]] = {}
    raw_results: Dict[str, Dict[str, object]] = {}
    raw_payload = {
        "config": {
            "domain": domain,
            "budget": int(budget),
            "n_seeds": int(n_seeds),
            "n_init": int(n_init),
            "categories": list(categories),
            "methods": methods,
            "n_ingredients": int(d),
            "ingredient_names": [str(x) for x in df["ingredient"].tolist()],
            "nutrients": [NUTRIENT_LABELS[c] for c in NUTRIENT_COLUMNS],
            "normalization": "z-score over domain ingredient ground set",
            "loss": "per-dim RMSD over measured nutrients (NaN dims masked out)",
            "llm_information": llm_information,
            "excluded_ingredients": list(exclude_ingredients or []),
        },
        "results": raw_results,
    }

    # Resume named runs from their category checkpoints. Completed categories
    # are preserved; an interrupted category is rerun from its first seed.
    # Per-category EGBO runners checkpoint individual seeds separately, but
    # this aggregate baseline runner stores all categories in one raw.json.
    if name and os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8") as handle:
            previous_payload = json.load(handle)
        previous_config = previous_payload.get("config", {})
        expected_config = raw_payload["config"]
        resume_fields = (
            "domain", "budget", "n_seeds", "n_init", "methods",
            "n_ingredients", "excluded_ingredients",
        )
        mismatches = [
            field for field in resume_fields
            if previous_config.get(field) != expected_config.get(field)
        ]
        if mismatches:
            raise ValueError(
                f"Refusing to resume {raw_path}: configuration mismatch in "
                f"{mismatches}"
            )
        raw_results.update(previous_payload.get("results", {}))
        print(
            f"Resuming named run from {raw_path} "
            f"({len(raw_results)} category checkpoints found)"
        )

    for cat in categories:
        existing = raw_results.get(cat)
        if existing is not None:
            completed = int(existing.get("n_seeds_completed", 0))
            if completed >= n_seeds:
                all_results[cat] = existing["traces"]
                print(f"[{cat}] already complete ({completed}/{n_seeds}); skipping")
                continue
            print(
                f"[{cat}] incomplete checkpoint ({completed}/{n_seeds}); "
                "rerunning this category"
            )
        target_vec, mask = targets[cat]

        def _checkpoint_partial(partial_result, cat=cat):
            """Write raw.json after every seed within this category."""
            raw_results[cat] = partial_result
            _atomic_write_json(raw_path, raw_payload)

        category_result = run_single_category(
            domain=domain,
            category=cat,
            df=df,
            nutrition_matrix=nutrition_matrix,
            target=target_vec,
            mask=mask,
            budget=budget,
            n_seeds=n_seeds,
            n_init=n_init,
            methods=methods,
            on_seed_complete=_checkpoint_partial,
            llm_client=llm_client,
            llm_rounds=llm_rounds,
            llm_k_init=llm_k_init,
            llm_k_add=llm_k_add,
            llm_information=llm_information,
        )
        traces = category_result["traces"]
        all_results[cat] = traces
        raw_results[cat] = category_result

        summary_payload = {
            done_cat: {
                meth: [[float(v) for v in t] for t in tlist]
                for meth, tlist in done_traces.items()
            }
            for done_cat, done_traces in all_results.items()
        }
        _atomic_write_json(summary_path, summary_payload)
        _atomic_write_json(raw_path, raw_payload)
        print(f"  checkpointed -> {summary_path}, {raw_path}")

        if plot:
            plot_category_convergence(
                cat, traces, budget, plot_dir,
                oracle_full=category_result["oracle_full"],
                oracle_llm=category_result["oracle_llm_subset"],
            )

    print("\n" + "=" * 92)
    print("RESULTS SUMMARY (final z-score RMSD, lower is better)")
    print("=" * 92)
    method_names = list(all_results[categories[0]].keys())
    header = f"{'Category':<36}" + "".join(f"{m:<26}" for m in method_names)
    print(header)
    print("-" * len(header))
    for cat in categories:
        row = f"{cat:<36}"
        for meth in method_names:
            finals = [-t[-1] for t in all_results[cat][meth]]
            row += f"{np.mean(finals):.4f} +/- {np.std(finals):.4f}  "
        print(row)

    if plot and len(categories) > 1:
        plot_summary_heatmap(all_results, plot_dir, domain=domain)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="FormulateBench experiment runner")
    parser.add_argument("--domain", type=str, required=True, choices=list(DOMAIN_CONFIG.keys()))
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Run only selected categories (default: all categories in the domain)",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None, choices=ALL_METHOD_KEYS,
        help=f"Methods to run (default: all). Choices: {', '.join(ALL_METHOD_KEYS)}",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Run name. Outputs go to <output-root>/<name>/<domain>/. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--output-root", type=str, default=os.path.join(_HERE, "results"),
        help="Root directory for run outputs (default: results/ next to this script)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=_DEFAULT_DATA_DIR,
        help="Directory containing usda_nutrition_data.csv and animal_products_nutrition.csv",
    )
    parser.add_argument(
        "--exclude-ingredients", nargs="*", default=None,
        help="Ingredient names to remove from the candidate pool before all methods run.",
    )
    parser.add_argument(
        "--openrouter-key", type=str, default=None,
        help="OpenRouter API key for live LLM methods. Falls back to "
             "$OPENROUTER_API_KEY. Required only for adaptive_llm or llm_optimizer.",
    )
    parser.add_argument(
        "--llm-model", type=str, default="anthropic/claude-opus-4.7",
        help="OpenRouter model id used by live LLM methods (default: %(default)s).",
    )
    parser.add_argument(
        "--llm-rounds", type=int, default=3,
        help="Number of expert rounds Q for adaptive_llm (default: %(default)d).",
    )
    parser.add_argument(
        "--llm-k-init", type=int, default=5,
        help="Initial active-set size for adaptive_llm (default: %(default)d).",
    )
    parser.add_argument(
        "--llm-k-add", type=int, default=2,
        help="Max additions per round for adaptive_llm (default: %(default)d).",
    )
    parser.add_argument(
        "--llm-cache-dir", type=str, default=None,
        help="Directory for SHA-256-keyed JSON cache of OpenRouter responses. "
             "If set, cache hits are bit-identical and free; cache misses are "
             "issued live and stored. Recommended for reproducibility.",
    )
    parser.add_argument(
        "--llm-information", type=str, default="full",
        choices=["full", "semantic-only"],
        help="Information exposed to live LLM methods. 'full' gives the "
             "nutrition table and numerical target. 'semantic-only' gives "
             "only the target product category, ingredient names, and scalar "
             "feedback; the evaluated objective is still nutrition matching.",
    )
    parser.add_argument(
        "--track", type=str, default="nutrition", choices=["nutrition", "sensory"],
        help="Benchmark track. 'nutrition' (default) = z-score nutrient matching "
             "(§4.2). 'sensory' = Gemini-as-sensory-surrogate animal-similarity "
             "(§4.3); requires a pre-evaluated pool generated by sensory_bench.py.",
    )
    parser.add_argument(
        "--gemini-model", type=str, default="google/gemini-3.1-pro-preview",
        help="OpenRouter model id for the sensory surrogate (default: %(default)s). "
             "Used only when --track sensory.",
    )
    parser.add_argument(
        "--sensory-pool-dir", type=str,
        default=os.path.join(_HERE, "data", "sensory_pools"),
        help="Directory holding pre-evaluated sensory pools "
             "(<category>.npz). Used only when --track sensory.",
    )
    args = parser.parse_args()

    methods = args.methods or _DEFAULT_METHOD_KEYS
    llm_client = None
    needs_live_llm = any(method in _LIVE_LLM_METHOD_KEYS for method in methods)
    if args.llm_information == "semantic-only" and "llm" in methods:
        needs_live_llm = True
    if needs_live_llm:
        key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit(
                "Live LLM methods require --openrouter-key or "
                "$OPENROUTER_API_KEY to be set."
            )
        llm_client = OpenRouterClient(
            api_key=key, model=args.llm_model, cache_dir=args.llm_cache_dir,
        )

    if args.track == "sensory":
        if args.llm_information != "full":
            raise SystemExit(
                "--llm-information semantic-only is currently defined only "
                "for the nutrition track."
            )
        # Lazy import to avoid pulling sensory_bench into nutrition-only runs.
        from sensory_bench import run_all_sensory
        run_all_sensory(
            domain=args.domain,
            categories=args.categories,
            budget=args.budget,
            n_seeds=args.seeds,
            n_init=args.n_init,
            methods=methods,
            plot=not args.no_plot,
            name=args.name,
            output_root=args.output_root,
            data_dir=args.data_dir,
            llm_client=llm_client,
            llm_rounds=args.llm_rounds,
            llm_k_init=args.llm_k_init,
            llm_k_add=args.llm_k_add,
            pool_dir=args.sensory_pool_dir,
        )
        return

    run_all(
        domain=args.domain,
        categories=args.categories,
        budget=args.budget,
        n_seeds=args.seeds,
        n_init=args.n_init,
        methods=methods,
        plot=not args.no_plot,
        name=args.name,
        output_root=args.output_root,
        data_dir=args.data_dir,
        llm_client=llm_client,
        llm_rounds=args.llm_rounds,
        llm_k_init=args.llm_k_init,
        llm_k_add=args.llm_k_add,
        llm_information=args.llm_information,
        exclude_ingredients=args.exclude_ingredients,
    )


if __name__ == "__main__":
    main()
