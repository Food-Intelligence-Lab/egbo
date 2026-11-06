"""Rigorous tests for the adaptivity + expert-config changes to food_bo.py.

Run in an environment with torch/botorch installed (the deployment env):

    pip install pytest
    pytest test_food_bo.py -v

Layers:
  * Logic         — encoding/re-encoding, guard, fork, config validation (fast)
  * Integration   — the live GP fit through ask() (real torch/botorch)
  * Behavioral    — does it actually optimize (BO vs random on a known objective)

To skip the slow config sweep:  pytest test_food_bo.py -v -k "not all_configs"
"""

import math
import os

import numpy as np
import pytest

from food_bo import FoodOptimizer, validate_bo_config


def _make(name, ingredients=("a", "b", "c"), lo=0, hi=10):
    o = FoodOptimizer(name)
    for n in ingredients:
        o.add_ingredient(n, lo, hi)
    o.add_objective("score", weight=1.0, goal="max", min_val=0, max_val=10)
    return o


def _seed_history(o, n=6, seed=0):
    rng = np.random.RandomState(seed)
    for _ in range(n):
        rec = {v["name"]: float(rng.rand() * 10) for v in o.variables}
        o.tell(rec, {"score": float(rng.rand() * 10)})


@pytest.fixture
def proj(tmp_path):
    os.chdir(tmp_path)
    return _make("t")


# --------------------------------------------------------------------------- #
#  Layer 1: Logic (fast, torch not exercised)
# --------------------------------------------------------------------------- #

def test_validate_config_coerces_and_empties():
    assert validate_bo_config(None) is None
    assert validate_bo_config({}) is None
    c = validate_bo_config(
        {"kernel": "junk", "noise": "fixed_tiny", "acquisition": "qucb",
         "lengthscale_prior": "long"}
    )
    assert c["kernel"] == "matern52"          # junk -> default
    assert c["noise"] == "fixed_tiny"          # valid preserved
    assert c["acquisition"] == "qucb"
    assert c["lengthscale_prior"] == "long"


def test_encode_tolerates_missing_keys(proj):
    proj.tell({"a": 2.0}, {"score": 5})        # b, c absent -> 0
    assert proj.X_history[-1] == [2.0, 0.0, 0.0]


def test_adaptive_add_reencodes_and_preserves_columns(proj):
    _seed_history(proj, 6)
    pre0 = list(proj.X_history[0])
    proj.add_ingredient("d", 5, 12)            # min deliberately >0 to test forcing
    assert all(len(x) == 4 for x in proj.X_history)
    assert all(x[3] == 0.0 for x in proj.X_history)      # absent in every past recipe
    assert proj.variables[-1]["bounds"] == (0.0, 12.0)   # min forced to 0
    assert proj.X_history[0][:3] == pre0                 # existing columns untouched


def test_process_param_midrun_encodes_at_baseline(proj):
    _seed_history(proj, 6)
    # Prior batches all "ran at" cook_temp = 80; encode past there, not 0.
    proj.add_process_parameter("cook_temp", 60, 100, baseline=80.0)
    assert all(len(x) == 4 for x in proj.X_history)
    assert all(x[3] == 80.0 for x in proj.X_history)         # baseline, not 0
    assert proj.variables[-1]["bounds"] == (60.0, 100.0)     # min NOT forced to 0
    assert proj.variables[-1]["_absent_value"] == 80.0


def test_process_param_midrun_requires_valid_baseline(proj):
    _seed_history(proj, 6)
    with pytest.raises(ValueError):                           # missing baseline
        proj.add_process_parameter("cook_temp", 60, 100)
    with pytest.raises(ValueError):                           # baseline out of [min,max]
        proj.add_process_parameter("cook_temp", 60, 100, baseline=200.0)


def test_add_guard_raises_on_incomplete_history(proj):
    _seed_history(proj, 3)
    proj.recipe_history.pop()                   # simulate a row without stored recipe
    with pytest.raises(ValueError):
        proj.add_ingredient("d", 0, 1)


def test_fork_is_independent(proj):
    _seed_history(proj, 6)
    n = len(proj.X_history)
    f = proj.fork("t_fork")
    f.tell({"a": 1, "b": 1, "c": 1}, {"score": 9})
    assert len(f.X_history) == n + 1
    assert len(proj.X_history) == n            # original unchanged


def test_config_survives_json_roundtrip(proj):
    proj.set_bo_config(
        {"kernel": "rbf", "noise": "fixed_tiny",
         "acquisition": "qlogei", "lengthscale_prior": "long"}
    )
    state = proj.export_json()
    assert state["bo_config"]["kernel"] == "rbf"
    proj.import_json(state)
    assert proj.bo_config["kernel"] == "rbf"
    assert proj.bo_config["noise"] == "fixed_tiny"


def test_export_trajectory_marks_best(proj):
    _seed_history(proj, 4)
    t = proj.export_trajectory()
    assert "iter" in t and "utility" in t and "*" in t


# --------------------------------------------------------------------------- #
#  Layer 2: Integration — the live GP fit through ask()
# --------------------------------------------------------------------------- #

def _assert_in_bounds(recipe, variables):
    for v in variables:
        lo, hi = v["bounds"]
        assert lo - 1e-6 <= recipe[v["name"]] <= hi + 1e-6, (v["name"], recipe[v["name"]])


def test_ask_default_runs_and_is_in_bounds(proj):
    _seed_history(proj, 6)
    recs = proj.ask(n_suggestions=3)
    assert len(recs) == 3
    for r in recs:
        _assert_in_bounds(r, proj.variables)


def test_ask_after_adaptive_add(proj):
    """The core risk: does BO fit and propose correctly after the dimension grows?"""
    _seed_history(proj, 6)
    proj.add_ingredient("d", 0, 8)
    recs = proj.ask(n_suggestions=2)
    assert len(recs) == 2
    for r in recs:
        assert "d" in r
        _assert_in_bounds(r, proj.variables)


def test_ask_after_adaptive_process_param(proj):
    """Same, but adding a process parameter mid-run (baseline-encoded history)."""
    _seed_history(proj, 6)
    proj.add_process_parameter("cook_temp", 60, 100, baseline=80.0)
    recs = proj.ask(n_suggestions=2)
    assert len(recs) == 2
    for r in recs:
        assert "cook_temp" in r
        _assert_in_bounds(r, proj.variables)


def test_ask_is_deterministic_given_history(proj):
    """ask() seeds torch by len(history); two calls without telling must match.
    This is the regression guard for the default path."""
    _seed_history(proj, 6, seed=42)
    r1 = proj.ask(n_suggestions=1)[0]
    r2 = proj.ask(n_suggestions=1)[0]
    assert all(abs(r1[k] - r2[k]) < 1e-8 for k in r1)


@pytest.mark.parametrize("kernel", ["matern52", "matern32", "rbf", "linear", "poly2"])
@pytest.mark.parametrize("noise", ["default", "low", "fixed_tiny"])
@pytest.mark.parametrize("acq", ["qlognei", "qlogei", "qucb"])
def test_ask_all_configs_run(proj, kernel, noise, acq):
    """Every expert-config combination must fit and propose without error —
    these hit less-traveled botorch paths (linear/poly2 kernels, fixed noise,
    qLogEI best_f, qUCB)."""
    _seed_history(proj, 6)
    proj.set_bo_config(
        {"kernel": kernel, "lengthscale_prior": "default",
         "noise": noise, "acquisition": acq}
    )
    recs = proj.ask(n_suggestions=1)
    assert len(recs) == 1
    _assert_in_bounds(recs[0], proj.variables)


# --------------------------------------------------------------------------- #
#  Layer 3: Behavioral — does it actually optimize?
# --------------------------------------------------------------------------- #

def test_bo_beats_random_on_known_objective(tmp_path):
    """Fair paired comparison: guided BO vs random search from an identical
    deterministic cold start, on a smooth objective with a known optimum.
    Guided should reach a better best-so-far. Tolerance guards mild stochasticity.
    """
    os.chdir(tmp_path)
    target = {"a": 8.0, "b": 2.0, "c": 5.0}

    def true_score(rec):                       # max 10 at target, smooth
        dist = sum((rec[k] - target[k]) ** 2 for k in target)
        return 10.0 * math.exp(-dist / 50.0)

    n_init, n_total = 5, 25

    guided = _make("guided")
    for _ in range(n_total):
        r = guided.ask(n_suggestions=1, n_init_random=n_init)[0]
        guided.tell(r, {"score": true_score(r)})

    rnd = _make("rand")
    rng = np.random.RandomState(123)
    for i in range(n_total):
        if i < n_init:                         # identical deterministic Sobol cold start
            r = rnd.ask(n_suggestions=1, n_init_random=n_init)[0]
        else:
            r = {v["name"]: float(rng.rand() * 10) for v in rnd.variables}
        rnd.tell(r, {"score": true_score(r)})

    guided_best = max(guided.Y_history[n_init:])
    random_best = max(rnd.Y_history[n_init:])
    print(f"\nguided_best={guided_best:.3f}  random_best={random_best:.3f}")
    assert guided_best >= random_best - 0.5    # BO at least as good as random
    assert guided_best > 0.5                   # found a genuinely good region

