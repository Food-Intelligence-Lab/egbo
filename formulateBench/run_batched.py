#!/usr/bin/env python3
"""Batched FormulateBench: R rounds x 3 recipes (default 7; --rounds 10 matches the deployment), optional noise.

Mirrors a laboratory that prepares and evaluates three formulations per session.
For a given LLM expert (its shared first recipe and ingredient set, as in the
paper's protocol):

  Round 1 (all arms, identical for a given seed): the shared first recipe plus
    two seeded logit-normal perturbations of it (sigma 0.35, the paper's local
    initialization).
  Rounds 2-R:
    llm_unres  - the LLM proposes three formulations from the full ingredient
                 pool (at most 9 ingredients each) given all past results.
    llm_fixed  - the same, restricted to the shared ingredient set.
    egbo       - BO on the shared ingredient set with the paper's
                 LLM-configured GP and qLogNEI, optimizing a batch of q=3
                 (sequential greedy), as in the deployment application.

Conventional baselines (no LLM; full ingredient catalog, logit parameterization
as in the paper's vanilla BO), --arms random vanilla_bo turbo:
    random      - three Dirichlet(1) recipes per round.
    vanilla_bo  - round 1: three simplex-initial points; then a GP (BoTorch
                  defaults, learned noise) with qLogNEI, batch q=3.
    turbo       - as vanilla_bo, inside a TuRBO trust region (success
                  tolerance 3, failure tolerance max(4, ceil(d/q))).
  Baseline results are stored under results/<out>/baselines/.

Observation noise (optional): y_obs = y + N(0, (level * norm)^2), as in
run_noise_sweep.py. Every evaluation logs its true and observed loss; methods
are scored by the true loss of the recommended recipe (best observed so far).
--noise-aware tells the LLM the noise SD (off by default).
Results: results/batched_v1/<model_slug>/<domain>/<category>/<arm>_noise<level>_seed<k>.json
Requires OPENROUTER_API_KEY for LLM arms (responses cached under results/batched_v1/llm_cache).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
R = HERE / "results"
ROUNDS, Q = 7, 3
SYSTEM = "You are an expert plant-based food scientist."


def _round1(x0, selected, seed, n_extra=2, sigma=0.35):
    rng = np.random.RandomState(seed)
    xs = [x0.copy()]
    z0 = np.log(np.clip(x0[selected], np.exp(-12.0), 1.0))
    for _ in range(n_extra):
        z = np.clip(z0 + rng.normal(0.0, sigma, size=len(selected)), -12.0, 0.0)
        f = np.exp(z - z.max()); f /= f.sum()
        x = np.zeros_like(x0); x[selected] = f
        xs.append(x)
    return xs


def _llm_batch(client, fb, df, category, history, max_ing, note, q=Q):
    hist = [{"iteration": h["iteration"], "recipe": h["recipe"], "target_dissimilarity": h["loss"]}
            for h in history]
    feedback = ("Previous evaluated formulations follow. Target dissimilarity is the evaluator's scalar "
                "output; lower is better. Use only these observations to propose the next formulations.\n"
                + json.dumps(hist, separators=(",", ":")))
    if note:
        feedback += "\n\n" + note
    prompt = f"""You are optimizing the overall similarity of a plant-based formulation to animal-based {category}. The evaluator is a black box.

Available plant-based ingredients:
{fb._ingredient_names_block(df)}

{feedback}

The laboratory prepares and evaluates {q} formulations in parallel per session. Propose exactly {q} complete formulations for the next session. Each uses at most {max_ing} available ingredients, nonnegative quantities, and no ingredient outside the catalog. Return only JSON in this exact shape:
{{"recipes":[{{"recipe":[{{"ingredient":"exact catalog name","fraction":0.25}}]}}]}}
The list must contain exactly {q} entries. Fractions in each recipe must sum to 1. Do not include commentary."""
    raw = client.chat(SYSTEM, prompt, temperature=0.0)
    for attempt in range(3):
        try:
            cleaned = re.sub(r"```(?:json)?\s*", "", raw or "").replace("```", "")
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            data = json.loads(m.group(0))["recipes"]
            if not isinstance(data, list) or len(data) < q:
                raise ValueError(f"expected {q} recipes, got {len(data) if isinstance(data, list) else 'none'}")
            out = [fb._parse_recipe(json.dumps({"recipe": r["recipe"] if isinstance(r, dict) else r}), df,
                                    max_ingredients=max_ing) for r in data[:q]]
            return out
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                raise
            raw = client.chat(SYSTEM, f"""Your response was invalid: {exc}.

Invalid response:
{raw}

Correct it without changing the optimization goal. Return exactly {q} formulations, each with at most {max_ing} catalog ingredients and nonnegative fractions summing to 1, only as:
{{"recipes":[{{"recipe":[{{"ingredient":"exact catalog name","fraction":0.25}}]}}]}}""", temperature=0.0)


BASELINES = ("random", "vanilla_bo", "turbo")


def _baseline_batches(fb, obj, d, seed, rounds, kind):
    """Batched conventional baselines over the full catalog (q = Q per round)."""
    import math
    import torch
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.utils.transforms import normalize, unnormalize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    batches = []
    if kind == "random":
        for r in range(rounds):
            idx = []
            for _ in range(Q):
                obj(rng.dirichlet(np.ones(d))); idx.append(len(obj.true) - 1)
            batches.append(idx)
        return batches
    bounds = torch.tensor([[-12.0] * d, [0.0] * d], dtype=torch.double)
    Z = list(fb._simplex_initial_logits(d, Q, seed))
    for z in Z:
        obj(fb._softmax(z))
    batches.append([0, 1, 2])
    length, n_succ, n_fail = 0.8, 0, 0
    fail_tol = max(4, math.ceil(d / Q))
    best = min(obj.obs)
    for r in range(1, rounds):
        Zt = normalize(torch.tensor(np.array(Z), dtype=torch.double), bounds)
        Yt = -torch.tensor(obj.obs, dtype=torch.double).unsqueeze(-1)
        gp = SingleTaskGP(Zt, Yt, outcome_transform=Standardize(m=1))
        ok = fb._fit_gpytorch_mll_with_warning(ExactMarginalLogLikelihood(gp.likelihood, gp), f"batched {kind}")
        if ok:
            lb, ub = torch.zeros(d, dtype=torch.double), torch.ones(d, dtype=torch.double)
            if kind == "turbo":
                center = Zt[int(np.argmin(obj.obs))]
                ls = gp.covar_module.lengthscale.detach().squeeze().clamp_min(1e-12)
                w = ls.mean() / ls
                w = w / torch.prod(w).pow(1.0 / d)
                lb = torch.clamp(center - w * length / 2, 0.0, 1.0)
                ub = torch.clamp(center + w * length / 2, 0.0, 1.0)
            acq = qLogNoisyExpectedImprovement(model=gp, X_baseline=Zt,
                                               sampler=SobolQMCNormalSampler(sample_shape=torch.Size([256])))
            cand, _ = optimize_acqf(acq, bounds=torch.stack([lb, ub]), q=Q, num_restarts=10,
                                    raw_samples=512, sequential=True)
            new_z = unnormalize(cand, bounds).detach().numpy()
        else:
            new_z = rng.uniform(-12.0, 0.0, size=(Q, d))
        idx = []
        for z in new_z:
            obj(fb._softmax(z)); Z.append(z); idx.append(len(obj.true) - 1)
        batches.append(idx)
        if kind == "turbo":
            nb = min(obj.obs[i] for i in idx)
            if nb < best:
                best, n_succ, n_fail = nb, n_succ + 1, 0
            else:
                n_succ, n_fail = 0, n_fail + 1
            if n_succ >= 3:
                length, n_succ = min(2.0 * length, 1.6), 0
            elif n_fail >= fail_tol:
                length, n_fail = length / 2.0, 0
            if length < 0.5 ** 7:
                length, n_succ, n_fail = 0.8, 0, 0
    return batches


def run_task(task):
    model, recipe_root, domain, category, level, arm, seed, out_path, cache_dir, noise_aware, rounds = task
    out_path = Path(out_path)
    if out_path.exists():
        return str(out_path), "skip"
    try:
        import torch
        torch.set_num_threads(1)
        from run_noise_sweep import Logged, _setup
        fb, df, M, target, mask, objective, norm, x0, selected, recipe0 = _setup(domain, category, recipe_root)
        sigma = level * norm
        arm_id = {"egbo": 0, "llm_unres": 1, "llm_fixed": 2, "random": 3, "vanilla_bo": 4, "turbo": 5}[arm]
        noise_seed = int(7_000_003 * arm_id + 10_007 * seed + int(level * 1000) + 17)
        names = [str(n) for n in df["ingredient"]]
        obj = Logged(objective, sigma, noise_seed, names)
        history, batches = [], []
        if arm in BASELINES:
            batches = _baseline_batches(fb, obj, len(df), seed, rounds, arm)
        else:
            xs1 = _round1(x0, selected, seed)
            for x in xs1:
                obj(x)
            batches.append([0, 1, 2])
        if arm in BASELINES:
            pass
        elif arm == "egbo":
            import egbo_bo_config as configured
            from botorch.optim import optimize_acqf
            from botorch.utils.transforms import normalize, unnormalize
            k = len(selected)
            bounds = torch.tensor([[-12.0] * k, [0.0] * k], dtype=torch.double)
            Z = [np.clip(np.log(np.clip(x[selected], np.exp(-12.0), 1.0)), -12.0, 0.0) for x in xs1]
            rng = np.random.RandomState(seed)
            torch.manual_seed(seed)
            cfg = configured.BOConfig()
            for r in range(1, rounds):
                Zt = normalize(torch.tensor(np.array(Z), dtype=torch.double), bounds)
                Yt = -torch.tensor(obj.obs, dtype=torch.double).unsqueeze(-1)  # maximize -loss
                try:
                    gp, mll = configured.build_gp(Zt, Yt, cfg)
                    ok = fb._fit_gpytorch_mll_with_warning(mll, f"batched egbo {category} seed {seed}")
                except Exception:  # noqa: BLE001
                    ok = False
                if ok:
                    acq = configured.build_acqf(gp, Zt, cfg)
                    cand, _ = optimize_acqf(acq, bounds=torch.stack([torch.zeros(k), torch.ones(k)]).double(),
                                            q=Q, num_restarts=10, raw_samples=512, sequential=True)
                    new_z = unnormalize(cand, bounds).detach().numpy()
                else:
                    new_z = rng.uniform(-12.0, 0.0, size=(Q, k))
                idx = []
                for z in new_z:
                    f = np.exp(z - z.max()); f /= f.sum()
                    x = np.zeros(len(df)); x[selected] = f
                    obj(x); Z.append(z); idx.append(len(obj.true) - 1)
                batches.append(idx)
        else:
            note = (f"Note: the evaluator's scores include random measurement noise. Each reported target "
                    f"dissimilarity equals the formulation's true dissimilarity plus Gaussian noise with standard "
                    f"deviation {sigma:.4f}, so repeated evaluations of the same formulation can differ."
                    ) if (noise_aware and sigma > 0) else None
            client = fb.OpenRouterClient(os.environ["OPENROUTER_API_KEY"], model, cache_dir=cache_dir)
            if arm == "llm_unres":
                sub, sub_sel, max_ing = df, list(range(len(df))), 9
            else:
                sub, sub_sel, max_ing = df.iloc[selected].reset_index(drop=True), selected, len(selected)
            for i in range(3):
                history.append({"iteration": i + 1, "recipe": [{"ingredient": n, "fraction": f}
                                for n, f in obj.recipes[i].items()], "loss": obj.obs[i]})
            for r in range(1, rounds):
                proposals = _llm_batch(client, fb, sub, category, history, max_ing, note)
                idx = []
                for xs_sub, recipe in proposals:
                    x = np.zeros(len(df)); x[sub_sel] = xs_sub
                    obj(x); idx.append(len(obj.true) - 1)
                    history.append({"iteration": len(obj.true), "recipe": recipe, "loss": obj.obs[-1]})
                batches.append(idx)
        assert len(obj.true) == rounds * Q, len(obj.true)
        rec_true, best_i = [], 0
        for t in range(len(obj.obs)):
            if obj.obs[t] < obj.obs[best_i]:
                best_i = t
            rec_true.append(obj.true[best_i])
        out = dict(model=model, domain=domain, category=category, level=level, sigma=sigma, norm=norm,
                   arm=arm, seed=seed, noise_seed=noise_seed, noise_aware=bool(noise_aware), rounds=rounds, q=Q,
                   batches=batches, true_loss=obj.true, observed_loss=obj.obs,
                   recommended_true_loss=rec_true, best_true_so_far=list(np.minimum.accumulate(obj.true)),
                   recipes=obj.recipes)
        from _run_utils import write_once
        if not write_once(out_path, json.dumps(out)):
            return str(out_path), "skip (result already written by another run)"
        return str(out_path), "ok"
    except Exception:
        return str(out_path), "FAIL\n" + traceback.format_exc()


MODELS = {  # model id -> recipe root (None = the paper's cached Opus 4.7 recipe)
    "anthropic/claude-opus-4.7": None,
    "openai/gpt-6.1-sol": str(R / "multi_llm_v1" / "openai__gpt-6.1-sol"),
    "google/gemini-4-argon": str(R / "multi_llm_v1" / "google__gemini-4-argon"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.15])
    ap.add_argument("--arms", nargs="+", default=["llm_unres", "llm_fixed", "egbo"])
    ap.add_argument("--egbo-seeds", type=int, default=10)
    ap.add_argument("--llm-seeds", type=int, default=1)
    ap.add_argument("--domains", nargs="+", default=["meat", "dairy"])
    ap.add_argument("--categories", nargs="+", default=None)
    ap.add_argument("--noise-aware", action="store_true")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", type=Path, default=R / "batched_v1")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--seeds", type=int, default=None, help="seeds for egbo and baselines (default --egbo-seeds)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="LLM response cache (default <out>/llm_cache); point at results/batched_v1/llm_cache "
                         "to replay the 7-round runs' first rounds")
    a = ap.parse_args()
    cache = str(a.cache_dir or a.out / "llm_cache")
    if any(x.startswith("llm") for x in a.arms) and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("Set OPENROUTER_API_KEY")
    tasks = []
    base_done = set()
    for m in a.models:
        root = MODELS.get(m)
        for dom in a.domains:
            cats = sorted(json.loads((R / "final_protocol_core_v1" / dom / "summary.json").read_text()))
            for c in cats:
                if a.categories and c not in a.categories:
                    continue
                for lvl in a.levels:
                    for arm in a.arms:
                        if arm in BASELINES:
                            if (dom, c, lvl, arm) in base_done:
                                continue
                            base_done.add((dom, c, lvl, arm))
                        n = a.llm_seeds if arm.startswith("llm") else (a.seeds or a.egbo_seeds)
                        slug = "baselines" if arm in BASELINES else m.replace("/", "__")
                        for s in range(n):
                            p = a.out / slug / dom / c.replace(" ", "_") / f"{arm}_noise{lvl:g}_seed{s}.json"
                            tasks.append((m, root, dom, c, lvl, arm, s, str(p), cache, a.noise_aware, a.rounds))
    tasks.sort(key=lambda t: not t[5].startswith("llm"))
    print(f"{len(tasks)} tasks", flush=True)
    done = fails = 0
    from _run_utils import watch_parent
    with ProcessPoolExecutor(max_workers=a.workers, initializer=watch_parent, initargs=(os.getpid(),)) as ex:
        for f in as_completed([ex.submit(run_task, t) for t in tasks]):
            path, status = f.result()
            done += 1; fails += status.startswith("FAIL")
            print(f"[{done}/{len(tasks)}] {status.splitlines()[0]} {path.split('batched_v1/')[-1]}"
                  + ("\n" + status if status.startswith("FAIL") else ""), flush=True)
    print(f"done: {done} tasks, {fails} failures", flush=True)


if __name__ == "__main__":
    main()
