#!/usr/bin/env python3
"""FormulateBench observation-noise sweep: iterative LLM vs one-shot EGBO.

Each method sees a noisy score y_obs = y_true + N(0, (level * norm)^2), where
`norm` is the category's mean random-search loss at evaluation 1 (the same
normalizer the paper's figures use). Every evaluation is logged with its true
(noiseless) loss, so methods are scored by the TRUE loss of the recipe they
would recommend (the one with the best observed score so far), not by the
best noisy observation, which would reward noise-chasing.

All arms start from the cached shared LLM recipe used in the paper (evaluation
1), exactly as in the noiseless protocol. One-shot EGBO keeps the cached
ingredient support; only its BO refinement is re-run under noise.

Arms: "egbo" (LLM-configured BO, 10 seeds), "llm_unres" (unrestricted iterative
LLM), "llm_fixed" (fixed-support iterative LLM). Results are resumable: each
(category, level, arm, seed) is one JSON file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

EXCL = {"meat": ["egg white", "egg whites", "fat", "milk", "milk proteins"], "dairy": ["mozzarella"]}
R = HERE / "results"


def _setup(domain, category, recipe_root=None):
    import run_formulatebench as fb
    df = fb.load_ingredients(domain)
    ex = {x.casefold() for x in EXCL[domain]}
    df = df[~df["ingredient"].astype(str).str.strip().str.casefold().isin(ex)].reset_index(drop=True)
    M = fb.get_nutrition_matrix(df)
    target, mask = fb.load_targets(domain)[category]
    objective = fb.make_objective(M, target, mask)
    core = json.loads((R / "final_protocol_core_v1" / domain / "summary.json").read_text())[category]
    rs = [k for k in core if k.startswith("Random Search")][0]
    norm = float(np.mean(-np.asarray(core[rs])[:, 0]))
    if recipe_root:  # first recipe from run_multi_llm.py for another model
        meta = json.loads((Path(recipe_root) / domain / category.replace(" ", "_") / "meta.json").read_text())
        raw = {"config": {"shared_initial_recipe": meta["shared_recipe"]}}
    else:
        raw = json.loads((R / "final_protocol_egbo_llm_config_v1" / domain
                          / category.replace(" ", "_") / "raw.json").read_text())
    lookup = {str(n).strip().casefold(): i for i, n in enumerate(df["ingredient"])}
    x0 = np.zeros(len(df))
    selected = []
    for item in raw["config"]["shared_initial_recipe"]:
        i = lookup[item["ingredient"].strip().casefold()]
        if i not in selected:
            selected.append(i)
        x0[i] += float(item["fraction"])
    x0 /= x0.sum()
    return fb, df, M, target, mask, objective, norm, x0, selected, raw["config"]["shared_initial_recipe"]


class Logged:
    """Wrap the true objective (returns -loss); add noise and log every call."""

    def __init__(self, objective, sigma, seed, names):
        self.f, self.sigma, self.names = objective, sigma, names
        self.rng = np.random.default_rng(seed)
        self.true, self.obs, self.recipes = [], [], []

    def __call__(self, x):
        x = np.asarray(x, float)
        y = float(self.f(x))
        y_obs = y + (float(self.rng.normal(0.0, self.sigma)) if self.sigma > 0 else 0.0)
        self.true.append(-y)
        self.obs.append(-y_obs)
        nz = np.flatnonzero(x > 1e-6)
        self.recipes.append({self.names[i]: round(float(x[i]), 5) for i in nz})
        return y_obs


def run_task(task):
    domain, category, level, arm, seed, out_path, model, recipe_root, cache_dir, noise_aware, known_noise = task
    if Path(out_path).exists():
        return out_path, "skip"
    try:
        import torch
        torch.set_num_threads(1)
        fb, df, M, target, mask, objective, norm, x0, selected, recipe0 = _setup(domain, category, recipe_root)
        sigma = level * norm
        arm_id = {"egbo": 0, "llm_unres": 1, "llm_fixed": 2}[arm]
        noise_seed = int(1_000_003 * arm_id + 10_007 * seed + int(level * 1000))
        names = [str(n) for n in df["ingredient"]]
        note = (f"Note: the evaluator's scores include random measurement noise. Each reported "
                f"target dissimilarity equals the formulation's true dissimilarity plus Gaussian noise "
                f"with standard deviation {sigma:.4f}, so repeated evaluations of the same formulation "
                f"can differ.") if (noise_aware and sigma > 0) else None
        if arm == "egbo":
            import egbo_bo_config as configured
            obj = Logged(objective, sigma, noise_seed, names)
            configured.reduced_bo_cfg(
                objective=obj, d_full=len(df), selected_indices=selected, budget=20, n_init=5,
                seed=seed, cfg=configured.BOConfig(known_noise_var=(sigma ** 2 if known_noise else None)), fit_mll=fb._fit_gpytorch_mll_with_warning,
                softmax=fb._softmax, init_logits=fb._simplex_initial_logits,
                initial_x=x0, initial_design="local", local_sigma=0.35)
        else:
            client = fb.OpenRouterClient(os.environ["OPENROUTER_API_KEY"], model,
                                         cache_dir=cache_dir)
            if arm == "llm_unres":
                obj = Logged(objective, sigma, noise_seed, names)
                fb.iterative_llm_optimizer(
                    obj, df, category, target, mask, M, budget=20, llm_client=client,
                    max_ingredients=9, temperature=0.0, information_regime="semantic-only", feedback_note=note,
                    initial_x=x0, initial_recipe=recipe0)
            else:
                sub = df.iloc[selected].reset_index(drop=True)
                sub_names = [str(n) for n in sub["ingredient"]]

                def restricted(x_small):
                    x = np.zeros(len(df))
                    x[selected] = x_small
                    return objective(x)
                obj = Logged(restricted, sigma, noise_seed, sub_names)
                fb.iterative_llm_optimizer(
                    obj, sub, category, target, mask, M[selected], budget=20, llm_client=client,
                    max_ingredients=len(selected), temperature=0.0,
                    information_regime="semantic-only", feedback_note=note, initial_x=x0[selected], initial_recipe=recipe0)
        if len(obj.true) != 20:
            raise RuntimeError(f"expected 20 evaluations, got {len(obj.true)}")
        # Recommended recipe at each t = argmin observed loss so far; score its TRUE loss.
        rec_true, best_i = [], 0
        for t in range(len(obj.obs)):
            if obj.obs[t] < obj.obs[best_i]:
                best_i = t
            rec_true.append(obj.true[best_i])
        out = dict(domain=domain, category=category, level=level, sigma=sigma, norm=norm, arm=arm,
                   seed=seed, noise_seed=noise_seed, model=model if arm != "egbo" else None,
                   true_loss=obj.true, observed_loss=obj.obs, recommended_true_loss=rec_true,
                   best_true_so_far=list(np.minimum.accumulate(obj.true)), recipes=obj.recipes)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(out_path) + ".tmp")
        tmp.write_text(json.dumps(out))
        tmp.replace(out_path)
        return out_path, "ok"
    except Exception:
        return out_path, "FAIL\n" + traceback.format_exc()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--levels", type=float, nargs="+", default=[0.05, 0.15])
    p.add_argument("--arms", nargs="+", default=["llm_unres", "llm_fixed", "egbo"])
    p.add_argument("--egbo-seeds", type=int, default=10)
    p.add_argument("--llm-seeds", type=int, default=1)
    p.add_argument("--domains", nargs="+", default=["meat", "dairy"])
    p.add_argument("--categories", nargs="+", default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--model", default="anthropic/claude-opus-4.7")
    p.add_argument("--out", type=Path, default=R / "noise_sweep_v1")
    p.add_argument("--egbo-known-noise", action="store_true",
                   help="Give EGBO's GP the true noise variance (fixed) instead of learning it")
    p.add_argument("--noise-aware", action="store_true",
                   help="Tell the LLM that scores contain Gaussian measurement noise of the true SD")
    p.add_argument("--recipe-root", default=None,
                   help="Use another model's first recipe/support from run_multi_llm.py, "
                        "e.g. results/multi_llm_v1/openai__gpt-6.1-sol (default: the paper's Opus 4.7 recipe)")
    a = p.parse_args()
    if any(arm.startswith("llm") for arm in a.arms) and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("Set OPENROUTER_API_KEY")
    tasks = []
    for dom in a.domains:
        cats = sorted(json.loads((R / "final_protocol_core_v1" / dom / "summary.json").read_text()))
        for cat in cats:
            if a.categories and cat not in a.categories:
                continue
            for level in a.levels:
                for arm in a.arms:
                    n = a.egbo_seeds if arm == "egbo" else a.llm_seeds
                    for s in range(n):
                        path = a.out / dom / cat.replace(" ", "_") / f"{arm}_noise{level:g}_seed{s}.json"
                        tasks.append((dom, cat, level, arm, s, str(path), a.model, a.recipe_root, str(a.out / "llm_cache"), a.noise_aware, a.egbo_known_noise))
    tasks.sort(key=lambda t: t[3] == "egbo")  # LLM tasks (slow, I/O-bound) first
    print(f"{len(tasks)} tasks", flush=True)
    done = fails = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_task, t) for t in tasks]
        for f in as_completed(futs):
            path, status = f.result()
            done += 1
            if status.startswith("FAIL"):
                fails += 1
                print(f"[{done}/{len(tasks)}] {path}\n{status}", flush=True)
            else:
                print(f"[{done}/{len(tasks)}] {status} {Path(path).parent.name}/{Path(path).name}", flush=True)
    print(f"done: {done} tasks, {fails} failures", flush=True)


if __name__ == "__main__":
    main()
