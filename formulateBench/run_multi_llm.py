#!/usr/bin/env python3
"""FormulateBench with other LLM experts (cross-model replication).

For each model and category this reruns the paper's semantic-only protocol:
  1. Unrestricted iterative LLM (20 evaluations); its first recipe is the shared
     evaluation 1 for every informed arm, exactly as in the paper.
  2. Fixed-support iterative LLM from that shared recipe.
  3. One-shot EGBO on the shared recipe's ingredients, with the paper's
     LLM-configured BO (Matern-5/2, default lengthscale prior, learned noise,
     qLogNEI, ARD), local initialization around the shared recipe, 10 seeds.
Results: results/multi_llm_v1/<model_slug>/<domain>/<category>/{llm_unres,llm_fixed,egbo_seed<k>}.json
Every evaluation's true loss is recorded. Resumable. Requires OPENROUTER_API_KEY
(LLM responses are cached under results/multi_llm_v1/llm_cache).
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
R = HERE / "results"
EXCL = {"meat": ["egg white", "egg whites", "fat", "milk", "milk proteins"], "dairy": ["mozzarella"]}


def _atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def _setup(domain, category):
    import run_formulatebench as fb
    df = fb.load_ingredients(domain)
    ex = {x.casefold() for x in EXCL[domain]}
    df = df[~df["ingredient"].astype(str).str.strip().str.casefold().isin(ex)].reset_index(drop=True)
    M = fb.get_nutrition_matrix(df)
    target, mask = fb.load_targets(domain)[category]
    return fb, df, M, target, mask, fb.make_objective(M, target, mask)


def run_task(task):
    model, domain, category, egbo_seeds, out_root = task
    slug = model.replace("/", "__")
    d = Path(out_root) / slug / domain / category.replace(" ", "_")
    try:
        import torch
        torch.set_num_threads(1)
        fb, df, M, target, mask, objective = _setup(domain, category)
        client = fb.OpenRouterClient(os.environ["OPENROUTER_API_KEY"], model,
                                     cache_dir=str(Path(out_root) / "llm_cache"))
        # 1. unrestricted iterative LLM (generates the shared first recipe)
        p_unres = d / "llm_unres.json"
        if not p_unres.exists():
            rec = fb.iterative_llm_optimizer(
                objective, df, category, target, mask, M, budget=20, llm_client=client,
                max_ingredients=9, temperature=0.0, information_regime="semantic-only")
            rec["true_loss"] = [-y for y in rec["all_y"]]
            rec["model"] = model
            _atomic(p_unres, rec)
        unres = json.loads(p_unres.read_text())
        recipe0 = unres["recipe_history"][0]["recipe"]
        lookup = {str(n).strip().casefold(): i for i, n in enumerate(df["ingredient"])}
        x0 = np.zeros(len(df))
        selected = []
        for item in recipe0:
            i = lookup[str(item["ingredient"]).strip().casefold()]
            if i not in selected:
                selected.append(i)
            x0[i] += float(item["fraction"])
        x0 /= x0.sum()
        _, o_sup = fb.nnls_simplex_oracle(M, target, mask, selected_indices=selected)
        _, o_full = fb.nnls_simplex_oracle(M, target, mask)
        meta = dict(model=model, domain=domain, category=category,
                    shared_recipe=recipe0, shared_support=[str(df["ingredient"].iloc[i]) for i in selected],
                    initial_loss=-float(objective(x0)), oracle_support=float(o_sup), oracle_full=float(o_full))
        _atomic(d / "meta.json", meta)
        # 2. fixed-support iterative LLM
        p_fixed = d / "llm_fixed.json"
        if not p_fixed.exists():
            sub = df.iloc[selected].reset_index(drop=True)

            def restricted(xs):
                x = np.zeros(len(df))
                x[selected] = xs
                return objective(x)
            rec = fb.iterative_llm_optimizer(
                restricted, sub, category, target, mask, M[selected], budget=20, llm_client=client,
                max_ingredients=len(selected), temperature=0.0, information_regime="semantic-only",
                initial_x=x0[selected], initial_recipe=recipe0)
            rec["true_loss"] = [-y for y in rec["all_y"]]
            rec["model"] = model
            _atomic(p_fixed, rec)
        # 3. one-shot EGBO with the paper's LLM-configured BO
        import egbo_bo_config as configured
        for s in range(egbo_seeds):
            p = d / f"egbo_seed{s}.json"
            if p.exists():
                continue
            rec = configured.reduced_bo_cfg(
                objective=objective, d_full=len(df), selected_indices=selected, budget=20, n_init=5,
                seed=s, cfg=configured.BOConfig(), fit_mll=fb._fit_gpytorch_mll_with_warning,
                softmax=fb._softmax, init_logits=fb._simplex_initial_logits,
                initial_x=x0, initial_design="local", local_sigma=0.35)
            rec["true_loss"] = [-y for y in rec["all_y"]]
            _atomic(p, rec)
        return str(d), "ok"
    except Exception:
        return str(d), "FAIL\n" + traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--domains", nargs="+", default=["meat", "dairy"])
    ap.add_argument("--categories", nargs="+", default=None)
    ap.add_argument("--egbo-seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=R / "multi_llm_v1")
    a = ap.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("Set OPENROUTER_API_KEY")
    tasks = []
    for m in a.models:
        for dom in a.domains:
            cats = sorted(json.loads((R / "final_protocol_core_v1" / dom / "summary.json").read_text()))
            for c in cats:
                if a.categories and c not in a.categories:
                    continue
                tasks.append((m, dom, c, a.egbo_seeds, str(a.out)))
    print(f"{len(tasks)} tasks", flush=True)
    done = fails = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(run_task, t) for t in tasks]):
            path, status = f.result()
            done += 1
            fails += status.startswith("FAIL")
            print(f"[{done}/{len(tasks)}] {status.splitlines()[0]} {path.split('multi_llm_v1/')[-1]}"
                  + ("\n" + status if status.startswith("FAIL") else ""), flush=True)
    print(f"done: {done} tasks, {fails} failures", flush=True)


if __name__ == "__main__":
    main()
