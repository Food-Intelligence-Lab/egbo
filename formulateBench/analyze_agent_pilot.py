#!/usr/bin/env python3
"""Analyze the tool-using agent pilot (run_agent_pilot.py). No API calls.

Strategy (from the code the agent ran, keyword-based; spot-check transcripts):
  none       - never used the python tool
  analysis   - used python, but fit no model and ran no optimizer
  model      - fit a surrogate (regression, random forest, ...) or ran a generic
               optimizer (scipy.optimize, ...) on the data, but no GP/BO
  gp_bo      - fit a Gaussian process / used a BO library
Also: whether a GP modeled observation noise (WhiteKernel, alpha, noise
likelihood), and how often the agent re-evaluated a near-identical formulation
(L1 distance < 0.02 to an earlier one), a direct way to average out noise.

Performance at evaluation 20 (true loss of the recommended recipe = best
observed, / category norm) against the same model's noise-sweep arms at the
same noise level: iterative LLM (same prompt, no tools), the better of the two
LLM arms, and EGBO (mean over seeds). Categories are the unit.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

R = Path(__file__).resolve().parent / "results"
NOISE_ROOT = {"anthropic/claude-opus-4.7": "noise_sweep_v1", "openai/gpt-6.1-sol": "noise_sweep_gpt61sol_v1",
              "google/gemini-4-argon": "noise_sweep_gemini4argon_v1"}
GP = re.compile(r"GaussianProcess|gaussian_process|botorch|gpytorch|skopt|bayes_opt|optuna|\bGPy\b", re.I)
NOISE_GP = re.compile(r"WhiteKernel|alpha\s*=|GaussianLikelihood|noise", re.I)
MODEL = re.compile(r"scipy\.optimize|from scipy import optimize|minimize\(|differential_evolution|nelder|"
                   r"LinearRegression|Ridge|Lasso|RandomForest|GradientBoosting|polyfit|lstsq|PolynomialFeatures|"
                   r"KernelRidge|SVR|MLPRegressor|curve_fit", re.I)


def strategy(code_cells):
    code = "\n".join(code_cells)
    if not code_cells:
        return "none", False
    if GP.search(code):
        return "gp_bo", bool(NOISE_GP.search(code))
    if MODEL.search(code):
        return "model", False
    return "analysis", False


def repeats(recipes):
    keys = sorted(set().union(*recipes))
    V = np.array([[r.get(k, 0.0) for k in keys] for r in recipes])
    return int(sum(any(np.abs(V[i] - V[j]).sum() < 0.02 for j in range(i)) for i in range(1, len(V))))


def baseline(root, level, T=20):
    acc = defaultdict(list)
    for p in (R / root).glob("*/*/*.json"):
        d = json.loads(p.read_text())
        if d.get("level") != level or (d["arm"] != "egbo" and d["seed"] != 0):
            continue
        acc[(d["arm"], d["domain"], d["category"])].append(d["recommended_true_loss"][T - 1] / d["norm"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=R / "agent_pilot_v1")
    a = ap.parse_args()
    runs = [json.loads(p.read_text()) for p in a.root.glob("*/*/*/agent*.json") if "transcript" not in p.name]
    summary = {}
    for model, level, arm in sorted({(r["model"], r["level"], r["arm"]) for r in runs}):
        if True:
            rs = [r for r in runs if r["model"] == model and r["level"] == level and r["arm"] == arm]
            print(f"\n== {model}, {arm}, noise {level:g}: {len(rs)} runs, "
                  f"{sum(r['complete'] for r in rs)} used all 20 evaluations ==")
            st = [strategy(r["python_code"]) for r in rs]
            cnt = Counter(s for s, _ in st)
            row = {"n": len(rs), "strategy": dict(cnt), "gp_models_noise": sum(n for _, n in st),
                   "python_calls_mean": round(float(np.mean([r["n_python_calls"] for r in rs])), 1),
                   "api_calls_mean": round(float(np.mean([r["n_api_calls"] for r in rs])), 1),
                   "cost_usd": round(float(sum(r["cost_usd"] for r in rs)), 2),
                   "runs_with_repeat_evals": sum(repeats(r["recipes"]) > 0 for r in rs)}
            gp_first = []
            for r in rs:
                ev = [e["n_evals"] for e in r["events"] if e["tool"] == "python"]
                hits = [n for c, n in zip(r["python_code"], ev) if GP.search(c)]
                if hits:
                    gp_first.append(hits[0])
            row["fit_any_model"] = cnt.get("gp_bo", 0) + cnt.get("model", 0)
            row["gp_first_eval_median"] = float(np.median(gp_first)) if gp_first else None
            row["runs_importing_botorch"] = sum(bool(re.search(r"botorch", "\n".join(r["python_code"]), re.I)) for r in rs)
            print(f"  fit any model: {row['fit_any_model']}/{len(rs)}; first GP at eval (median) {row['gp_first_eval_median']}; "
                  f"runs importing BoTorch: {row['runs_importing_botorch']}")
            print(f"  strategy: {dict(cnt)}; GP runs modelling noise: {row['gp_models_noise']}; "
                  f"runs re-evaluating a formulation: {row['runs_with_repeat_evals']}")
            print(f"  python calls/run {row['python_calls_mean']}, API calls/run {row['api_calls_mean']}, "
                  f"cost ${row['cost_usd']}")
            base = baseline(NOISE_ROOT[model], level) if model in NOISE_ROOT else {}
            done = [r for r in rs if r["complete"]]
            ag = {(r["domain"], r["category"]): r["recommended_true_loss"][-1] / r["norm"] for r in done}
            own = {(r["domain"], r["category"]): r["own_pick_true_loss"] / r["norm"]
                   for r in done if r["own_pick_true_loss"] is not None}
            e5 = np.mean([r["recommended_true_loss"][4] / r["norm"] for r in done]) if done else float("nan")
            row["agent_e5"], row["agent_e20"] = round(float(e5), 3), round(float(np.mean(list(ag.values()))), 3)
            row["agent_own_pick_e20"] = round(float(np.mean(list(own.values()))), 3) if own else None
            print(f"  agent: e5 {row['agent_e5']} -> e20 {row['agent_e20']}; own pick at e20 {row['agent_own_pick_e20']}")
            for label, f in (("iterative LLM, same prompt", lambda c: base.get(("llm_unres", *c))),
                             ("best LLM arm", lambda c: min(base[("llm_unres", *c)], base[("llm_fixed", *c)])
                              if ("llm_unres", *c) in base and ("llm_fixed", *c) in base else None),
                             ("EGBO", lambda c: base.get(("egbo", *c)))):
                cats = [c for c in ag if f(c) is not None]
                if not cats:
                    continue
                d = np.array([f(c) - ag[c] for c in cats])  # >0: agent better
                p = wilcoxon(d).pvalue if len(cats) > 5 and np.any(d != 0) else float("nan")
                row[f"vs {label}"] = dict(n=len(cats), other=round(float(np.mean([f(c) for c in cats])), 3),
                                          agent_better=int((d > 0).sum()), mean_diff=round(float(d.mean()), 3))
                print(f"  vs {label:27s}: {np.mean([f(c) for c in cats]):.3f}; agent better in "
                      f"{(d > 0).sum()}/{len(cats)}; mean {d.mean():+.3f} (positive = agent better); p={p:.2g}")
            summary[f"{model}|{arm}|{level}"] = row
    (a.root / "agent_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
