#!/usr/bin/env python3
"""Observation noise with another LLM expert (e.g. GPT-6.1 Sol).

Noise-free runs: results/multi_llm_v1/<slug>/ (run_multi_llm.py).
Noisy runs:      results/noise_sweep_<tag>/ (run_noise_sweep.py --recipe-root ...).
Scores are the true loss of the recommended recipe (best observed so far),
normalized by the category's mean random-search loss at evaluation 1. No API calls.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
R = HERE / "results"
E = (5, 10, 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="openai__gpt-6.1-sol")
    ap.add_argument("--noisy", type=Path, default=R / "noise_sweep_gpt61sol_v1")
    a = ap.parse_args()
    curves = defaultdict(dict)  # level -> arm -> {(dom,cat): curve}
    # noise-free (level 0) from the multi-LLM run
    import analyze_multi_llm as aml
    N = aml.norms()
    for cdir in sorted((R / "multi_llm_v1" / a.slug).glob("*/*")):
        meta = json.loads((cdir / "meta.json").read_text())
        key = (meta["domain"], meta["category"])
        norm = N[key][0]
        cur = lambda tl: np.minimum.accumulate(np.asarray(tl, float)) / norm
        curves[0.0].setdefault("llm_unres", {})[key] = cur(json.loads((cdir / "llm_unres.json").read_text())["true_loss"])
        curves[0.0].setdefault("llm_fixed", {})[key] = cur(json.loads((cdir / "llm_fixed.json").read_text())["true_loss"])
        curves[0.0].setdefault("egbo", {})[key] = np.mean(
            [cur(json.loads(p.read_text())["true_loss"]) for p in sorted(cdir.glob("egbo_seed*.json"))], 0)
    acc = defaultdict(list)
    best_true = defaultdict(list)
    for p in [q for d in ("meat", "dairy") for q in (a.noisy / d).glob("*/*.json")]:
        d = json.loads(p.read_text())
        k = (d["level"], d["arm"], d["domain"], d["category"])
        acc[k].append(np.asarray(d["recommended_true_loss"]) / d["norm"])
        best_true[k].append(np.asarray(d["best_true_so_far"]) / d["norm"])
    for (lvl, arm, dom, cat), v in acc.items():
        curves[lvl].setdefault(arm, {})[(dom, cat)] = np.mean(v, 0)
    summary = {}
    for lvl in sorted(curves):
        keys = sorted(set.intersection(*[set(curves[lvl][a_]) for a_ in ("llm_unres", "llm_fixed", "egbo")]))
        print(f"\n== {a.slug}, noise {lvl:g} x baseline loss ({len(keys)} categories) ==")
        row = {}
        for arm in ("llm_unres", "llm_fixed", "egbo"):
            m = np.array([curves[lvl][arm][k] for k in keys])
            row[arm] = [round(float(m[:, t - 1].mean()), 3) for t in E]
            print(f"  {arm:10s} e5/e10/e20 = {row[arm]}   improvement e5->e20 = {m[:, 4].mean() - m[:, 19].mean():.3f}")
        for t in E:
            llm = np.array([min(curves[lvl]['llm_unres'][k][t - 1], curves[lvl]['llm_fixed'][k][t - 1]) for k in keys])
            eg = np.array([curves[lvl]['egbo'][k][t - 1] for k in keys])
            p = wilcoxon(llm - eg).pvalue
            row[f"wins_e{t}"] = int((eg < llm).sum())
            print(f"  eval {t:2d}: EGBO beats best LLM arm in {(eg < llm).sum()}/{len(keys)}; mean adv {np.mean(llm - eg):+.3f}; Wilcoxon p={p:.2g}")
        if lvl > 0:
            for arm in ("llm_unres", "llm_fixed", "egbo"):
                bt = np.array([np.mean(best_true[(lvl, arm, *k)], 0) for k in keys])
                print(f"  best TRUE recipe ever evaluated, {arm:10s}: e5 {bt[:, 4].mean():.3f} -> e20 {bt[:, 19].mean():.3f}")
        summary[str(lvl)] = row
    out = a.noisy / "noise_summary_vs_noisefree.json"
    out.write_text(json.dumps(summary, indent=1))
    print("\nsaved", out)


if __name__ == "__main__":
    main()
