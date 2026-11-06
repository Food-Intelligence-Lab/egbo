#!/usr/bin/env python3
"""Analyze batched FormulateBench (run_batched.py): R rounds x 3 recipes (R read from the results).

Scores: true loss of the recommended recipe (best observed so far), normalized
by the category's mean random-search loss at evaluation 1; seeds averaged per
category, categories are the unit. Reports per model and noise level:
mean loss after selected rounds for the LLM arms, EGBO and (if present) the
conventional baselines (random search, vanilla BO, TuRBO; results/<root>/baselines);
EGBO vs the better LLM arm and vs each baseline (wins, mean advantage, Wilcoxon p);
and within-batch diversity (mean pairwise L1 distance within each round 2-R).
No API calls.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
R = HERE / "results"
ROUNDS_SHOW = (1, 3, 5, 7, 10)
BASE = ("random", "vanilla_bo", "turbo")


def diversity(d):
    out = []
    for b in d["batches"][1:]:
        X = [d["recipes"][i] for i in b]
        keys = sorted(set().union(*X))
        V = np.array([[x.get(k, 0.0) for k in keys] for x in X])
        out.append(np.mean([np.abs(V[i] - V[j]).sum() for i in range(len(V)) for j in range(i + 1, len(V))]))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=R / "batched_v1")
    a = ap.parse_args()
    cur = defaultdict(list)
    div = defaultdict(list)
    R_ = set()
    for p in a.root.glob("*/*/*/*.json"):
        if "llm_cache" in p.parts:
            continue
        d = json.loads(p.read_text())
        model = "baselines" if d["arm"] in BASE else d["model"]
        k = (model, d["level"], d["arm"], d["domain"], d["category"])
        rec = np.asarray(d["recommended_true_loss"]) / d["norm"]
        R_.add(d["rounds"])
        cur[k].append([rec[3 * r - 1] for r in range(1, d["rounds"] + 1)])
        div[k].append(diversity(d))
    assert len(R_) == 1, f"mixed round counts {R_}"
    RR = R_.pop()
    show = [r for r in ROUNDS_SHOW if r <= RR]
    summary = {}
    for model in sorted({k[0] for k in cur} - {"baselines"}):
        for lvl in sorted({k[1] for k in cur if k[0] == model}):
            cats = sorted({(k[3], k[4]) for k in cur if k[0] == model and k[1] == lvl})
            ok = [c for c in cats if all((model, lvl, arm, *c) in cur for arm in ("llm_unres", "llm_fixed", "egbo"))]
            if not ok:
                continue
            m = {arm: np.array([np.mean(cur[(model, lvl, arm, *c)], 0) for c in ok]) for arm in ("llm_unres", "llm_fixed", "egbo")}
            for b in BASE:
                if all(("baselines", lvl, b, *c) in cur for c in ok):
                    m[b] = np.array([np.mean(cur[("baselines", lvl, b, *c)], 0) for c in ok])
            print(f"\n== {model}, noise {lvl:g} ({len(ok)} categories) ==")
            row = {"n": len(ok)}
            for arm in m:
                row[arm] = [round(float(m[arm][:, r - 1].mean()), 3) for r in show]
                src = "baselines" if arm in BASE else model
                dv = np.mean([np.mean(div[(src, lvl, arm, *c)]) for c in ok])
                row[f"{arm}_diversity"] = round(float(dv), 3)
                print(f"  {arm:10s} loss after rounds {show} = {row[arm]}   batch diversity (L1) = {dv:.3f}")
            for r in sorted({2, 3, 5, RR}):
                llm = np.minimum(m["llm_unres"][:, r - 1], m["llm_fixed"][:, r - 1])
                adv = llm - m["egbo"][:, r - 1]
                p = wilcoxon(adv).pvalue if np.any(adv != 0) else float("nan")
                row[f"wins_r{r}"] = int((adv > 0).sum())
                print(f"  after round {r} (eval {3 * r:2d}): EGBO beats best LLM arm in {(adv > 0).sum()}/{len(ok)}; mean adv {adv.mean():+.3f}; p={p:.2g}")
            for b in [b for b in BASE if b in m]:
                adv = m[b][:, RR - 1] - m["egbo"][:, RR - 1]
                p = wilcoxon(adv).pvalue if np.any(adv != 0) else float("nan")
                row[f"wins_vs_{b}"] = int((adv > 0).sum())
                print(f"  final round: EGBO beats {b} in {(adv > 0).sum()}/{len(ok)}; mean adv {adv.mean():+.3f}; p={p:.2g}")
            row["rounds"] = RR
            row["curves"] = {arm: [round(float(v), 4) for v in m[arm].mean(0)] for arm in m}
            row["curves_ci"] = {arm: [round(float(1.96 * v), 4) for v in m[arm].std(0, ddof=1) / np.sqrt(len(ok))] for arm in m}
            summary[f"{model}|{lvl}"] = row
    (a.root / "batched_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
