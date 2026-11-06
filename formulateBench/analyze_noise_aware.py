#!/usr/bin/env python3
"""Noise-aware prompt arm vs uninformed LLM vs EGBO at noise level 0.15 (eval 20).

The noise-aware LLM is told the exact noise SD; EGBO is not. Scores are the true
loss of the recommended recipe (best observed so far) / category norm.
Pre-registered rule for running a known-noise EGBO arm: the noise-aware LLM (best
of its two arms) ties or beats EGBO in >= 12/24 categories, or cuts EGBO's mean
advantage by more than half relative to the uninformed LLM. No API calls.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent / "results"
LVL, T = 0.15, 20
PAIRS = {"Opus 4.7": ("noise_sweep_v1", "noise_aware_opus_v1"),
         "GPT-6.1 Sol": ("noise_sweep_gpt61sol_v1", "noise_aware_gpt61sol_v1")}


def load(root, arms, seeds=None):
    acc = defaultdict(list)
    for p in (R / root).glob("*/*/*.json"):
        d = json.loads(p.read_text())
        if d.get("level") != LVL or d["arm"] not in arms or (seeds is not None and d["seed"] not in seeds):
            continue
        acc[(d["arm"], d["domain"], d["category"])].append(d["recommended_true_loss"][T - 1] / d["norm"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


out = {}
for name, (base, aware) in PAIRS.items():
    egbo = load(base, {"egbo"})
    rows = {}
    for label, root in (("uninformed", base), ("noise_aware", aware)):
        llm = load(root, {"llm_unres", "llm_fixed"}, seeds={0})
        cats = sorted({k[1:] for k in llm if ("llm_unres", *k[1:]) in llm and ("llm_fixed", *k[1:]) in llm
                       and ("egbo", *k[1:]) in egbo})
        b = np.array([min(llm[("llm_unres", *c)], llm[("llm_fixed", *c)]) for c in cats])
        e = np.array([egbo[("egbo", *c)] for c in cats])
        rows[label] = dict(n=len(cats), best_llm=round(b.mean(), 3), egbo=round(e.mean(), 3),
                           llm_ties_or_beats=int((b <= e).sum()), egbo_adv=round(float((b - e).mean()), 3))
    u, w = rows["uninformed"], rows["noise_aware"]
    rows["rule_triggered"] = bool(w["llm_ties_or_beats"] >= 12 or w["egbo_adv"] < 0.5 * u["egbo_adv"])
    out[name] = rows
    print(name, json.dumps(rows))
(R / "noise_aware_summary.json").write_text(json.dumps(out, indent=1))
