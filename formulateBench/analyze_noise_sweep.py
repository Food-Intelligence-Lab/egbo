#!/usr/bin/env python3
"""Summarize the observation-noise sweep (run_noise_sweep.py) with the noiseless
cached runs as level 0. Scores are the TRUE loss of each method's recommended
recipe (best observed so far), divided by the category's random-search eval-1
loss. Per category, seeds are averaged first; categories are the unit.

A robustness section at the end additionally reports paired Wilcoxon tests,
the eval-5 -> eval-20 improvement per arm, and a "best true recipe ever
evaluated" variant (field ``best_true_so_far``) that separates search (did the
method ever evaluate a good recipe?) from selection (could it identify it
under noise?). It is written to noise_robustness.json; noise_summary.json and
the figure are unchanged."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
R = HERE / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else R / "noise_sweep_v1"
ARMS = {"llm_unres": "Unrestricted iterative LLM", "llm_fixed": "Fixed-support iterative LLM",
        "egbo": "One-shot EGBO (LLM-configured BO)"}

# curves[level][arm][(dom, cat)] -> mean normalized curve (20,)
curves = defaultdict(lambda: defaultdict(dict))
# Same, but scoring the best TRUE loss among all recipes evaluated so far.
curves_bt = defaultdict(lambda: defaultdict(dict))
SEM = {"meat": R / "meat_semantic_all14_v1", "dairy": R / "dairy_semantic_all10_v3"}
for dom in ["meat", "dairy"]:
    core = json.loads((R / "final_protocol_core_v1" / dom / "summary.json").read_text())
    for cat, c in core.items():
        rs = [k for k in c if k.startswith("Random Search")][0]
        norm = float(np.mean(-np.asarray(c[rs])[:, 0]))
        d = cat.replace(" ", "_")
        sem = json.loads((SEM[dom] / d / "raw.json").read_text())["methods"]
        eg = json.loads((R / "final_protocol_egbo_llm_config_v1" / dom / d / "raw.json").read_text())["methods"]
        f = lambda recs: (-np.asarray([r["best_values"] for r in recs])).mean(0) / norm
        curves[0.0]["llm_unres"][(dom, cat)] = f(sem["Unrestricted iterative LLM"])
        curves[0.0]["llm_fixed"][(dom, cat)] = f(sem["Restricted iterative LLM"])
        curves[0.0]["egbo"][(dom, cat)] = f(eg["EGBO (LLM-configured BO)"])
        # Noiseless: recommended recipe == best true recipe evaluated so far.
        for a in ("llm_unres", "llm_fixed", "egbo"):
            curves_bt[0.0][a][(dom, cat)] = curves[0.0][a][(dom, cat)]

acc = defaultdict(list)
acc_bt = defaultdict(list)
for p in [q for dom in ("meat", "dairy") for q in (OUT / dom).glob("*/*.json")]:
    d = json.loads(p.read_text())
    acc[(d["level"], d["arm"], d["domain"], d["category"])].append(np.asarray(d["recommended_true_loss"]) / d["norm"])
    acc_bt[(d["level"], d["arm"], d["domain"], d["category"])].append(np.asarray(d["best_true_so_far"]) / d["norm"])
for (lvl, arm, dom, cat), v in acc.items():
    curves[lvl][arm][(dom, cat)] = np.mean(v, 0)
for (lvl, arm, dom, cat), v in acc_bt.items():
    curves_bt[lvl][arm][(dom, cat)] = np.mean(v, 0)

levels = sorted(curves)
keys = sorted(curves[0.0]["egbo"])
print("Runs found per level/arm:", {l: {a: len(curves[l][a]) for a in curves[l]} for l in levels})
summary = {}
for lvl in levels:
    ok = [k for k in keys if all(k in curves[lvl][a] for a in ARMS)]
    print(f"\n== noise level {lvl:g} x norm  ({len(ok)} complete categories) ==")
    row = {}
    for a in ARMS:
        m = np.array([curves[lvl][a][k] for k in ok])
        row[a] = {e: float(m[:, e - 1].mean()) for e in (5, 10, 20)}
        print(f"  {ARMS[a]:36s} " + "  ".join(f"e{e}={row[a][e]:.3f}" for e in (5, 10, 20)))
    for e in (5, 10, 20):
        llm = np.array([min(curves[lvl]["llm_unres"][k][e - 1], curves[lvl]["llm_fixed"][k][e - 1]) for k in ok])
        eg = np.array([curves[lvl]["egbo"][k][e - 1] for k in ok])
        row[f"egbo_beats_best_llm_e{e}"] = int((eg < llm).sum())
        print(f"  eval {e:2d}: EGBO beats best LLM arm in {int((eg < llm).sum())}/{len(ok)}; "
              f"mean advantage {np.mean(llm - eg):+.3f}")
    summary[lvl] = row
(OUT / "noise_summary.json").write_text(json.dumps({str(k): v for k, v in summary.items()}, indent=1))

# Figure: mean curves per noise level (one panel per level), same colors as the paper.
C = {"llm_unres": ("#4c3f91", "--"), "llm_fixed": ("#8c564b", (0, (5, 1, 1, 1))), "egbo": ("#2ecc71", "-")}
fig, axes = plt.subplots(1, len(levels), figsize=(4.3 * len(levels), 3.6), sharey=True)
for ax, lvl in zip(np.atleast_1d(axes), levels):
    ok = [k for k in keys if all(k in curves[lvl][a] for a in ARMS)]
    x = np.arange(1, 21)
    for a in ARMS:
        m = np.array([curves[lvl][a][k] for k in ok])
        mu, se = m.mean(0), m.std(0, ddof=1) / np.sqrt(len(ok))
        ax.plot(x, mu, color=C[a][0], ls=C[a][1], lw=2.2, label=ARMS[a])
        ax.fill_between(x, mu - 1.96 * se, mu + 1.96 * se, color=C[a][0], alpha=0.12, lw=0)
    ax.set_title(f"Noise σ = {lvl:g} × baseline loss" if lvl else "No noise (paper setting)", loc="left", fontsize=10)
    ax.set_xlabel("Evaluation"); ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(color="#e6e6e6", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
np.atleast_1d(axes)[0].set_ylabel("True loss of recommended recipe\n(normalized, lower is better)")
np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"formulatebench_noise.{ext}", dpi=200, bbox_inches="tight")
print("\nsaved", OUT / "formulatebench_noise.png")


# ---------------------------------------------------------------------------
# Robustness section (additional outputs only; nothing above depends on it)
# ---------------------------------------------------------------------------
def _best_llm(cv, lvl, k, e):
    return min(cv[lvl]["llm_unres"][k][e - 1], cv[lvl]["llm_fixed"][k][e - 1])


robust = {}
print("\n== Robustness ==")
print("(a) Paired Wilcoxon signed-rank test over categories of (best LLM arm - EGBO), "
      "two-sided; positive = EGBO better")
for lvl in levels:
    ok = [k for k in keys if all(k in curves[lvl][a] for a in ARMS)]
    r = robust.setdefault(str(lvl), {})
    for e in (5, 20):
        diff = np.array([_best_llm(curves, lvl, k, e) - curves[lvl]["egbo"][k][e - 1] for k in ok])
        stat, pval = wilcoxon(diff)
        r[f"wilcoxon_p_e{e}"] = float(pval)
        print(f"  noise {lvl:<4g} eval {e:2d}: median diff {np.median(diff):+.3f}, "
              f"W={stat:.1f}, p={pval:.2g} (n={len(ok)})")

print("(b) Mean improvement from eval 5 to eval 20 (mean e5 loss - mean e20 loss)")
for lvl in levels:
    ok = [k for k in keys if all(k in curves[lvl][a] for a in ARMS)]
    r = robust[str(lvl)]
    parts = []
    for a in ARMS:
        m = np.array([curves[lvl][a][k] for k in ok])
        imp = float(m[:, 4].mean() - m[:, 19].mean())
        r[f"improvement_e5_to_e20_{a}"] = imp
        parts.append(f"{a}={imp:.3f}")
    print(f"  noise {lvl:<4g} " + "  ".join(parts))

print("(c) Best TRUE recipe ever evaluated (search only, no selection under noise)")
for lvl in levels:
    ok = [k for k in keys if all(k in curves_bt[lvl][a] for a in ARMS)]
    r = robust[str(lvl)]
    for a in ARMS:
        m = np.array([curves_bt[lvl][a][k] for k in ok])
        r[f"best_true_{a}"] = {e: float(m[:, e - 1].mean()) for e in (5, 10, 20)}
    print(f"  noise {lvl:<4g} " + "  ".join(
        f"{a}: e5={r[f'best_true_{a}'][5]:.3f} e20={r[f'best_true_{a}'][20]:.3f}" for a in ARMS))
    for e in (5, 20):
        llm = np.array([_best_llm(curves_bt, lvl, k, e) for k in ok])
        eg = np.array([curves_bt[lvl]["egbo"][k][e - 1] for k in ok])
        r[f"best_true_egbo_beats_best_llm_e{e}"] = int((eg < llm).sum())
        print(f"    eval {e:2d}: EGBO beats best LLM arm in {int((eg < llm).sum())}/{len(ok)}; "
              f"mean advantage {np.mean(llm - eg):+.3f}")
(OUT / "noise_robustness.json").write_text(json.dumps(robust, indent=1))
print("saved", OUT / "noise_robustness.json")
