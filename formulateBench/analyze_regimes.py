#!/usr/bin/env python3
"""Regime analysis on cached FormulateBench results (no new runs, no API calls).

Asks *when* each strategy family wins, and why, using only the released
result files.

Arms
  LLM only : Unrestricted iterative LLM, Fixed-support (restricted) iterative LLM
  BO only  : Vanilla BO (d=full, archived 10 seeds), Warm-start full-space BO
  EGBO     : LLM-configured EGBO (final protocol, 10 seeds)
Loss = normalized distance (divided by mean random-search loss at eval 1),
lower is better. Per category, each arm's curve is the mean over seeds of the
best-so-far loss.

Per-category regime descriptors
  sup_gap  = oracle loss on the LLM's ingredient support - oracle loss on all
             ingredients (error in *which* ingredients the LLM picked)
  init_gap = loss of the LLM's first recipe - oracle loss on its support
             (error in the LLM's *amounts*)

Usage (from the repository root or formulateBench/):
  python formulateBench/analyze_regimes.py
  python formulateBench/analyze_regimes.py --results-dir formulateBench/results \
      --output-dir formulateBench/results/regime_analysis_v1

Outputs (in --output-dir): regime_rows.json, regime_table.json,
formulatebench_regimes.{pdf,png}.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_formulatebench as fb  # noqa: E402

EXCL = {"meat": ["egg white", "egg whites", "fat", "milk", "milk proteins"], "dairy": ["mozzarella"]}
SEM_DIRS = {"meat": "meat_semantic_all14_v1", "dairy": "dairy_semantic_all10_v3"}


def load_dir(p):
    out = {}
    for f in p.glob("*/raw.json"):
        d = json.loads(f.read_text())
        out[d["config"]["category"]] = d
    return out


def loss(records):
    return -np.asarray([r["best_values"] for r in records], float)  # seeds x T


def crossover(a, b):
    """First eval (1-based) from which a <= b for all later evals; None if never."""
    better = a <= b + 1e-12
    for t in range(len(a)):
        if better[t:].all():
            return t + 1
    return None


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def build_rows(R):
    rows = []
    for dom in ["meat", "dairy"]:
        core = json.loads((R / "final_protocol_core_v1" / dom / "summary.json").read_text())
        sem = load_dir(R / SEM_DIRS[dom])
        egbo = load_dir(R / "final_protocol_egbo_llm_config_v1" / dom)
        df = fb.load_ingredients(dom)
        ex = {x.casefold() for x in EXCL[dom]}
        df = df[~df["ingredient"].astype(str).str.strip().str.casefold().isin(ex)].reset_index(drop=True)
        M = fb.get_nutrition_matrix(df)
        targets = fb.load_targets(dom)
        lookup = {str(n).strip().casefold(): i for i, n in enumerate(df["ingredient"])}
        for cat in sorted(egbo):
            c = core[cat]
            vb_key = [k for k in c if k.startswith("Vanilla BO")][0]
            rs_key = [k for k in c if k.startswith("Random Search")][0]
            norm = float(np.mean(-np.asarray(c[rs_key])[:, 0]))
            m = sem[cat]["methods"]
            curves = {
                "llm_unres": loss(m["Unrestricted iterative LLM"]).mean(0) / norm,
                "llm_fixed": loss(m["Restricted iterative LLM"]).mean(0) / norm,
                "bo_vanilla": (-np.asarray(c[vb_key])).mean(0) / norm,
                "bo_warm": loss(m["Warm-start full-space BO"]).mean(0) / norm,
                "egbo": loss(egbo[cat]["methods"]["EGBO (LLM-configured BO)"]).mean(0) / norm,
            }
            sel = [lookup[s.strip().casefold()] for s in egbo[cat]["config"]["shared_support"]]
            tgt, mask = targets[cat]
            _, o_sup = fb.nnls_simplex_oracle(M, tgt, mask, selected_indices=sel)
            _, o_full = fb.nnls_simplex_oracle(M, tgt, mask)
            init = egbo[cat]["config"]["shared_initial_loss"] / norm
            rows.append(dict(domain=dom, category=cat, norm=norm, init=init,
                             oracle_support=o_sup / norm, oracle_full=o_full / norm,
                             curves={k: v.tolist() for k, v in curves.items()}))
    return rows


def build_table(rows):
    print(f"{'category':38s} {'init':>5s} {'supGap':>6s} {'initGap':>7s} | "
          f"{'x_unres':>7s} {'x_fixed':>7s} | {'E20':>5s} {'Lu20':>5s} {'Lf20':>5s} {'BOv20':>5s} {'BOw20':>5s}")
    tab = []
    for r in rows:
        cv = {k: np.asarray(v) for k, v in r["curves"].items()}
        sup_gap = r["oracle_support"] - r["oracle_full"]          # structural knowledge (0 = perfect support)
        init_gap = r["init"] - r["oracle_support"]                 # quantitative knowledge within support
        xu, xf = crossover(cv["egbo"], cv["llm_unres"]), crossover(cv["egbo"], cv["llm_fixed"])
        best_llm = np.minimum(cv["llm_unres"], cv["llm_fixed"])
        best_bo = np.minimum(cv["bo_vanilla"], cv["bo_warm"])
        t = dict(cat=r["category"], dom=r["domain"], sup_gap=sup_gap, init_gap=init_gap, init=r["init"],
                 x_unres=xu, x_fixed=xf,
                 **{f"{k}{e}": cv[k][e - 1] for k in cv for e in (5, 10, 20)},
                 adv_llm10=best_llm[9] - cv["egbo"][9], adv_llm20=best_llm[19] - cv["egbo"][19],
                 adv_bo10=best_bo[9] - cv["egbo"][9], adv_bo20=best_bo[19] - cv["egbo"][19],
                 winner10=min({"LLM": best_llm[9], "BO": best_bo[9], "EGBO": cv["egbo"][9]}.items(), key=lambda kv: kv[1])[0],
                 winner20=min({"LLM": best_llm[19], "BO": best_bo[19], "EGBO": cv["egbo"][19]}.items(), key=lambda kv: kv[1])[0],
                 winner5=min({"LLM": best_llm[4], "BO": best_bo[4], "EGBO": cv["egbo"][4]}.items(), key=lambda kv: kv[1])[0])
        tab.append(t)
        print(f"{r['domain'][0]}:{r['category'][:36]:36s} {r['init']:5.2f} {sup_gap:6.3f} {init_gap:7.3f} | "
              f"{str(xu):>7s} {str(xf):>7s} | {cv['egbo'][19]:5.2f} {cv['llm_unres'][19]:5.2f} {cv['llm_fixed'][19]:5.2f} "
              f"{cv['bo_vanilla'][19]:5.2f} {cv['bo_warm'][19]:5.2f}")
    return tab


def summarize(tab):
    n = len(tab)
    print("\n== Winners (best of each family) ==")
    for e in (5, 10, 20):
        w = [t[f"winner{e}"] for t in tab]
        print(f"eval {e:2d}: " + ", ".join(f"{k}={w.count(k)}" for k in ["LLM", "BO", "EGBO"]))

    print("\n== Mean normalized loss across 24 categories ==")
    for k in ["llm_unres", "llm_fixed", "bo_vanilla", "bo_warm", "egbo"]:
        print(f"{k:11s} " + "  ".join(f"e{e}={np.mean([t[f'{k}{e}'] for t in tab]):.3f}" for e in (5, 10, 20)))

    print("\n== Head-to-head: categories where EGBO mean loss < LLM arm mean loss ==")
    for k, name in [("llm_fixed", "fixed-support LLM"), ("llm_unres", "unrestricted LLM")]:
        print(f"EGBO beats {name:18s} " + "  ".join(
            f"eval {e}: {sum(t[f'egbo{e}'] < t[f'{k}{e}'] for t in tab)}/{n}" for e in (5, 10, 20)))

    xs = [t["x_unres"] for t in tab]
    print("\n== Crossover eval where EGBO stays <= unrestricted LLM ==", sorted(x for x in xs if x), "never:", xs.count(None))
    xs = [t["x_fixed"] for t in tab]
    print("== Crossover eval where EGBO stays <= fixed-support LLM ==", sorted(x for x in xs if x), "never:", xs.count(None))

    print(f"\n== Spearman correlations (n={n}) ==")
    for a in ["sup_gap", "init_gap", "init"]:
        for b in ["adv_llm10", "adv_llm20", "adv_bo10", "adv_bo20"]:
            rho, p = spearmanr([t[a] for t in tab], [t[b] for t in tab])
            print(f"{a:9s} vs {b:10s} rho={rho:+.2f} p={p:.3f}")
    rho, p = spearmanr([t["init_gap"] for t in tab], [t["x_unres"] if t["x_unres"] else 21 for t in tab])
    print(f"init_gap vs crossover(unres) rho={rho:+.2f} p={p:.3f}")
    rho, p = spearmanr([t["sup_gap"] for t in tab], [t["x_unres"] if t["x_unres"] else 21 for t in tab])
    print(f"sup_gap  vs crossover(unres) rho={rho:+.2f} p={p:.3f}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot(t, rows, D):
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.size": 9.5,
                                "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    C = {"LLM": "#4c3f91", "BO": "#7f7f7f", "EGBO": "#2ecc71"}
    INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"

    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.9), gridspec_kw={"width_ratios": [1.15, 1, 1]})

    # (a) which family is best at each budget
    evals = np.arange(1, 21)
    wins = {k: [] for k in C}
    for e in evals:
        for k in C:
            wins[k].append(0)
        for r in rows:
            cv = {k: np.asarray(v) for k, v in r["curves"].items()}
            cand = {"LLM": min(cv["llm_unres"][e - 1], cv["llm_fixed"][e - 1]),
                    "BO": min(cv["bo_vanilla"][e - 1], cv["bo_warm"][e - 1]),
                    "EGBO": cv["egbo"][e - 1]}
            if e == 1:
                continue  # all informed arms share evaluation 1
            wins[min(cand, key=cand.get)][-1] += 1
    styles = {"LLM": ("--", "s"), "BO": (":", "^"), "EGBO": ("-", "o")}
    lab = {"LLM": "LLM only (best iterative LLM)", "BO": "BO only (best full-space BO)",
           "EGBO": "EGBO (LLM picks space, BO refines)"}
    for k in C:
        ax[0].plot(evals[1:], wins[k][1:], styles[k][0], marker=styles[k][1], ms=4, lw=2, color=C[k], label=lab[k])
    ax[0].set_xlabel("Evaluation budget")
    ax[0].set_ylabel("Categories where family is best (of 24)")
    ax[0].set_title("(a) Best strategy by evaluation budget", loc="left", fontsize=10, color=INK)
    ax[0].set_xticks([2, 5, 10, 15, 20]); ax[0].set_ylim(0, 24)
    ax[0].grid(axis="y", color=GRID, lw=0.8)
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")

    def scatter(a, xkey, ykey, xlabel, ylabel, title, annotate, corner):
        x = np.array([r[xkey] for r in t]); y = np.array([r[ykey] for r in t])
        for dom, m in [("meat", "o"), ("dairy", "D")]:
            s = np.array([r["dom"] == dom for r in t])
            a.scatter(x[s], y[s], marker=m, s=34, color=C["EGBO"], edgecolor="white", lw=1.2, zorder=3,
                      label=dom)
        a.axhline(0, color=MUTED, lw=0.9)
        b = np.polyfit(x, y, 1); xx = np.linspace(x.min(), x.max(), 50)
        a.plot(xx, np.polyval(b, xx), color=INK, lw=1, alpha=0.6)
        rho, p = spearmanr(x, y)
        xa, ha = (0.02, "left") if corner == "upper left" else (0.98, "right")
        a.text(xa, 0.80, f"Spearman ρ = {rho:+.2f}\np = {p:.3f}, n = 24", transform=a.transAxes, ha=ha, va="top",
               fontsize=8, color=MUTED)
        for r in t:
            if r["cat"] in annotate:
                a.annotate(r["cat"], (r[xkey], r[ykey]), xytext=(4, 4), textcoords="offset points", fontsize=7.5,
                           color=MUTED)
        a.set_xlabel(xlabel); a.set_ylabel(ylabel)
        a.set_title(title, loc="left", fontsize=10, color=INK)
        a.grid(color=GRID, lw=0.8); a.legend(frameon=False, fontsize=8, loc=corner)

    scatter(ax[1], "init_gap", "adv_llm20",
            "LLM's concentration error\n(first recipe − best achievable on its ingredients)",
            "EGBO advantage over best LLM at eval 20",
            "(b) EGBO vs. direct LLM optimization",
            {"coffee creamer", "sour cream", "butter", "deli slices (turkey)"}, "upper left")
    scatter(ax[2], "sup_gap", "adv_bo20",
            "LLM's ingredient-choice error\n(best on its ingredients − best on all)",
            "EGBO advantage over best BO at eval 20",
            "(c) EGBO vs. full-space BO",
            {"sour cream", "bacon", "meatballs", "butter", "coffee creamer"}, "upper right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(D / f"formulatebench_regimes.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {D / 'formulatebench_regimes.pdf'} and .png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", type=Path, default=HERE / "results",
                    help="FormulateBench results root (default: formulateBench/results)")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Output directory (default: <results-dir>/regime_analysis_v1)")
    ap.add_argument("--no-plot", action="store_true", help="Skip the figure")
    args = ap.parse_args()
    R = args.results_dir
    out = args.output_dir or (R / "regime_analysis_v1")

    rows = build_rows(R)
    out.mkdir(parents=True, exist_ok=True)
    (out / "regime_rows.json").write_text(json.dumps(rows, indent=1))
    tab = build_table(rows)
    (out / "regime_table.json").write_text(json.dumps(tab, indent=1, default=float))
    summarize(tab)
    if not args.no_plot:
        plot(json.loads((out / "regime_table.json").read_text()), rows, out)


if __name__ == "__main__":
    main()
