#!/usr/bin/env python3
"""Cross-model replication of the FormulateBench strategy comparison.

Claude Opus 4.7 rows come from the paper's cached runs (results/regime_analysis_v1/
regime_table.json); other models from run_multi_llm.py (results/multi_llm_v1/).
Losses are normalized by the category's mean random-search loss at evaluation 1.
Per model, reports: mean loss of each arm at evaluations 5/10/20; categories in
which EGBO beats the better iterative-LLM arm (Wilcoxon signed-rank p); the
quality of the model's first recipe (concentration and ingredient-choice
errors, as in Appendix F.4); and the Spearman correlations of F.4.
No API calls.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent
R = HERE / "results"
E = (5, 10, 20)


def norms():
    out = {}
    for dom in ("meat", "dairy"):
        core = json.loads((R / "final_protocol_core_v1" / dom / "summary.json").read_text())
        for cat, c in core.items():
            rs = [k for k in c if k.startswith("Random Search")][0]
            vb = [k for k in c if k.startswith("Vanilla BO")][0]
            out[(dom, cat)] = (float(np.mean(-np.asarray(c[rs])[:, 0])),
                               (-np.asarray(c[vb])).mean(0))
    return out


def load_new(root: Path, N):
    models = {}
    for mdir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "llm_cache"):
        rows = []
        for cdir in sorted(mdir.glob("*/*")):
            meta_p = cdir / "meta.json"
            eg = sorted(cdir.glob("egbo_seed*.json"))
            if not (meta_p.exists() and (cdir / "llm_unres.json").exists()
                    and (cdir / "llm_fixed.json").exists() and len(eg) >= 10):
                continue
            meta = json.loads(meta_p.read_text())
            norm, vb = N[(meta["domain"], meta["category"])]
            cur = lambda tl: np.minimum.accumulate(np.asarray(tl, float)) / norm
            un = cur(json.loads((cdir / "llm_unres.json").read_text())["true_loss"])
            fx = cur(json.loads((cdir / "llm_fixed.json").read_text())["true_loss"])
            e = np.mean([cur(json.loads(p.read_text())["true_loss"]) for p in eg], 0)
            r = dict(cat=meta["category"], dom=meta["domain"],
                     init=meta["initial_loss"] / norm,
                     init_gap=(meta["initial_loss"] - meta["oracle_support"]) / norm,
                     sup_gap=(meta["oracle_support"] - meta["oracle_full"]) / norm)
            es = sorted(cdir.glob("egbo_selfcfg_seed*.json"))
            e_self = (np.mean([cur(json.loads(p.read_text())["true_loss"]) for p in es], 0)
                      if len(es) >= 10 else None)
            for t in E:
                r[f"llm_unres{t}"], r[f"llm_fixed{t}"], r[f"egbo{t}"] = un[t - 1], fx[t - 1], e[t - 1]
                if e_self is not None:
                    r[f"egbo_self{t}"] = e_self[t - 1]
                r[f"bo_vanilla{t}"] = vb[t - 1] / norm
            models[mdir.name.replace("__", "/")] = models.get(mdir.name.replace("__", "/"), []) + [r]
    return models


def report(name, rows):
    n = len(rows)
    out = {"n": n}
    g = lambda k: np.array([r[k] for r in rows])
    print(f"\n== {name}  ({n} categories) ==")
    for arm in ("llm_unres", "llm_fixed", "egbo"):
        out[arm] = [round(float(g(f'{arm}{t}').mean()), 3) for t in E]
        print(f"  {arm:10s} mean loss e5/e10/e20 = {out[arm]}")
    for t in E:
        best_llm = np.minimum(g(f"llm_unres{t}"), g(f"llm_fixed{t}"))
        adv = best_llm - g(f"egbo{t}")
        p = wilcoxon(adv).pvalue if np.any(adv != 0) else float("nan")
        out[f"wins_e{t}"] = int((adv > 0).sum())
        out[f"adv_e{t}"] = round(float(adv.mean()), 3)
        out[f"p_e{t}"] = float(p)
        print(f"  eval {t:2d}: EGBO beats best LLM arm in {(adv > 0).sum()}/{n}; mean adv {adv.mean():+.3f}; Wilcoxon p={p:.2g}")
    if all("egbo_self20" in r for r in rows):
        out["egbo_self"] = [round(float(g(f'egbo_self{t}').mean()), 3) for t in E]
        print(f"  egbo (model's own BO config) mean loss e5/e10/e20 = {out['egbo_self']}")
        for t in E:
            adv = np.minimum(g(f"llm_unres{t}"), g(f"llm_fixed{t}")) - g(f"egbo_self{t}")
            p = wilcoxon(adv).pvalue if np.any(adv != 0) else float("nan")
            out[f"self_wins_e{t}"] = int((adv > 0).sum())
            print(f"    eval {t:2d}: own-config EGBO beats best LLM arm in {(adv > 0).sum()}/{n}; mean adv {adv.mean():+.3f}; p={p:.2g}")
    out["first_recipe"] = round(float(g("init").mean()), 3)
    out["conc_err"] = round(float(g("init_gap").mean()), 3)
    out["ingr_err"] = round(float(g("sup_gap").mean()), 3)
    print(f"  first recipe {out['first_recipe']:.3f}; concentration error {out['conc_err']:.3f}; ingredient-choice error {out['ingr_err']:.3f}")
    adv20 = np.minimum(g("llm_unres20"), g("llm_fixed20")) - g("egbo20")
    advbo = g("bo_vanilla20") - g("egbo20")
    r1, p1 = spearmanr(g("init_gap"), adv20)
    r2, p2 = spearmanr(g("sup_gap"), advbo)
    out.update(rho_conc=round(float(r1), 2), p_conc=float(p1), rho_ingr=round(float(r2), 2), p_ingr=float(p2))
    print(f"  Spearman conc. error vs EGBO-LLM adv (e20): rho={r1:+.2f} p={p1:.3f}")
    print(f"  Spearman ingr. error vs EGBO-vanillaBO adv (e20): rho={r2:+.2f} p={p2:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi-dir", type=Path, default=R / "multi_llm_v1")
    a = ap.parse_args()
    N = norms()
    opus = json.loads((R / "regime_analysis_v1" / "regime_table.json").read_text())
    for r in opus:
        r["dom"], r["cat"] = r["dom"], r["cat"]
    allm = {"anthropic/claude-opus-4.7 (paper)": opus}
    allm.update(load_new(a.multi_dir, N))
    summary = {m: report(m, rows) for m, rows in allm.items()}
    pooled = [r for rows in allm.values() for r in rows]
    print("\n== pooled across models ==")
    summary["pooled"] = report("pooled", pooled)
    (a.multi_dir / "multi_llm_summary.json").write_text(json.dumps(summary, indent=1))
    print("\nsaved", a.multi_dir / "multi_llm_summary.json")


if __name__ == "__main__":
    main()
