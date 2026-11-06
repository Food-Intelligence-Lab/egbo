#!/usr/bin/env python3
"""Rescore the observation-noise sweep with each method's OWN recommendation.

run_noise_sweep.py scores every method by the true loss of the recipe with the
best *observed* (noisy) score. Here each method instead picks its recommendation
the way it naturally would, at evaluations t in {5, 10, 20}:

  * EGBO: refit the paper's GP (Matern-5/2 ARD, learned noise, Standardize) on
    the first t noisy observations, in the same normalized-logit input space BO
    uses (logits reconstructed from the logged fractions as log x - max log x,
    clipped to [-12, 0]), and recommend the evaluated recipe with the best
    posterior mean.
  * Iterative LLM: show the model (the same model as in the sweep) its first t
    evaluated formulations and their (noisy) scores in the same semantic-only
    format, and ask which formulation it recommends as final. The LLM is not
    told the evaluator is noisy, matching the sweep.

Each pick is scored by its TRUE (noiseless) loss, normalized as in the paper.
LLM calls require OPENROUTER_API_KEY and are cached under
results/noise_sweep_v1/llm_cache_ownpick. Use --no-llm to rescore EGBO only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
R = HERE / "results"
SWEEP = R / "noise_sweep_v1"
TS = (5, 10, 20)


def _gp_pick(path: str):
    import torch
    torch.set_num_threads(1)
    import egbo_bo_config as configured
    import run_formulatebench as fb
    from botorch.utils.transforms import normalize
    d = json.loads(Path(path).read_text())
    names = sorted({n for r in d["recipes"] for n in r})
    X = np.array([[r.get(n, 0.0) for n in names] for r in d["recipes"]], float)
    Z = np.log(np.clip(X, np.exp(-12.0), 1.0))
    Z = np.clip(Z - Z.max(axis=1, keepdims=True), -12.0, 0.0)
    bounds = torch.tensor([[-12.0] * len(names), [0.0] * len(names)], dtype=torch.double)
    y = -np.asarray(d["observed_loss"], float)  # maximize, as in BO
    out = {}
    for t in TS:
        torch.manual_seed(0)
        Zt = normalize(torch.tensor(Z[:t], dtype=torch.double), bounds)
        Yt = torch.tensor(y[:t], dtype=torch.double).unsqueeze(-1)
        gp, mll = configured.build_gp(Zt, Yt, configured.BOConfig())
        fb._fit_gpytorch_mll_with_warning(mll, "own-pick GP")
        with torch.no_grad():
            mu = gp.posterior(Zt).mean.squeeze(-1).numpy()
        i = int(np.argmax(mu))
        out[t] = d["true_loss"][i] / d["norm"]
    return path, out


_PROMPT = """You are optimizing the overall similarity of a plant-based formulation to animal-based {category}. The evaluator is a black box.

Previous evaluated formulations follow. Target dissimilarity is the evaluator's scalar output; lower is better.
{history}

The experimental budget is now spent. Which ONE of the formulations above do you recommend as the final formulation? Return only JSON in this exact shape:
{{"iteration": 3}}
Do not include commentary."""


def _llm_pick(args):
    path, model, cache = args
    import run_formulatebench as fb
    d = json.loads(Path(path).read_text())
    client = fb.OpenRouterClient(os.environ["OPENROUTER_API_KEY"], model, cache_dir=cache)
    out, fallbacks = {}, 0
    for t in TS:
        hist = [{"iteration": i + 1,
                 "recipe": [{"ingredient": n, "fraction": f} for n, f in d["recipes"][i].items()],
                 "target_dissimilarity": d["observed_loss"][i]} for i in range(t)]
        prompt = _PROMPT.format(category=d["category"], history=json.dumps(hist, separators=(",", ":")))
        k = None
        for _ in range(2):
            raw = client.chat("You are an expert plant-based food scientist.", prompt, temperature=0.0)
            m = re.search(r'"iteration"\s*:\s*(\d+)', raw or "")
            if m and 1 <= int(m.group(1)) <= t:
                k = int(m.group(1))
                break
        if k is None:  # fall back to best observed and count it
            fallbacks += 1
            k = int(np.argmin(d["observed_loss"][:t])) + 1
        out[t] = d["true_loss"][k - 1] / d["norm"]
    return path, out, fallbacks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--model", default="anthropic/claude-opus-4.7")
    a = ap.parse_args()
    files = sorted(str(p) for dom in ("meat", "dairy") for p in (SWEEP / dom).glob("*/*.json"))
    out_path = SWEEP / "own_pick.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {"egbo": {}, "llm": {}, "llm_fallbacks": 0}
    rel = lambda f: os.path.relpath(f, SWEEP)  # keys are stored relative to the sweep folder
    eg = [f for f in files if Path(f).name.startswith("egbo_") and rel(f) not in res["egbo"]]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for path, out in ex.map(_gp_pick, eg, chunksize=4):
            res["egbo"][rel(path)] = out
    out_path.write_text(json.dumps(res))
    print(f"EGBO GP-mean picks: {len(res['egbo'])} runs", flush=True)
    if not a.no_llm:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise SystemExit("Set OPENROUTER_API_KEY (or use --no-llm)")
        llm = [f for f in files if Path(f).name.startswith("llm_") and rel(f) not in res["llm"]]
        cache = str(SWEEP / "llm_cache_ownpick")
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for path, out, fb_ in ex.map(_llm_pick, [(f, a.model, cache) for f in llm]):
                res["llm"][rel(path)] = out
                res["llm_fallbacks"] += fb_
                out_path.write_text(json.dumps(res))
        print(f"LLM own picks: {len(res['llm'])} runs, {res['llm_fallbacks']} parse fallbacks", flush=True)
    summarize(res)


def summarize(res):
    from scipy.stats import wilcoxon
    # per (level, arm, category) mean over seeds/draws
    acc = defaultdict(list)
    base = defaultdict(list)
    for kind in ("egbo", "llm"):
        for path, out in res[kind].items():
            d = json.loads((SWEEP / path).read_text())
            key = (d["level"], d["arm"], d["domain"], d["category"])
            acc[key].append([out[str(t)] if str(t) in out else out[t] for t in TS])
            rec = np.asarray(d["recommended_true_loss"]) / d["norm"]
            base[key].append([rec[t - 1] for t in TS])
    levels = sorted({k[0] for k in acc})
    summary = {}
    print("\nTrue loss of each method's OWN pick (vs. best-observed rule), mean over 24 categories")
    for lvl in levels:
        cats = sorted({(k[2], k[3]) for k in acc if k[0] == lvl})
        row = {}
        for arm in ("llm_unres", "llm_fixed", "egbo"):
            own = np.array([np.mean(acc[(lvl, arm, *c)], 0) for c in cats if (lvl, arm, *c) in acc])
            bo = np.array([np.mean(base[(lvl, arm, *c)], 0) for c in cats if (lvl, arm, *c) in base])
            row[arm] = dict(own=own.mean(0).round(3).tolist(), best_observed=bo.mean(0).round(3).tolist(), n=len(own))
            print(f"  noise {lvl:<5g} {arm:10s} own pick e5/e10/e20 = {row[arm]['own']}   (best-observed: {row[arm]['best_observed']})")
        ok = [c for c in cats if all((lvl, a, *c) in acc for a in ("llm_unres", "llm_fixed", "egbo"))]
        for j, t in enumerate(TS):
            llm = np.array([min(np.mean(acc[(lvl, 'llm_unres', *c)], 0)[j], np.mean(acc[(lvl, 'llm_fixed', *c)], 0)[j]) for c in ok])
            e = np.array([np.mean(acc[(lvl, 'egbo', *c)], 0)[j] for c in ok])
            p = wilcoxon(llm - e).pvalue if np.any(llm != e) else float("nan")
            row[f"egbo_beats_best_llm_e{t}"] = int((e < llm).sum())
            print(f"     eval {t:2d}: EGBO (own pick) beats best LLM (own pick) in {(e < llm).sum()}/{len(ok)}; "
                  f"mean adv {np.mean(llm - e):+.3f}; Wilcoxon p={p:.2g}")
        summary[str(lvl)] = row
    (SWEEP / "own_pick_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
