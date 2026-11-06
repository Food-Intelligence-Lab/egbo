#!/usr/bin/env python3
"""EGBO with each model's OWN BO configuration (cross-model replication).

run_multi_llm.py runs EGBO with the paper's fixed configuration, which Claude
Opus 4.7 selected. This script elicits nothing new: it reads the configuration
each model chose with query_matched_bo_config.py (saved as
results/multi_llm_v1/<model_slug>/bo_config.json) and reruns only EGBO on that
model's shared first recipe and ingredients, writing egbo_selfcfg_seed<k>.json
next to the existing files. No API calls.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_multi_llm import _atomic, _setup  # noqa: E402


def run_cell(args):
    cdir, spec, seeds = args
    import torch
    torch.set_num_threads(1)
    import egbo_bo_config as configured
    meta = json.loads((Path(cdir) / "meta.json").read_text())
    fb, df, M, target, mask, objective = _setup(meta["domain"], meta["category"])
    lookup = {str(n).strip().casefold(): i for i, n in enumerate(df["ingredient"])}
    x0 = np.zeros(len(df))
    sel = []
    for item in meta["shared_recipe"]:
        i = lookup[str(item["ingredient"]).strip().casefold()]
        if i not in sel:
            sel.append(i)
        x0[i] += float(item["fraction"])
    x0 /= x0.sum()
    cfg = configured.BOConfig.from_spec(spec, context="self-config")
    for s in range(seeds):
        p = Path(cdir) / f"egbo_selfcfg_seed{s}.json"
        if p.exists():
            continue
        rec = configured.reduced_bo_cfg(
            objective=objective, d_full=len(df), selected_indices=sel, budget=20, n_init=5,
            seed=s, cfg=cfg, fit_mll=fb._fit_gpytorch_mll_with_warning, softmax=fb._softmax,
            init_logits=fb._simplex_initial_logits, initial_x=x0, initial_design="local",
            local_sigma=0.35)
        rec["true_loss"] = [-y for y in rec["all_y"]]
        rec["bo_config_label"] = cfg.label()
        _atomic(p, rec)
    return cdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=HERE / "results" / "multi_llm_v1")
    ap.add_argument("--models", nargs="+", required=True, help="model slugs, e.g. openai__gpt-6.1-sol")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    tasks = []
    for slug in a.models:
        spec = json.loads((a.root / slug / "bo_config.json").read_text())["validated_config"]
        for cdir in sorted((a.root / slug).glob("*/*")):
            if (cdir / "meta.json").exists():
                tasks.append((str(cdir), spec, a.seeds))
    print(f"{len(tasks)} cells", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(run_cell, t) for t in tasks]), 1):
            print(f"[{i}/{len(tasks)}] {f.result().split('multi_llm_v1/')[-1]}", flush=True)


if __name__ == "__main__":
    main()
