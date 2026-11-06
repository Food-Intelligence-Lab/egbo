#!/usr/bin/env python3
"""Run cached semantic EGBO supports with a selected BO configuration.

The runner reuses the cached LLM recipe, support, local initialization rule,
budget, and seeds.  Without ``--bo-config-json`` it uses BoTorch's native
``SingleTaskGP`` and qLogNEI; with a config file it uses the validated
configuration recorded there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import egbo_bo_config as configured
import run_formulatebench as fb
from run_semantic_llm_suite import (
    DEFAULT_CATEGORIES,
    DEFAULT_EXCLUSIONS,
    _atomic_json,
    _initial_from_unrestricted,
)


def _slug(category: str) -> str:
    return category.replace(" ", "_")


def _run_category(category, args, df, matrix, targets, cfg):
    source_path = args.source_dir / _slug(category) / "raw.json"
    source = json.loads(source_path.read_text())
    initial_recipe, initial_x, selected = _initial_from_unrestricted(
        source["methods"]["Unrestricted iterative LLM"][0], df,
    )

    target, mask = targets[category]
    objective = fb.make_objective(matrix, target, mask)
    out_path = args.output_dir / _slug(category) / "raw.json"
    if out_path.exists():
        payload = json.loads(out_path.read_text())
    else:
        payload = {
            "config": {
                "category": category,
                "domain": args.domain,
                "n_ingredients": len(df),
                "budget": args.budget,
                "bo_seeds": args.bo_seeds,
                "n_init": args.n_init,
                "local_sigma": args.local_sigma,
                "bo_config": (
                    cfg.label() if cfg is not None else
                    "BoTorch 0.18.1 native SingleTaskGP defaults + qLogNEI"
                ),
                "bo_config_source": (
                    str(args.bo_config_json.resolve())
                    if args.bo_config_json is not None else "native BoTorch"
                ),
                "initialization": (
                    "cached LLM recipe + local logit-normal perturbations"
                ),
                "excluded_ingredients": list(args.exclude_ingredients),
                "shared_initial_loss": -float(objective(initial_x)),
                "shared_initial_recipe": initial_recipe,
                "shared_support": [
                    str(df["ingredient"].iloc[i]) for i in selected
                ],
                "source_raw": str(source_path.resolve()),
                "status": "running",
            },
            "methods": {args.method_label: []},
        }
        _atomic_json(out_path, payload)

    payload["config"]["bo_seeds"] = args.bo_seeds
    records = payload["methods"][args.method_label]
    for seed in range(len(records), args.bo_seeds):
        print(
            f"[{args.domain}/{category}] {args.method_label} seed "
            f"{seed + 1}/{args.bo_seeds}",
            flush=True,
        )
        common = {
            "objective": objective,
            "d_full": len(df),
            "selected_indices": selected,
            "budget": args.budget,
            "n_init": args.n_init,
            "seed": seed,
            "initial_x": initial_x,
            "initial_design": "local",
            "local_sigma": args.local_sigma,
        }
        if cfg is None:
            record = fb.reduced_bo(**common)
        else:
            record = configured.reduced_bo_cfg(
                **common,
                cfg=cfg,
                fit_mll=fb._fit_gpytorch_mll_with_warning,
                softmax=fb._softmax,
                init_logits=fb._simplex_initial_logits,
            )
        records.append(record)
        _atomic_json(out_path, payload)

    payload["config"]["status"] = "complete"
    _atomic_json(out_path, payload)
    return payload


def _write_summary(results, output_dir, method_label):
    rows = []
    for category, payload in results.items():
        records = payload["methods"][method_label]
        arr = -np.asarray([record["best_values"] for record in records])
        eval10_idx = min(9, arr.shape[1] - 1)
        eval20_idx = arr.shape[1] - 1
        rows.append({
            "category": category,
            "n_runs": len(records),
            "eval10_mean": float(arr[:, eval10_idx].mean()),
            "eval20_mean": float(arr[:, eval20_idx].mean()),
            "eval10_sd": float(arr[:, eval10_idx].std()),
            "eval20_sd": float(arr[:, eval20_idx].std()),
        })
    _atomic_json(output_dir / "summary.json", {"results": rows})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=["meat", "dairy"])
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--bo-seeds", type=int, default=10)
    # One cached LLM recipe plus four local perturbations.
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--local-sigma", type=float, default=0.35)
    parser.add_argument("--exclude-ingredients", nargs="*", default=None)
    parser.add_argument("--bo-config-json", type=Path, default=None)
    parser.add_argument(
        "--method-label", default="EGBO (native BoTorch defaults)",
    )
    args = parser.parse_args()

    if args.categories is None:
        args.categories = list(DEFAULT_CATEGORIES[args.domain])
    if args.exclude_ingredients is None:
        args.exclude_ingredients = list(DEFAULT_EXCLUSIONS[args.domain])

    df = fb.load_ingredients(args.domain)
    excluded = {name.strip().casefold() for name in args.exclude_ingredients}
    df = df[
        ~df["ingredient"].astype(str).str.strip().str.casefold().isin(excluded)
    ].reset_index(drop=True)
    matrix = fb.get_nutrition_matrix(df)
    targets = fb.load_targets(args.domain)
    cfg = None
    if args.bo_config_json is not None:
        config_payload = json.loads(args.bo_config_json.read_text())
        config_spec = config_payload.get("validated_config", config_payload)
        cfg = configured.BOConfig.from_spec(
            config_spec, context=str(args.bo_config_json),
        )

    results = {}
    for category in args.categories:
        results[category] = _run_category(
            category, args, df, matrix, targets, cfg,
        )
        _write_summary(results, args.output_dir, args.method_label)
    print(f"Completed EGBO suite: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
