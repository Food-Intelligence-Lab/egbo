#!/usr/bin/env python3
"""Matched semantic-only adaptive-EGBO ablation for FormulateBench.

The expert sees ingredient names, evaluated recipes, and scalar target
dissimilarity only.  It may monotonically add up to two ingredients after
evaluation 10 (Q=2).  All other settings match the current one-shot suite.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import egbo_bo_config as configured
import run_formulatebench as fb
from run_adaptive_semantic_yogurt import adaptive_bo
from run_semantic_llm_suite import (
    DEFAULT_CATEGORIES,
    DEFAULT_EXCLUSIONS,
    _atomic_json,
    _initial_from_unrestricted,
)


def category_slug(category: str) -> str:
    return category.replace(" ", "_")


def run_category(category, args, df, matrix, targets, client, cfg):
    source_path = args.source_dir / category_slug(category) / "raw.json"
    if not source_path.exists():
        raise FileNotFoundError(f"missing matched semantic source: {source_path}")
    source = json.loads(source_path.read_text())
    if source.get("config", {}).get("status") != "complete":
        raise ValueError(f"source is not complete: {source_path}")

    initial_recipe, initial_x, selected = _initial_from_unrestricted(
        source["methods"]["Unrestricted iterative LLM"][0], df,
    )
    target, mask = targets[category]
    objective = fb.make_objective(matrix, target, mask)

    out_dir = args.output_dir / category_slug(category)
    out_path = out_dir / "raw.json"
    if out_path.exists():
        result = json.loads(out_path.read_text())
    else:
        result = {
            "config": {
                "category": category,
                "domain": args.domain,
                "n_ingredients": len(df),
                "budget": args.budget,
                "bo_seeds": args.bo_seeds,
                "n_init": args.n_init,
                "local_sigma": args.local_sigma,
                "expert_rounds": 2,
                "expansion_after_evaluation": args.checkpoint,
                "max_additions": args.max_additions,
                "bo_config": cfg.label(),
                "llm_model": args.llm_model,
                "llm_temperature": 0.0,
                "information_regime": (
                    "target product + ingredient names + evaluated recipes "
                    "+ scalar target-dissimilarity feedback"
                ),
                "excluded_ingredients": list(args.exclude_ingredients),
                "shared_initial_loss": -float(objective(initial_x)),
                "shared_initial_recipe": initial_recipe,
                "shared_initial_support": [
                    str(df["ingredient"].iloc[i]) for i in selected
                ],
                "status": "running",
            },
            "methods": {"Adaptive semantic-only EGBO (Q=2)": []},
        }
        _atomic_json(out_path, result)

    records = result["methods"].setdefault(
        "Adaptive semantic-only EGBO (Q=2)", [],
    )
    for seed in range(len(records), args.bo_seeds):
        print(
            f"[{args.domain}/{category}] adaptive seed "
            f"{seed + 1}/{args.bo_seeds}", flush=True,
        )
        record = adaptive_bo(
            objective=objective,
            df=df,
            category=category,
            initial_x=initial_x,
            initial_selected=selected,
            budget=args.budget,
            n_init=args.n_init,
            seed=seed,
            checkpoints={args.checkpoint},
            local_sigma=args.local_sigma,
            cfg=cfg,
            client=client,
            max_additions=args.max_additions,
        )
        records.append(record)
        _atomic_json(out_path, result)

    result["config"]["status"] = "complete"
    _atomic_json(out_path, result)
    return result


def write_summary(results, output_dir):
    rows = []
    for category, payload in results.items():
        records = payload["methods"]["Adaptive semantic-only EGBO (Q=2)"]
        arr = -np.asarray([record["best_values"] for record in records])
        added = [
            sum(len(item.get("additions", [])) for item in record["selection_history"])
            for record in records
        ]
        rows.append({
            "category": category,
            "n_runs": len(records),
            "eval10_mean": float(arr[:, 9].mean()),
            "eval20_mean": float(arr[:, 19].mean()),
            "mean_ingredients_added": float(np.mean(added)),
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
    # One cached LLM recipe plus four local perturbations, as in one-shot EGBO.
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--local-sigma", type=float, default=0.35)
    parser.add_argument("--checkpoint", type=int, default=10)
    parser.add_argument("--max-additions", type=int, default=2)
    parser.add_argument("--llm-model", default="anthropic/claude-opus-4.7")
    parser.add_argument("--llm-cache-dir", required=True)
    parser.add_argument(
        "--bo-config-json", required=True, type=Path,
        help="Frozen global BO configuration elicited from the LLM expert.",
    )
    parser.add_argument("--exclude-ingredients", nargs="*", default=None)
    parser.add_argument("--openrouter-key", default=None)
    args = parser.parse_args()

    if args.categories is None:
        args.categories = list(DEFAULT_CATEGORIES[args.domain])
    if args.exclude_ingredients is None:
        args.exclude_ingredients = list(DEFAULT_EXCLUSIONS[args.domain])
    if not (args.n_init <= args.checkpoint < args.budget):
        raise SystemExit("Require n_init <= checkpoint < budget")

    key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY or pass --openrouter-key")
    client = fb.OpenRouterClient(
        key, args.llm_model, cache_dir=args.llm_cache_dir,
    )
    df = fb.load_ingredients(args.domain)
    excluded = {name.strip().casefold() for name in args.exclude_ingredients}
    df = df[
        ~df["ingredient"].astype(str).str.strip().str.casefold().isin(excluded)
    ].reset_index(drop=True)
    matrix = fb.get_nutrition_matrix(df)
    targets = fb.load_targets(args.domain)
    config_payload = json.loads(args.bo_config_json.read_text())
    config_spec = config_payload.get("validated_config", config_payload)
    cfg = configured.BOConfig.from_spec(
        config_spec, context=str(args.bo_config_json),
    )

    results = {}
    for category in args.categories:
        results[category] = run_category(
            category, args, df, matrix, targets, client, cfg,
        )
        write_summary(results, args.output_dir)
    print(f"Completed adaptive suite: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
