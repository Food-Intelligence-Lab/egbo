#!/usr/bin/env python3
"""Frozen semantic-only evaluation suite for FormulateBench targets."""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "Helvetica Neue"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial"]
matplotlib.rcParams["mathtext.fontset"] = "stixsans"

import egbo_bo_config as configured
import run_formulatebench as fb


# Fixed BO configuration used for the BO arms of the semantic suite. This is
# the configuration recorded as ``"bo_config"`` in every run of the released
# results (results/meat_semantic_all14_v1, results/dairy_semantic_all10_v3:
# "matern52/long/default/qlognei/isotropic"). Inlined here because the
# original helper module (run_researcher_tuned_bo.py) is not part of the
# release. Note it differs from egbo_bo_config.DEFAULT_CONFIG_SPEC
# (long lengthscale prior, isotropic kernel); it is kept so that rerunning
# this script reproduces the released results.
RESEARCHER_CONFIG_SPEC = {
    "kernel": "matern52",
    "lengthscale_prior": "long",
    "noise": "default",
    "acquisition": "qlognei",
    "ard": "isotropic",
}


def _atomic_json(path, obj):
    """Write ``obj`` as JSON to ``path`` atomically (temp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as handle:
        json.dump(obj, handle, indent=2)
    os.replace(tmp, path)


DEFAULT_CATEGORIES = {
    "dairy": [
        "barista milk", "butter", "cheddar", "coffee creamer", "cream cheese",
        "ice cream", "milk", "mozzarella", "sour cream", "yogurt",
    ],
    "meat": [
        "bacon", "bratwurst", "breaded chicken fillets",
        "breakfast sausage patties", "burgers", "chicken nuggets",
        "deli slices (ham)", "deli slices (turkey)", "hot dogs", "meatballs",
        "pulled pork", "steak fillets", "unbreaded chicken fillet",
        "unbreaded chicken strips and chunks",
    ],
}

DEFAULT_EXCLUSIONS = {
    "dairy": ["mozzarella"],
    "meat": ["egg white", "egg whites", "fat", "milk", "milk proteins"],
}

ANALYSIS_METHODS = [
    "Unrestricted iterative LLM",
    "One-shot LLM recipe",
    "Random Search",
    "Standard full-space BO",
    "Warm-start full-space BO",
    "One-shot EGBO",
    "Restricted iterative LLM",
]

DISPLAY_LABELS = {
    "Restricted iterative LLM": "Fixed-support iterative LLM",
    "One-shot EGBO": "EGBO",
}


def _initial_from_unrestricted(record, df):
    recipe = record["recipe_history"][0]["recipe"]
    lookup = {str(name).strip().casefold(): i for i, name in enumerate(df["ingredient"])}
    x = np.zeros(len(df), dtype=float)
    selected = []
    for item in recipe:
        idx = lookup[str(item["ingredient"]).strip().casefold()]
        if idx not in selected:
            selected.append(idx)
        x[idx] += float(item["fraction"])
    x /= x.sum()
    return recipe, x, selected


def _plot_category(payload, out_path):
    styles = {
        "Random Search": ("#e9c349", ":", True),
        "Standard full-space BO": ("#d62728", "-", True),
        "Warm-start full-space BO": ("#9467bd", "-", True),
        "One-shot LLM recipe": ("#7f7f7f", ":", False),
        "One-shot EGBO": ("#2ca02c", "-", True),
        "Restricted iterative LLM": ("#1f77b4", "--", False),
        "Unrestricted iterative LLM": ("#8e44ad", "--", False),
    }
    budget = payload["config"]["budget"]
    x = np.arange(1, budget + 1)
    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    for label, records in payload["methods"].items():
        if not records or label not in styles:
            continue
        arr = -np.asarray([r["best_values"] for r in records], dtype=float)
        color, ls, band = styles[label]
        mean = arr.mean(axis=0)
        ax.plot(x, mean, color=color, linestyle=ls, linewidth=2.1, label=label)
        if band and len(arr) > 1:
            half = 1.96 * arr.std(axis=0) / np.sqrt(len(arr))
            ax.fill_between(x, mean - half, mean + half, color=color, alpha=0.10)
    ax.axvline(10, color="0.55", linewidth=1.0, linestyle=":")
    ax.set(xlabel="Objective evaluations",
           ylabel="Best-so-far nutrition z-RMSD (lower is better)",
           title=f"{payload['config']['category'].title()}: semantic-only optimization",
           xlim=(1, budget))
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _run_category(category, args, df, matrix, targets, client, cfg):
    category_dir = args.output_dir / category.replace(" ", "_")
    raw_path = category_dir / "raw.json"
    if raw_path.exists():
        payload = json.loads(raw_path.read_text())
    else:
        payload = {
            "config": {
                "category": category, "domain": args.domain,
                "n_ingredients": len(df), "budget": args.budget,
                "primary_evaluation": 10, "bo_seeds": args.bo_seeds,
                "n_init": args.n_init, "local_sigma": args.local_sigma,
                "adaptive_checkpoints": [5], "adaptive_max_additions": 1,
                "bo_config": cfg.label(), "llm_model": args.llm_model,
                "llm_temperature": 0.0,
                "information_regime": "target product + ingredient names + scalar feedback",
                "excluded_ingredients": list(args.exclude_ingredients),
                "status": "running",
            },
            "methods": {},
        }
        _atomic_json(raw_path, payload)

    target, mask = targets[category]
    objective = fb.make_objective(matrix, target, mask)
    methods = payload["methods"]
    payload["config"]["domain"] = args.domain
    payload["config"]["n_ingredients"] = len(df)
    # Adaptive EGBO was exploratory and is not part of the final comparison.
    methods.pop("Adaptive EGBO (Q=2)", None)
    payload["config"].pop("adaptive_checkpoints", None)
    payload["config"].pop("adaptive_max_additions", None)

    # This trajectory supplies the common first recipe for every informed arm.
    if not methods.get("Unrestricted iterative LLM"):
        print(f"[{category}] unrestricted iterative LLM", flush=True)
        record = fb.iterative_llm_optimizer(
            objective, df, category, target, mask, matrix,
            budget=args.budget, llm_client=client, max_ingredients=9,
            temperature=0.0, information_regime="semantic-only",
        )
        methods["Unrestricted iterative LLM"] = [record]
        _atomic_json(raw_path, payload)
    unrestricted = methods["Unrestricted iterative LLM"][0]
    initial_recipe, initial_x, selected = _initial_from_unrestricted(unrestricted, df)
    initial_y = float(objective(initial_x))
    payload["config"]["shared_initial_loss"] = -initial_y
    payload["config"]["shared_support"] = [str(df["ingredient"].iloc[i]) for i in selected]

    if not methods.get("One-shot LLM recipe"):
        methods["One-shot LLM recipe"] = [{
            "best_values": [initial_y] * args.budget,
            "all_y": [initial_y], "final_best": initial_y,
            "recipe_history": [{"iteration": 1, "recipe": initial_recipe, "loss": -initial_y}],
        }]
        _atomic_json(raw_path, payload)

    random_records = methods.setdefault("Random Search", [])
    for seed in range(len(random_records), args.bo_seeds):
        print(f"[{category}] Random Search seed {seed + 1}/{args.bo_seeds}", flush=True)
        random_records.append(fb.random_search(
            objective, len(df), budget=args.budget, seed=seed,
        ))
        _atomic_json(raw_path, payload)

    bo_arms = [
        ("Standard full-space BO", list(range(len(df))), None, "global"),
        ("Warm-start full-space BO", list(range(len(df))), initial_x, "local"),
        ("One-shot EGBO", selected, initial_x, "local"),
    ]
    for label, indices, x0, design in bo_arms:
        records = methods.setdefault(label, [])
        for seed in range(len(records), args.bo_seeds):
            print(f"[{category}] {label} seed {seed + 1}/{args.bo_seeds}", flush=True)
            records.append(configured.reduced_bo_cfg(
                objective=objective, d_full=len(df), selected_indices=indices,
                budget=args.budget, n_init=args.n_init, seed=seed, cfg=cfg,
                fit_mll=fb._fit_gpytorch_mll_with_warning,
                softmax=fb._softmax, init_logits=fb._simplex_initial_logits,
                initial_x=x0, initial_design=design, local_sigma=args.local_sigma,
            ))
            _atomic_json(raw_path, payload)

    if not methods.get("Restricted iterative LLM"):
        print(f"[{category}] restricted iterative LLM", flush=True)
        restricted_df = df.iloc[selected].reset_index(drop=True)
        restricted_matrix = matrix[selected]
        restricted_x0 = initial_x[selected]

        def restricted_objective(x_small):
            x_full = np.zeros(len(df), dtype=float)
            x_full[selected] = x_small
            return objective(x_full)

        record = fb.iterative_llm_optimizer(
            restricted_objective, restricted_df, category, target, mask,
            restricted_matrix, budget=args.budget, llm_client=client,
            max_ingredients=len(selected), temperature=0.0,
            information_regime="semantic-only", initial_x=restricted_x0,
            initial_recipe=initial_recipe,
        )
        methods["Restricted iterative LLM"] = [record]
        _atomic_json(raw_path, payload)

    payload["config"]["status"] = "complete"
    _atomic_json(raw_path, payload)
    _plot_category(payload, category_dir / "comparison.png")
    return payload


def _write_summary(results, output_dir):
    rows = []
    for category, payload in results.items():
        for label in ANALYSIS_METHODS:
            records = payload["methods"].get(label, [])
            if not records:
                continue
            arr = -np.asarray([r["best_values"] for r in records], dtype=float)
            rows.append({
                "category": category, "method": label, "n_runs": len(records),
                "eval10_mean": float(arr[:, 9].mean()),
                "eval10_sd": float(arr[:, 9].std()),
                "eval20_mean": float(arr[:, -1].mean()),
                "eval20_sd": float(arr[:, -1].std()),
                "trajectory_1_10_mean": float(arr[:, :10].mean(axis=1).mean()),
            })
    _atomic_json(output_dir / "summary.json", rows)
    methods = [
        method for method in ANALYSIS_METHODS
        if all(results[c]["methods"].get(method) for c in results)
    ]
    x = np.arange(len(results))
    width = 0.11
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for j, method in enumerate(methods):
        values = []
        errors = []
        for payload in results.values():
            arr = -np.asarray([r["best_values"] for r in payload["methods"][method]])
            values.append(arr[:, 9].mean())
            # BO arms have 10 seeds; the deterministic LLM arms currently have
            # one trajectory and therefore no empirical uncertainty estimate.
            errors.append(arr[:, 9].std() if len(arr) > 1 else np.nan)
        ax.bar(
            x + (j - (len(methods)-1)/2)*width,
            values,
            width,
            yerr=errors,
            capsize=2.5,
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
            label=DISPLAY_LABELS.get(method, method),
        )
    ax.set_xticks(x, [c.title() for c in results])
    ax.set_ylabel("Best-so-far z-RMSD at evaluation 10 (lower is better)")
    ax.set_title("Dairy categories: primary endpoint")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "evaluation10_summary.png", dpi=220)
    plt.close(fig)
    _plot_aggregate(results, output_dir)


def _plot_aggregate(results, output_dir):
    """Paper-style trajectories with categories as the uncertainty unit."""
    styles = {
        "Unrestricted iterative LLM": ("#756bb1", "--", 2.2, "Unrestricted iterative LLM"),
        "One-shot LLM recipe": ("#e6ab02", ":", 2.0, "One-shot LLM recipe"),
        "Random Search": ("#e9c349", ":", 1.8, "Random Search"),
        "Standard full-space BO": ("#7f7f7f", "--", 2.0, "Standard full-space BO"),
        "Warm-start full-space BO": ("#2ca9b7", "-", 2.0, "Warm-start full-space BO"),
        "One-shot EGBO": ("#2ecc71", "-", 3.0, "EGBO"),
        "Restricted iterative LLM": ("#e76f51", "-.", 2.2, "Fixed-support iterative LLM"),
    }
    budget = next(iter(results.values()))["config"]["budget"]
    x = np.arange(1, budget + 1)
    fig, ax = plt.subplots(figsize=(7.8, 5.3))
    normalizers = {}
    for category, payload in results.items():
        random_records = payload["methods"]["Random Search"]
        random_arr = -np.asarray(
            [r["best_values"] for r in random_records], dtype=float,
        )
        normalizers[category] = float(random_arr[:, 0].mean())
    for method in ANALYSIS_METHODS:
        category_trajectories = []
        for category, payload in results.items():
            records = payload["methods"].get(method, [])
            if not records:
                break
            arr = -np.asarray([r["best_values"] for r in records], dtype=float)
            category_trajectories.append(
                arr.mean(axis=0) / normalizers[category]
            )
        if len(category_trajectories) != len(results):
            continue
        arr = np.asarray(category_trajectories)
        mean = arr.mean(axis=0)
        half = 1.96 * arr.std(axis=0, ddof=1) / np.sqrt(len(arr))
        color, linestyle, linewidth, display = styles[method]
        ax.plot(
            x, mean, color=color, linestyle=linestyle,
            linewidth=linewidth, label=display,
        )
        ax.fill_between(x, mean - half, mean + half, color=color, alpha=0.12)
    ax.axvline(10, color="0.45", linewidth=1.0, linestyle=":")
    ax.set(
        xlabel="Evaluation",
        ylabel="Normalized Distance (Lower Is Better)",
        title=(
            f"Plant-Based {next(iter(results.values()))['config']['domain'].title()} "
            f"($d$={next(iter(results.values()))['config']['n_ingredients']}, "
            f"{len(results)} categories)"
        ),
        xlim=(1, budget),
    )
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "aggregate_trajectories.png", dpi=300)
    fig.savefig(output_dir / "aggregate_trajectories.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["meat", "dairy"], default="dairy")
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--bo-seeds", type=int, default=10)
    # Count the shared LLM recipe as the first design point, followed by four
    # local perturbations, matching the n_init=5 conventional-BO protocol.
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--local-sigma", type=float, default=0.35)
    parser.add_argument("--llm-model", default="anthropic/claude-opus-4.7")
    parser.add_argument("--llm-cache-dir", default="semantic_suite_cache")
    parser.add_argument(
        "--exclude-ingredients", nargs="*", default=None,
        help="Candidate names excluded before every method. Domain-specific "
             "animal-product leakage rows are removed by default.",
    )
    parser.add_argument("--openrouter-key", default=None)
    args = parser.parse_args()
    if args.categories is None:
        args.categories = list(DEFAULT_CATEGORIES[args.domain])
    if args.exclude_ingredients is None:
        args.exclude_ingredients = list(DEFAULT_EXCLUSIONS[args.domain])
    key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY or pass --openrouter-key")
    df = fb.load_ingredients(args.domain)
    excluded = {name.strip().casefold() for name in args.exclude_ingredients}
    df = df[
        ~df["ingredient"].astype(str).str.strip().str.casefold().isin(excluded)
    ].reset_index(drop=True)
    matrix = fb.get_nutrition_matrix(df)
    targets = fb.load_targets(args.domain)
    missing = [c for c in args.categories if c not in targets]
    if missing:
        raise SystemExit(f"Unknown {args.domain} categories: {missing}")
    client = fb.OpenRouterClient(key, args.llm_model, cache_dir=args.llm_cache_dir)
    cfg = configured.BOConfig.from_spec(RESEARCHER_CONFIG_SPEC)
    results = {}
    for category in args.categories:
        results[category] = _run_category(category, args, df, matrix, targets, client, cfg)
    _write_summary(results, args.output_dir)
    print(f"Completed suite: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
