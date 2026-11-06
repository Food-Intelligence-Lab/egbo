#!/usr/bin/env python3
"""Revised Figure 2: original BO baselines plus semantic-only LLM arms.

The algorithmic baseline traces are the archived 10-seed paper runs. The
semantic-only EGBO and LLM traces come from the matched-initialization revision
suite. All curves are renormalized with the archived per-category random-search
distance at evaluation 1, and categories (rather than individual seeds) are the
uncertainty unit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_formulatebench as fb


BASELINE_ORDER = [
    "Random Search",
    "Random subset + BO ($k$=9)",
    "Vanilla BO",
    "TuRBO",
    "SAASBO",
    "SEBO",
    "REMBO ($k$=9)",
]

STYLES = {
    "Random Search": ("#e9c349", ":", 1.8, 3),
    "Random subset + BO ($k$=9)": ("#7d6caf", "--", 1.9, 4),
    "Vanilla BO": ("#7f7f7f", "--", 1.9, 5),
    "TuRBO": ("#2c70b6", "-.", 1.9, 6),
    "SAASBO": ("#e76f51", "-.", 1.9, 7),
    "SEBO": ("#d268a4", "-.", 1.9, 8),
    "REMBO ($k$=9)": ("#2bb3c0", "-", 1.9, 9),
    r"EGBO (default BO; $k\leq9$)": ("#66bfa3", "--", 2.5, 12),
    r"EGBO (LLM-configured BO; $k\leq9$)": ("#2ecc71", "-", 3.2, 13),
    r"Fixed-support iterative LLM ($k\leq9$)": ("#8c564b", (0, (5, 1, 1, 1)), 2.3, 12),
    "Unrestricted iterative LLM": ("#4c3f91", "--", 2.4, 11),
}


def _canonical_legacy(label: str):
    if "Random Subset" in label:
        return "Random subset + BO ($k$=9)"
    if "Random Search" in label:
        return "Random Search"
    if "Vanilla" in label:
        return "Vanilla BO"
    if "TuRBO" in label:
        return "TuRBO"
    if "SAASBO" in label:
        return "SAASBO"
    if "SEBO" in label:
        return "SEBO"
    if "REMBO" in label:
        return "REMBO ($k$=9)"
    return None


def _load_legacy(results_root: Path, names, domain):
    methods = {}
    oracles = {"full": {}, "support": {}}
    d = None
    categories = set()
    for name in names:
        path = results_root / name / domain / "raw.json"
        payload = json.loads(path.read_text())
        if d is None:
            d = int(payload["config"]["n_ingredients"])
        for category, result in payload["results"].items():
            categories.add(category)
            if "oracle_full" in result:
                oracles["full"][category] = float(result["oracle_full"])
            for label, traces in result["traces"].items():
                method = _canonical_legacy(label)
                if method is None or not traces:
                    continue
                arr = np.asarray(traces, dtype=float)
                previous = methods.setdefault(method, {}).get(category)
                if previous is None or len(arr) >= len(previous):
                    methods[method][category] = arr
    return methods, oracles, d, sorted(categories)


def _load_semantic(path: Path):
    result = {}
    for raw_path in path.glob("*/raw.json"):
        payload = json.loads(raw_path.read_text())
        if payload.get("config", {}).get("status") == "complete":
            result[payload["config"]["category"]] = payload
    return result


def _normalizers(legacy):
    return {
        category: float(np.mean(-traces[:, 0]))
        for category, traces in legacy["Random Search"].items()
    }


def _legacy_category_curves(legacy, method, categories, norms):
    return np.asarray([
        -legacy[method][category].mean(axis=0) / norms[category]
        for category in categories
    ])


def _semantic_category_curves(semantic, method, categories, norms):
    rows = []
    for category in categories:
        records = semantic[category]["methods"][method]
        arr = -np.asarray([r["best_values"] for r in records], dtype=float)
        rows.append(arr.mean(axis=0) / norms[category])
    return np.asarray(rows)


def _semantic_oracles(domain, semantic, categories):
    exclusions = {
        "meat": ["egg white", "egg whites", "fat", "milk", "milk proteins"],
        "dairy": ["mozzarella"],
    }
    df = fb.load_ingredients(domain)
    excluded = {x.casefold() for x in exclusions[domain]}
    df = df[
        ~df["ingredient"].astype(str).str.strip().str.casefold().isin(excluded)
    ].reset_index(drop=True)
    matrix = fb.get_nutrition_matrix(df)
    targets = fb.load_targets(domain)
    lookup = {
        str(name).strip().casefold(): i
        for i, name in enumerate(df["ingredient"])
    }
    support_values = {}
    full_values = {}
    for category in categories:
        selected = [
            lookup[name.strip().casefold()]
            for name in semantic[category]["config"]["shared_support"]
        ]
        target, mask = targets[category]
        _, support_distance = fb.nnls_simplex_oracle(
            matrix, target, mask, selected_indices=selected,
        )
        _, full_distance = fb.nnls_simplex_oracle(matrix, target, mask)
        support_values[category] = float(support_distance)
        full_values[category] = float(full_distance)
    return support_values, full_values, len(df)


def _semantic_support_oracles(domain, semantic, categories):
    """Backward-compatible helper used by older plotting utilities."""
    support, _, _ = _semantic_oracles(domain, semantic, categories)
    return support


def _plot_curve(ax, x, curves, label):
    mean = curves.mean(axis=0)
    half = 1.96 * curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
    color, linestyle, linewidth, zorder = STYLES[label]
    line, = ax.plot(
        x, mean, color=color, linestyle=linestyle, linewidth=linewidth,
        label=label, zorder=zorder,
    )
    ax.fill_between(
        x, mean - half, mean + half, color=color, alpha=0.11,
        linewidth=0, zorder=zorder - 1,
    )
    return line


def _panel(
    ax, domain, legacy, legacy_oracles, semantic, default_egbo, native_egbo,
    egbo_method_label, d, categories,
):
    norms = _normalizers(legacy)
    x = np.arange(1, 21)
    handles = []
    for method in BASELINE_ORDER:
        curves = _legacy_category_curves(legacy, method, categories, norms)
        handles.append(_plot_curve(ax, x, curves, method))

    semantic_methods = [
        (
            native_egbo,
            "EGBO (default BO)",
            r"EGBO (default BO; $k\leq9$)",
        ),
        (
            default_egbo,
            egbo_method_label,
            r"EGBO (LLM-configured BO; $k\leq9$)",
        ),
        (
            semantic,
            "Restricted iterative LLM",
            r"Fixed-support iterative LLM ($k\leq9$)",
        ),
        (
            semantic,
            "Unrestricted iterative LLM",
            "Unrestricted iterative LLM",
        ),
    ]
    for source, raw_method, display in semantic_methods:
        curves = _semantic_category_curves(
            source, raw_method, categories, norms,
        )
        handles.append(_plot_curve(ax, x, curves, display))

    support_oracles, full_oracles, semantic_d = _semantic_oracles(
        domain, semantic, categories,
    )
    support_floor = np.mean([
        support_oracles[c] / norms[c] for c in categories
    ])
    full_floor = np.mean([
        full_oracles[c] / norms[c] for c in categories
    ])
    support_line = ax.axhline(
        support_floor, color="#2c3e50", linestyle=(0, (1, 1)),
        linewidth=1.4, label="Oracle (LLM-selected ingredients)", zorder=2,
    )
    full_line = ax.axhline(
        full_floor, color="black", linestyle=(0, (3, 3)),
        linewidth=1.4, label="Oracle (all allowed ingredients)", zorder=2,
    )
    handles.extend([support_line, full_line])
    ax.axvline(10, color="0.42", linestyle=":", linewidth=1.0)
    ax.set(
        xlabel="Evaluation",
        title=f"Plant-Based {domain.title()} ({len(categories)} categories)",
        xlim=(1, 20), ylim=(-0.02, 1.42),
    )
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(alpha=0.22)
    return handles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--legacy-names", nargs="+",
        default=["release_v1_10seed", "release_v1_10seed_saasbo", "release_v1_10seed_sebo"],
    )
    parser.add_argument("--meat-semantic", type=Path, required=True)
    parser.add_argument("--dairy-semantic", type=Path, required=True)
    parser.add_argument("--meat-default-egbo", type=Path, required=True)
    parser.add_argument("--dairy-default-egbo", type=Path, required=True)
    parser.add_argument("--meat-native-egbo", type=Path, required=True)
    parser.add_argument("--dairy-native-egbo", type=Path, required=True)
    parser.add_argument(
        "--egbo-method-label",
        default="EGBO (native BoTorch defaults)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "stixsans", "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.2), sharey=True)
    all_handles = None
    for ax, domain, semantic_path, default_egbo_path, native_egbo_path in [
        (
            axes[0], "meat", args.meat_semantic,
            args.meat_default_egbo, args.meat_native_egbo,
        ),
        (
            axes[1], "dairy", args.dairy_semantic,
            args.dairy_default_egbo, args.dairy_native_egbo,
        ),
    ]:
        legacy, oracles, d, categories = _load_legacy(
            args.results_root, args.legacy_names, domain,
        )
        semantic = _load_semantic(semantic_path)
        default_egbo = _load_semantic(default_egbo_path)
        native_egbo = _load_semantic(native_egbo_path)
        missing = sorted(set(categories) - set(semantic))
        if missing:
            raise ValueError(f"semantic suite missing {domain} categories: {missing}")
        missing_egbo = sorted(set(categories) - set(default_egbo))
        if missing_egbo:
            raise ValueError(
                f"default EGBO suite missing {domain} categories: {missing_egbo}"
            )
        missing_native = sorted(set(categories) - set(native_egbo))
        if missing_native:
            raise ValueError(
                f"native EGBO suite missing {domain} categories: {missing_native}"
            )
        handles = _panel(
            ax, domain, legacy, oracles, semantic, default_egbo, native_egbo,
            args.egbo_method_label, d, categories,
        )
        if all_handles is None:
            all_handles = handles
    axes[0].set_ylabel("Normalized Distance (Lower Is Better)")
    labels = [h.get_label() for h in all_handles]
    fig.legend(
        all_handles, labels, loc="lower center", ncol=4, fontsize=9,
        frameon=True, bbox_to_anchor=(0.5, -0.005),
        columnspacing=1.35, handlelength=3.0,
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.92, bottom=0.27, wspace=0.08)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
