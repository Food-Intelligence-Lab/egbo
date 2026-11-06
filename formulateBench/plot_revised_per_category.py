#!/usr/bin/env python3
"""Revised per-category FormulateBench figures with LLM-only baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_revised_figure2 import (
    BASELINE_ORDER,
    STYLES,
    _load_legacy,
    _load_semantic,
    _semantic_oracles,
)


SEMANTIC_METHODS = [
    ("Restricted iterative LLM", r"Fixed-support iterative LLM ($k\leq9$)"),
    ("Unrestricted iterative LLM", "Unrestricted iterative LLM"),
]


def _curve(records):
    return -np.asarray([record["best_values"] for record in records], dtype=float)


def _legacy_curve(legacy, method, category):
    return -np.asarray(legacy[method][category], dtype=float)


def _plot_method(ax, x, arr, label):
    color, linestyle, linewidth, zorder = STYLES[label]
    mean = arr.mean(axis=0)
    line, = ax.plot(
        x, mean, color=color, linestyle=linestyle,
        linewidth=linewidth, label=label, zorder=zorder,
    )
    if len(arr) > 1:
        half = 1.96 * arr.std(axis=0, ddof=1) / np.sqrt(len(arr))
        ax.fill_between(
            x, mean - half, mean + half, color=color,
            alpha=0.11, linewidth=0, zorder=zorder - 1,
        )
    return line


def plot_domain(
    domain, results_root, legacy_names, semantic_path, default_egbo_path,
    native_egbo_path, egbo_method_label, output,
):
    legacy, legacy_oracles, d, categories = _load_legacy(
        results_root, legacy_names, domain,
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
    support_oracles, full_oracles, semantic_d = _semantic_oracles(
        domain, semantic, categories,
    )

    ncols = 5
    nrows = int(np.ceil(len(categories) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(15.4, 3.15 * nrows + 1.55), squeeze=False,
    )
    x = np.arange(1, 21)
    legend_handles = None

    for index, category in enumerate(categories):
        row, col = divmod(index, ncols)
        ax = axes[row, col]
        handles = []
        upper_candidates = []
        for method in BASELINE_ORDER:
            arr = _legacy_curve(legacy, method, category)
            handles.append(_plot_method(ax, x, arr, method))
            upper_candidates.append(float(arr.mean(axis=0).max()))
        egbo_arr = _curve(
            default_egbo[category]["methods"][egbo_method_label]
        )
        handles.append(
            _plot_method(
                ax, x, egbo_arr,
                r"EGBO (LLM-configured BO; $k\leq9$)",
            )
        )
        upper_candidates.append(float(egbo_arr.mean(axis=0).max()))
        native_arr = _curve(
            native_egbo[category]["methods"]["EGBO (default BO)"]
        )
        handles.append(
            _plot_method(
                ax, x, native_arr, r"EGBO (default BO; $k\leq9$)",
            )
        )
        upper_candidates.append(float(native_arr.mean(axis=0).max()))
        for raw_method, display in SEMANTIC_METHODS:
            arr = _curve(semantic[category]["methods"][raw_method])
            handles.append(_plot_method(ax, x, arr, display))
            upper_candidates.append(float(arr.mean(axis=0).max()))

        support_line = ax.axhline(
            support_oracles[category], color="#2c3e50",
            linestyle=(0, (1, 1)), linewidth=1.25,
            label="Oracle (LLM-selected ingredients)", zorder=2,
        )
        full_line = ax.axhline(
            full_oracles[category], color="black",
            linestyle=(0, (3, 3)), linewidth=1.25,
            label="Oracle (all allowed ingredients)", zorder=2,
        )
        handles.extend([support_line, full_line])
        if legend_handles is None:
            legend_handles = handles

        ax.axvline(10, color="0.48", linestyle=":", linewidth=0.8)
        ax.set_title(category.title(), fontsize=9.5, pad=5)
        ax.set_xlim(1, 20)
        ax.set_xticks([5, 10, 15, 20])
        top = max(upper_candidates) * 1.09
        ax.set_ylim(-0.015, top)
        ax.grid(alpha=0.20)
        ax.tick_params(labelsize=7.5)
        if col == 0:
            ax.set_ylabel("Distance", fontsize=9)
        if row == nrows - 1:
            ax.set_xlabel("Evaluation", fontsize=9)

    for index in range(len(categories), nrows * ncols):
        axes.flat[index].axis("off")

    fig.suptitle(
        f"Plant-Based {domain.title()}: Per-Category Convergence",
        fontsize=14, y=0.985,
    )
    labels = [handle.get_label() for handle in legend_handles]
    fig.legend(
        legend_handles, labels, loc="lower center", ncol=4,
        fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, 0.015),
        columnspacing=1.4, handlelength=3.0,
    )
    fig.subplots_adjust(
        left=0.055, right=0.995, top=0.92,
        bottom=0.18 if nrows == 2 else 0.145,
        hspace=0.42, wspace=0.24,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output.with_suffix(".png"))
    print(output.with_suffix(".pdf"))


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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "stixsans", "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    plot_domain(
        "meat", args.results_root, args.legacy_names,
        args.meat_semantic, args.meat_default_egbo,
        args.meat_native_egbo,
        args.egbo_method_label,
        args.output_dir / "formulatebench_meat_per_category",
    )
    plot_domain(
        "dairy", args.results_root, args.legacy_names,
        args.dairy_semantic, args.dairy_default_egbo,
        args.dairy_native_egbo,
        args.egbo_method_label,
        args.output_dir / "formulatebench_dairy_per_category",
    )


if __name__ == "__main__":
    main()
