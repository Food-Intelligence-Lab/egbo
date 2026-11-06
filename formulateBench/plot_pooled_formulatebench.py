#!/usr/bin/env python3
"""Optional pooled 24-category FormulateBench convergence figure."""

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
    _legacy_category_curves,
    _load_legacy,
    _load_semantic,
    _normalizers,
    _plot_curve,
    _semantic_category_curves,
    _semantic_oracles,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--legacy-names", nargs="+",
        default=[
            "release_v1_10seed", "release_v1_10seed_saasbo",
            "release_v1_10seed_sebo",
        ],
    )
    parser.add_argument("--meat-semantic", type=Path, required=True)
    parser.add_argument("--dairy-semantic", type=Path, required=True)
    parser.add_argument("--meat-egbo", type=Path, required=True)
    parser.add_argument("--dairy-egbo", type=Path, required=True)
    parser.add_argument("--meat-native-egbo", type=Path, required=True)
    parser.add_argument("--dairy-native-egbo", type=Path, required=True)
    parser.add_argument("--egbo-method-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "stixsans", "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    legacy_curves = {method: [] for method in BASELINE_ORDER}
    semantic_curves = {
        r"EGBO (default BO; $k\leq9$)": [],
        r"EGBO (LLM-configured BO; $k\leq9$)": [],
        r"Fixed-support iterative LLM ($k\leq9$)": [],
        "Unrestricted iterative LLM": [],
    }
    support_floors, full_floors = [], []

    for domain, semantic_path, egbo_path, native_egbo_path in [
        (
            "meat", args.meat_semantic, args.meat_egbo,
            args.meat_native_egbo,
        ),
        (
            "dairy", args.dairy_semantic, args.dairy_egbo,
            args.dairy_native_egbo,
        ),
    ]:
        legacy, _, _, categories = _load_legacy(
            args.results_root, args.legacy_names, domain,
        )
        semantic = _load_semantic(semantic_path)
        egbo = _load_semantic(egbo_path)
        native_egbo = _load_semantic(native_egbo_path)
        norms = _normalizers(legacy)
        for method in BASELINE_ORDER:
            legacy_curves[method].append(
                _legacy_category_curves(legacy, method, categories, norms)
            )
        semantic_curves[r"EGBO (LLM-configured BO; $k\leq9$)"].append(
            _semantic_category_curves(
                egbo, args.egbo_method_label, categories, norms,
            )
        )
        semantic_curves[r"EGBO (default BO; $k\leq9$)"].append(
            _semantic_category_curves(
                native_egbo, "EGBO (default BO)",
                categories, norms,
            )
        )
        semantic_curves[r"Fixed-support iterative LLM ($k\leq9$)"].append(
            _semantic_category_curves(
                semantic, "Restricted iterative LLM", categories, norms,
            )
        )
        semantic_curves["Unrestricted iterative LLM"].append(
            _semantic_category_curves(
                semantic, "Unrestricted iterative LLM", categories, norms,
            )
        )
        support, full, _ = _semantic_oracles(domain, semantic, categories)
        support_floors.extend(support[c] / norms[c] for c in categories)
        full_floors.extend(full[c] / norms[c] for c in categories)

    fig, ax = plt.subplots(figsize=(8.9, 5.9))
    x = np.arange(1, 21)
    handles = []
    for method in BASELINE_ORDER:
        handles.append(
            _plot_curve(ax, x, np.concatenate(legacy_curves[method]), method)
        )
    for method in [
        r"EGBO (default BO; $k\leq9$)",
        r"EGBO (LLM-configured BO; $k\leq9$)",
        r"Fixed-support iterative LLM ($k\leq9$)",
        "Unrestricted iterative LLM",
    ]:
        handles.append(
            _plot_curve(ax, x, np.concatenate(semantic_curves[method]), method)
        )
    handles.append(ax.axhline(
        np.mean(support_floors), color="#2c3e50", linestyle=(0, (1, 1)),
        linewidth=1.4, label="Oracle (LLM-selected ingredients)",
    ))
    handles.append(ax.axhline(
        np.mean(full_floors), color="black", linestyle=(0, (3, 3)),
        linewidth=1.4, label="Oracle (all allowed ingredients)",
    ))
    ax.axvline(10, color="0.42", linestyle=":", linewidth=1.0)
    ax.set(
        xlabel="Evaluation", ylabel="Normalized Distance (Lower Is Better)",
        title="FormulateBench (24 categories)", xlim=(1, 20),
        ylim=(-0.02, 1.28),
    )
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(alpha=0.22)
    labels = [handle.get_label() for handle in handles]
    fig.legend(
        handles, labels, loc="lower center", ncol=3, fontsize=8.5,
        frameon=True, bbox_to_anchor=(0.5, -0.01),
        columnspacing=1.25, handlelength=3.0,
    )
    fig.subplots_adjust(left=0.11, right=0.985, top=0.91, bottom=0.31)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
