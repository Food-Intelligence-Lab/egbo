#!/usr/bin/env python3
"""Appendix F.3: matched semantic-only one-shot vs adaptive EGBO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_revised_figure2 import _load_legacy, _normalizers


def _load_curves(
    domain, baseline_root, adaptive_root, results_root, legacy_names,
    one_shot_method_label,
):
    legacy, _, _, categories = _load_legacy(results_root, legacy_names, domain)
    norms = _normalizers(legacy)
    one_shot, adaptive = [], []
    rows = []
    for category in categories:
        slug = category.replace(" ", "_")
        baseline = json.loads((baseline_root / slug / "raw.json").read_text())
        adapted = json.loads((adaptive_root / slug / "raw.json").read_text())
        one = -np.asarray([
            record["best_values"]
            for record in baseline["methods"][one_shot_method_label]
        ], dtype=float).mean(axis=0) / norms[category]
        ada_records = adapted["methods"]["Adaptive semantic-only EGBO (Q=2)"]
        ada = -np.asarray([
            record["best_values"] for record in ada_records
        ], dtype=float).mean(axis=0) / norms[category]
        additions = [
            sum(len(item.get("additions", [])) for item in record["selection_history"])
            for record in ada_records
        ]
        one_shot.append(one)
        adaptive.append(ada)
        rows.append({
            "category": category,
            "one_shot_eval10": float(one[9]),
            "adaptive_eval10": float(ada[9]),
            "one_shot_eval20": float(one[19]),
            "adaptive_eval20": float(ada[19]),
            "adaptive_minus_one_shot_eval20": float(ada[19] - one[19]),
            "mean_ingredients_added": float(np.mean(additions)),
        })
    return np.asarray(one_shot), np.asarray(adaptive), rows


def _plot_curve(ax, x, curves, label, color, linestyle, linewidth):
    mean = curves.mean(axis=0)
    half = 1.96 * curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
    ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
    ax.fill_between(x, mean - half, mean + half, color=color, alpha=0.14, linewidth=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--meat-one-shot", type=Path, required=True)
    parser.add_argument("--dairy-one-shot", type=Path, required=True)
    parser.add_argument("--meat-adaptive", type=Path, required=True)
    parser.add_argument("--dairy-adaptive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--one-shot-method-label",
        default="One-shot EGBO",
        help="Method key in the one-shot raw.json files.",
    )
    parser.add_argument(
        "--legacy-names", nargs="+",
        default=["release_v1_10seed", "release_v1_10seed_saasbo", "release_v1_10seed_sebo"],
    )
    args = parser.parse_args()

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "stixsans", "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    summaries = {}
    for ax, domain, one_root, adaptive_root in [
        (axes[0], "meat", args.meat_one_shot, args.meat_adaptive),
        (axes[1], "dairy", args.dairy_one_shot, args.dairy_adaptive),
    ]:
        one, ada, rows = _load_curves(
            domain, one_root, adaptive_root, args.results_root, args.legacy_names,
            args.one_shot_method_label,
        )
        x = np.arange(1, 21)
        _plot_curve(ax, x, one, "One-shot EGBO ($Q=1$)", "#2ecc71", "-", 3.0)
        _plot_curve(ax, x, ada, "Adaptive EGBO ($Q=2$)", "#756bb1", "--", 2.5)
        ax.axvline(10, color="0.35", linestyle=":", linewidth=1.1)
        ax.text(10.25, 0.04, "expert expansion query", transform=ax.get_xaxis_transform(),
                fontsize=8.5, color="0.30", va="bottom", rotation=90)
        ax.set(
            title=f"Plant-Based {domain.title()} ({len(rows)} categories)",
            xlabel="Evaluation", xlim=(1, 20),
        )
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, loc="upper right")
        diff = ada[:, 19] - one[:, 19]
        summaries[domain] = {
            "n_categories": len(rows),
            "one_shot_eval10": float(one[:, 9].mean()),
            "adaptive_eval10": float(ada[:, 9].mean()),
            "one_shot_eval20": float(one[:, 19].mean()),
            "adaptive_eval20": float(ada[:, 19].mean()),
            "mean_adaptive_minus_one_shot_eval20": float(diff.mean()),
            "ci95_halfwidth_category_delta": float(
                1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
            ),
            "adaptive_category_wins": int(np.sum(diff < 0)),
            "one_shot_category_wins": int(np.sum(diff > 0)),
            "mean_ingredients_added": float(np.mean([
                row["mean_ingredients_added"] for row in rows
            ])),
            "per_category": rows,
        }
    axes[0].set_ylabel("Normalized Distance (Lower Is Better)")
    fig.suptitle("Adaptive EGBO Ablation", fontsize=14, y=0.995)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.15, top=0.86, wspace=0.08)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    args.output.with_suffix(".json").write_text(json.dumps(summaries, indent=2) + "\n")
    plt.close(fig)
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_suffix(".json"))


if __name__ == "__main__":
    main()
