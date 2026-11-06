#!/usr/bin/env python3
"""Main-text FormulateBench figure: batched optimization (10 rounds x 3 recipes,
as in the laboratory deployment), without and with observation noise.

Reads results/<root>/batched_summary.json written by analyze_batched.py (mean
curves and 95% CIs over categories of the true loss of the recommended recipe,
normalized by the category's mean random-search loss at evaluation 1). One LLM
(default: Claude Opus 4.7, the paper's LLM); conventional baselines do not use
the LLM. No API calls.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
# colors / line styles follow the paper's other FormulateBench figures
STYLE = {
    "egbo":       ("EGBO (LLM-selected ingredients + BO)", "#2ecc71", "-", 3.0, 10),
    "llm_unres":  ("Unrestricted iterative LLM", "#4c3f91", "--", 2.3, 9),
    "llm_fixed":  ("Fixed-support iterative LLM", "#8c564b", (0, (5, 1, 1, 1)), 2.1, 8),
    "turbo":      ("TuRBO", "#2c70b6", "-.", 1.8, 6),
    "vanilla_bo": ("Vanilla BO", "#7f7f7f", (0, (4, 2)), 1.8, 5),
    "random":     ("Random search", "#e9c349", ":", 1.9, 4),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=HERE / "results" / "batched_r10_v1")
    ap.add_argument("--model", default="anthropic/claude-opus-4.7")
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--output", type=Path, default=HERE / "results" / "batched_r10_v1" / "formulatebench_batched")
    a = ap.parse_args()
    summ = json.loads((a.root / "batched_summary.json").read_text())
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "stixsans", "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.6), sharey=True)
    titles = ["(a) Noiseless evaluations", rf"(b) Noisy evaluations ($\sigma = {a.noise:g}$)"]
    handles = []
    for ax, lvl, title in zip(axes, (0.0, a.noise), titles):
        row = summ[f"{a.model}|{lvl}"]
        R = row["rounds"]
        x = np.arange(1, R + 1)
        for arm, (label, color, ls, lw, z) in STYLE.items():
            if arm not in row["curves"]:
                continue
            m = np.asarray(row["curves"][arm]); ci = np.asarray(row["curves_ci"][arm])
            ax.fill_between(x, m - ci, m + ci, color=color, alpha=0.16 if arm == "egbo" else 0.05, lw=0, zorder=z - 1)
            h, = ax.plot(x, m, color=color, ls=ls, lw=lw, zorder=z, label=label,
                         marker="o" if arm == "egbo" else None, ms=4)
            if ax is axes[0]:
                handles.append(h)
        ax.set_title(title, fontsize=15, pad=8)
        ax.set_xlabel("Batch (3 recipes per batch)", fontsize=13)
        ax.set_xticks(x)
        ax.set_xlim(x[0] - 0.2, x[-1] + 0.2)
        ax.tick_params(labelsize=12)
        ax.grid(color="#e6e6e6", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Normalized distance (lower is better)", fontsize=13)
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center", ncol=3, fontsize=12,
               frameon=False, bbox_to_anchor=(0.5, -0.01), columnspacing=2.0, handlelength=3.2)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.27, wspace=0.06)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(a.output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print("saved", a.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
