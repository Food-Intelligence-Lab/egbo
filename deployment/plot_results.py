"""Reproduce data-driven deployment figures (paper Figures 12-14 and 25-29)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def sensory_plot(data: pd.DataFrame, fields: list[tuple[str, str]], out: Path) -> None:
    fig, axes = plt.subplots(2, (len(fields) + 1) // 2, figsize=(12, 6), sharex=True)
    axes = axes.ravel()
    for ax, (field, target) in zip(axes, fields):
        for _, group in data.groupby("Batch"):
            pass
        ax.scatter(data["Iteration"], data[field], s=22, alpha=0.65)
        means = data.groupby("Iteration")[field].mean()
        ax.plot(means.index, means.values, linewidth=2)
        ax.axhline(float(data[target].iloc[0]), color="black", linestyle="--", linewidth=1)
        ax.set_title(field.replace("_", " "))
        ax.set_ylabel("Panel score")
        ax.grid(alpha=0.2)
    for ax in axes[-((len(axes) - len(fields)) or len(axes)):]:
        ax.set_xlabel("Iteration")
    save(fig, out)


def utility_plot(data: pd.DataFrame, out: Path, baseline: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(baseline, color="0.5", linestyle=":", label="FSA baseline")
    ax.plot(data["Iteration"], data["Best_Utility_In_Batch"], marker="o", label="Best in batch")
    ax.plot(data["Iteration"], data["Best_So_Far"], marker="o", linestyle="--", label="Best so far")
    ax.set(xlabel="Iteration", ylabel="Utility", xlim=(1, 10), ylim=(0.7, 1.01))
    ax.grid(alpha=0.2)
    ax.legend()
    save(fig, out)


def ingredient_plot(data: pd.DataFrame, out: Path) -> None:
    ingredients = [c for c in data.columns if c not in {"Batch", "Iteration"}]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
    for ax, ingredient in zip(axes.ravel(), ingredients):
        means = data.groupby("Iteration")[ingredient].mean()
        ax.plot(means.index, means.values, marker="o")
        ax.set_title(ingredient.replace("_", " "), fontsize=10)
        ax.grid(alpha=0.2)
    for ax in axes.ravel()[len(ingredients):]:
        ax.axis("off")
    save(fig, out)


FSA_BASELINE = 0.767
HFS_COLOR = "#c2185b"
EGBO_COLOR = "#26a69a"
HFS2_COLOR = "#e67e22"


def hfs_exploration_plot(leg1: pd.DataFrame, out: Path, n_trials: int = 29) -> None:
    """Experiment 1 (Fig. 28): HFS from-scratch exploration, with failed trials shaded."""
    leg1 = leg1.sort_values("trial")
    evaluated = set(leg1["trial"].astype(int))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for trial in range(1, n_trials + 1):
        if trial not in evaluated:
            ax.axvspan(trial - 0.4, trial + 0.4, color=HFS_COLOR, alpha=0.06, lw=0)
    ax.plot(leg1["trial"], leg1["utility"], marker="o", color=HFS_COLOR, lw=1.8,
            label="HFS (evaluated trials)", zorder=3)
    ax.plot(leg1["trial"], leg1["utility"].cummax(), ls="--", color=HFS_COLOR,
            alpha=0.5, lw=1.4, label="HFS (best so far)", zorder=2)
    ax.axhline(FSA_BASELINE, color=EGBO_COLOR, ls=":", lw=1.5,
               label=f"FSA starting recipe ({FSA_BASELINE:.3f})")
    best = leg1.loc[leg1["utility"].idxmax()]
    ax.annotate(f"Best: {best['utility']:.3f}", xy=(best["trial"], best["utility"]),
                xytext=(best["trial"] - 6, best["utility"] + 0.03),
                fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="0.3", lw=0.8))
    ax.text(1, 0.605, "Shaded columns = failed trials (no utility evaluated)",
            fontsize=8, color=HFS_COLOR, alpha=0.6, style="italic")
    ax.set(xlim=(0.4, n_trials + 0.6), ylim=(0.60, 0.90),
           xlabel="Trial", ylabel="Total Utility Score")
    ax.set_xticks(range(1, n_trials + 1))
    ax.set_title(f"Experiment 1: HFS Recipe Exploration ({n_trials} trials, ~40h)",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, out)


def egbo_vs_hfs_plot(egbo: pd.DataFrame, leg2: pd.DataFrame, out: Path) -> None:
    """Experiment 2 (Fig. 29): EGBO vs HFS refining the same FSA starting recipe."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    iters = egbo["Iteration"].astype(int)
    for it in iters:
        if it % 2 == 1:
            ax.axvspan(it - 0.5, it + 0.5, color="#c5d3e8", alpha=0.35, lw=0)
    ax.plot(iters, egbo["Best_Utility_In_Batch"], marker="s", color=EGBO_COLOR, lw=1.8,
            label="EGBO (best in batch of 3)")
    ax.plot(iters, egbo["Best_So_Far"], ls="--", color=EGBO_COLOR, alpha=0.5, lw=1.4,
            label="EGBO (best so far)")
    ax.plot(leg2["Iteration"], leg2["Total_Utility_Score"], marker="D", color=HFS2_COLOR,
            lw=1.8, label="HFS (1 recipe/iter)")
    ax.plot(leg2["Iteration"], leg2["Best_So_Far"], ls="--", color=HFS2_COLOR, alpha=0.5,
            lw=1.4, label="HFS (best so far)")
    ax.set_xticks(list(iters))
    ax.set_xticklabels([f"Iter {i}" for i in iters], rotation=35, ha="right")
    ax.set(xlim=(0.5, len(iters) + 0.5), ylim=(0.72, 1.005),
           xlabel="Iteration", ylabel="Total Utility Score")
    ax.set_title("Experiment 2: EGBO vs. HFS Optimizing the Same Starting Recipe",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "figures")
    args = parser.parse_args()
    read = lambda name: pd.read_csv(args.data_dir / name)

    sensory_plot(read("yogurt_sensory_data.csv"), [
        ("Consistency", "Consistency_target"), ("Creaminess", "Creaminess_target"),
        ("Tangy", "Tangy_target"), ("Thickness_Uniformity", "Thickness_Uniformity_target"),
    ], args.out_dir / "figure15_yogurt_sensory.png")
    utility_plot(read("yogurt_utility_per_iteration.csv"),
                 args.out_dir / "figure16_yogurt_utility.png", 0.767)
    ingredient_plot(read("yogurt_ingredient_data.csv"),
                    args.out_dir / "figure17_yogurt_ingredients.png")
    sensory_plot(read("cottage_sensory_data.csv"), [
        ("Homogeneity", "Homogeneity_target"),
        ("Dispersion Phase", "Dispersion Phase_target"),
        ("Slipperiness", "Slipperiness_target"), ("Sour", "Sour_target"),
        ("Dairy", "Dairy_target"), ("Off-notes", "Off-notes_target"),
    ], args.out_dir / "figure28_cottage_sensory.png")
    utility_plot(read("cottage_utility_per_iteration.csv"),
                 args.out_dir / "figure29_cottage_utility.png", 0.776)

    combined = read("combined_utility_data.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(combined["Iteration"], combined["Yogurt_Best_So_Far"], marker="o", label="Yogurt")
    ax.plot(combined["Iteration"], combined["Cottage_Best_So_Far"], marker="o", label="Cottage cheese")
    ax.set(xlabel="Iteration", ylabel="Best-so-far utility", xlim=(1, 10), ylim=(0.75, 1.0))
    ax.grid(alpha=0.2); ax.legend()
    save(fig, args.out_dir / "figure30_campaign_comparison.png")

    hfs_exploration_plot(read("leg1_data.csv"), args.out_dir / "figure31_hfs_exploration.png")
    egbo_vs_hfs_plot(
        read("yogurt_utility_per_iteration.csv"), read("leg2_data.csv"),
        args.out_dir / "figure32_egbo_vs_hfs.png",
    )
    print(f"saved deployment figures to {args.out_dir}")


if __name__ == "__main__":
    main()
