"""Hartmann6 convergence figures (paper Figs 1 and 6).

Reads results/<NAME>/{oneshot,adaptive}/raw.json (produced by merge_hartmann6.py)
and writes:
    results/<NAME>/oneshot_convergence.png
    results/<NAME>/adaptive_convergence.png

Usage:
    python plot_hartmann6.py --name release_v1                 # default convergence
    python plot_hartmann6.py --mode q-sweep --qsweep-glob ...  # Fig 4a
    python plot_hartmann6.py --mode pq-heatmap --pq-glob ...   # Fig 4b
    python plot_hartmann6.py --mode n-panel --n-glob ...       # Fig 4c
    python plot_hartmann6.py --mode active-set --qsweep-glob ... # Fig 4d
"""

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

matplotlib.rcParams["font.family"] = "Helvetica Neue"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial"]
matplotlib.rcParams["mathtext.fontset"] = "stixsans"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

GLOBAL_MAX = 3.32237

# Methods that get a full 95% CI band along the trajectory. Everything else
# gets a terminal error-bar tick at iter=T to keep the middle of the plot clean.
HERO_BAND_METHODS = {"oracle", "egbo_strong", "egbo_weak"}

# Hero methods (Oracle + EGBO variants) use thicker lines; baselines are dimmer.
METHOD_STYLE = {
    "random":         {"color": "#8c564b", "ls": ":",  "lw": 1.5, "label": "Random Search",                    "zorder": 1},
    "vanilla":        {"color": "#7f7f7f", "ls": "--", "lw": 1.5, "label": "Vanilla BO",                        "zorder": 2},
    "random_subset":  {"color": "#7d6caf", "ls": "--", "lw": 1.5, "label": "Random subset + BO ($k$=6)",        "zorder": 3},
    "rembo":          {"color": "#2bb3c0", "ls": "-",  "lw": 1.5, "label": "REMBO ($k$=6)",                     "zorder": 4},
    "turbo":          {"color": "#2c70b6", "ls": "-.", "lw": 1.5, "label": "TuRBO",                             "zorder": 5},
    "saasbo":         {"color": "#e76f51", "ls": "-.", "lw": 1.5, "label": "SAASBO",                            "zorder": 6},
    "sebo":           {"color": "#d268a4", "ls": "-.", "lw": 1.5, "label": "SEBO",                              "zorder": 7},
    "oracle":         {"color": "black",    "ls": "-",  "lw": 2.5, "label": "Oracle ($p$=1, $q$=0)",            "zorder": 8},
    "egbo_strong":    {"color": "#1b9e77", "ls": "-",  "lw": 2.5, "label": "EGBO ($p$=0.9, $q$=0.01)",          "zorder": 11},
    "egbo_weak":      {"color": "#d95f02", "ls": "-",  "lw": 2.5, "label": "EGBO ($p$=0.2, $q$=0.10)",          "zorder": 10},
}


def plot_one(raw_path: str, out_path: str, title: str):
    with open(raw_path) as f:
        d = json.load(f)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    iters = None

    # Plot in zorder-ascending so EGBO ends up on top.
    items = sorted(
        d["results"].items(),
        key=lambda kv: METHOD_STYLE.get(kv[0], {"zorder": 0})["zorder"],
    )
    for method, data in items:
        traces = np.array(data["all_traces"])
        n_seeds, n_iters = traces.shape
        if iters is None:
            iters = np.arange(1, n_iters + 1)
        mean = traces.mean(axis=0)
        sem = (traces.std(axis=0, ddof=1) / np.sqrt(n_seeds)
               if n_seeds > 1 else np.zeros_like(mean))
        style = METHOD_STYLE.get(method, {"color": "grey", "ls": "-",
                                           "lw": 1.5, "label": method, "zorder": 0})
        ax.plot(iters, mean, color=style["color"], ls=style["ls"],
                label=style["label"], linewidth=style["lw"], zorder=style["zorder"])
        ax.fill_between(iters, mean - 1.96 * sem, mean + 1.96 * sem,
                        color=style["color"], alpha=0.08,
                        zorder=style["zorder"] - 1, linewidth=0)

    ax.axhline(GLOBAL_MAX, color="gold", ls=":", lw=1.5,
               label=f"Global opt. ({GLOBAL_MAX:.4f})", zorder=20)
    ax.set_xlabel("Evaluation", fontsize=12)
    ax.set_ylabel("Best Hartmann6 Value Found", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right", fontsize=9, ncol=2, frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_both(oneshot_path: str, adaptive_path: str, out_path: str,
              hero_only: bool = False,
              right_panel_label: str = "(b) adaptive ($Q$=$T$)"):
    """Single figure with both modes side-by-side, shared y-axis and legend.

    If hero_only is True, only HERO_BAND_METHODS get the trajectory CI band;
    other methods get a terminal 95% CI error bar at the final iteration.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)
    panel_titles = [
        "(a) one-shot ($Q$=1)",
        right_panel_label,
    ]
    raw_paths = [oneshot_path, adaptive_path]
    handles, labels = [], []

    for ax, panel_title, raw_path in zip(axes, panel_titles, raw_paths):
        with open(raw_path) as f:
            d = json.load(f)
        items = sorted(
            d["results"].items(),
            key=lambda kv: METHOD_STYLE.get(kv[0], {"zorder": 0})["zorder"],
        )
        iters = None
        for method, data in items:
            traces = np.array(data["all_traces"])
            n_seeds, n_iters = traces.shape
            if iters is None:
                iters = np.arange(1, n_iters + 1)
            mean = traces.mean(axis=0)
            sem = (traces.std(axis=0, ddof=1) / np.sqrt(n_seeds)
                   if n_seeds > 1 else np.zeros_like(mean))
            style = METHOD_STYLE.get(method, {"color": "grey", "ls": "-",
                                               "lw": 1.5, "label": method, "zorder": 0})
            line, = ax.plot(iters, mean, color=style["color"], ls=style["ls"],
                            label=style["label"], linewidth=style["lw"],
                            zorder=style["zorder"])
            if (not hero_only) or method in HERO_BAND_METHODS:
                ax.fill_between(iters, mean - 1.96 * sem, mean + 1.96 * sem,
                                color=style["color"], alpha=0.08,
                                zorder=style["zorder"] - 1, linewidth=0)
            else:
                # Terminal 95% CI error bar nudged just past iter=T so it
                # doesn't overlap the line endpoints.
                ax.errorbar(iters[-1] + 0.25, mean[-1],
                            yerr=1.96 * sem[-1],
                            color=style["color"], capsize=3, capthick=1.0,
                            elinewidth=1.0, alpha=0.85,
                            zorder=style["zorder"])
            if style["label"] not in labels:
                handles.append(line); labels.append(style["label"])

        line_opt = ax.axhline(GLOBAL_MAX, color="gold", ls=":", lw=1.5, zorder=20)
        ax.set_xlabel("Evaluation", fontsize=11)
        ax.set_title(panel_title, fontsize=12)
        ax.grid(True, alpha=0.3)

    if "Global opt." not in [l.split(" (")[0] for l in labels]:
        opt_label = f"Global opt. ({GLOBAL_MAX:.4f})"
        handles.append(line_opt); labels.append(opt_label)

    axes[0].set_ylabel("Best Hartmann6 Value Found", fontsize=11)
    fig.suptitle("EGBO vs Baselines on Hartmann6 ($d$=200)", fontsize=13, y=0.99)
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.10, 1, 0.96])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# Helpers for the new plots: load raw.json with config + results.
# ---------------------------------------------------------------------------

def _load_raw(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _final_regret(traces: np.ndarray) -> Tuple[float, float, int]:
    """Return (mean, sem, n_seeds) of GLOBAL_MAX − final-best across seeds."""
    finals = traces[:, -1]
    n = len(finals)
    gap = GLOBAL_MAX - finals
    mean = float(gap.mean())
    sem = float(gap.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, sem, n


# ---------------------------------------------------------------------------
# Fig 4a: intermediate-Q sweep (regret-at-T vs Q)
# ---------------------------------------------------------------------------

def plot_q_sweep(
    qsweep_paths: List[str], out_path: str,
    methods: List[str] = ("egbo_strong", "egbo_weak"),
    title: str = "Q-sweep on Hartmann6 ($d$=200)",
) -> None:
    """Each path is a raw.json from one (Q, methods) run. Q is read from
    config['n_rounds']."""
    by_method: Dict[str, Dict[int, Tuple[float, float, int]]] = {m: {} for m in methods}
    for path in qsweep_paths:
        d = _load_raw(path)
        q = d["config"].get("n_rounds")
        if q is None:
            # Fall back to expert_mode → Q ∈ {1, T=budget} mapping.
            q = 1 if d["config"].get("expert_mode") == "oneshot" else int(d["config"].get("budget", 0))
        # Under the sweep's schedule (one query before the initial design and
        # one before each BO step) at most T - n_init + 1 distinct rounds exist;
        # larger n_rounds (e.g. Q=20 with T=20, n_init=5) duplicate that cell.
        max_rounds = int(d["config"].get("budget", 20)) - int(d["config"].get("n_init", 5)) + 1
        if int(q) > max_rounds:
            continue
        for method in methods:
            if method not in d["results"]:
                continue
            traces = np.array(d["results"][method]["all_traces"])
            by_method[method][int(q)] = _final_regret(traces)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for method in methods:
        if not by_method[method]:
            continue
        qs = sorted(by_method[method].keys())
        means = [by_method[method][q][0] for q in qs]
        sems = [by_method[method][q][1] for q in qs]
        style = METHOD_STYLE.get(method, {"color": "grey", "ls": "-", "lw": 2.0,
                                          "label": method})
        ax.errorbar(qs, means, yerr=[1.96 * s for s in sems],
                    color=style["color"], ls=style["ls"], lw=style["lw"],
                    marker="o", capsize=3, label=style["label"])
    ax.set_xscale("log")
    ax.set_xlabel("Expert query rounds $Q$ (max. 16 under this schedule)", fontsize=11)
    ax.set_ylabel("Regret at $T$ (global opt $-$ best found)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="center right", fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# Fig 4b: (p, q_fp) heatmap of fraction of oracle gap closed
# ---------------------------------------------------------------------------

def plot_pq_heatmap(
    pq_paths: List[str], out_path: str,
    title: str = "EGBO performance vs expert quality on Hartmann6 ($d$=200)",
    method: str = "egbo_custom",
    baseline_method: str = "vanilla",
    oracle_method: str = "oracle",
) -> None:
    """Each path is a raw.json from one (p, q) cell. Reads config['expert_p'],
    config['expert_q'], plus baseline_method and oracle_method from the same
    file (or any file with them) for normalization."""
    cells: Dict[Tuple[float, float], float] = {}
    base_finals: List[float] = []
    oracle_finals: List[float] = []
    for path in pq_paths:
        d = _load_raw(path)
        # Always pick up baseline/oracle finals from any input file that has
        # them — companion baseline runs may live in their own raw.json
        # without (p, q) set.
        if baseline_method in d.get("results", {}):
            base_finals.append(float(np.array(d["results"][baseline_method]["all_traces"])[:, -1].mean()))
        if oracle_method in d.get("results", {}):
            oracle_finals.append(float(np.array(d["results"][oracle_method]["all_traces"])[:, -1].mean()))
        p = d["config"].get("expert_p")
        q = d["config"].get("expert_q")
        if p is None or q is None:
            continue
        if method not in d["results"]:
            continue
        traces = np.array(d["results"][method]["all_traces"])
        cells[(float(p), float(q))] = float(traces[:, -1].mean())

    if not cells:
        raise ValueError("plot_pq_heatmap: no (p, q) cells found in inputs.")
    if not base_finals or not oracle_finals:
        warn = (
            "plot_pq_heatmap: missing baseline ({}) or oracle ({}) finals in inputs; "
            "heatmap will show raw final-best instead of fraction-of-gap-closed."
        ).format(baseline_method, oracle_method)
        print(f"  WARN: {warn}")
        normalize = False
    else:
        normalize = True
        base = float(np.mean(base_finals))
        oracle = float(np.mean(oracle_finals))

    ps = sorted({pq[0] for pq in cells})
    qs = sorted({pq[1] for pq in cells})
    grid = np.full((len(qs), len(ps)), np.nan)
    for (p, q), v in cells.items():
        i = qs.index(q)
        j = ps.index(p)
        if normalize:
            denom = (oracle - base) if abs(oracle - base) > 1e-9 else 1.0
            grid[i, j] = (v - base) / denom
        else:
            grid[i, j] = v

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    if normalize:
        # Diverging scale centred at 0 (= vanilla BO) so that cells where EGBO
        # underperforms vanilla BO remain visible instead of being clipped.
        lo = min(-1.0, float(np.nanmin(grid)))
        norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=1.0)
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdYlGn", norm=norm)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i - 0.28, f"{grid[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color="black")
    else:
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels([f"{p:g}" for p in ps])
    ax.set_yticks(range(len(qs)))
    ax.set_yticklabels([f"{q:g}" for q in qs])
    ax.set_xlabel("Per-variable recall $p$", fontsize=11)
    ax.set_ylabel("Per-variable false-positive rate $q_{fp}$", fontsize=11)
    ax.set_title(title, fontsize=12)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fraction of oracle gap closed (0 = vanilla BO, 1 = oracle)" if normalize else "Final best", fontsize=9)

    # Annotate the cells corresponding to the paper's strong/weak presets.
    for (p_anno, q_anno, label) in [
        (0.9, 0.01, "egbo_strong"),
        (0.2, 0.10, "egbo_weak"),
    ]:
        if p_anno in ps and q_anno in qs:
            ax.scatter([ps.index(p_anno)], [qs.index(q_anno)],
                       marker="*", s=140, c="white", edgecolors="black",
                       linewidths=0.8, zorder=5)
            ax.text(ps.index(p_anno), qs.index(q_anno) + 0.2, label,
                    color="white", fontsize=8, ha="center",
                    bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# Fig 4c: larger-N panel
# ---------------------------------------------------------------------------

def plot_n_panel(
    n_paths: List[str], out_path: str,
    title: str = "Convergence vs ambient dimension $N$",
) -> None:
    """Each path is a raw.json from one (N) full-method run. Reads
    config['d_total'] and produces side-by-side convergence panels."""
    runs = []
    for path in n_paths:
        d = _load_raw(path)
        N = int(d["config"]["d_total"])
        runs.append((N, d))
    runs.sort(key=lambda kv: kv[0])
    n_panels = len(runs)
    if n_panels == 0:
        raise ValueError("plot_n_panel: no inputs")

    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]
    handles, labels = [], []
    for ax, (N, d) in zip(axes, runs):
        items = sorted(
            d["results"].items(),
            key=lambda kv: METHOD_STYLE.get(kv[0], {"zorder": 0})["zorder"],
        )
        iters = None
        for method, data in items:
            traces = np.array(data["all_traces"])
            n_seeds, n_iters = traces.shape
            if iters is None:
                iters = np.arange(1, n_iters + 1)
            mean = traces.mean(axis=0)
            sem = (traces.std(axis=0, ddof=1) / np.sqrt(n_seeds)
                   if n_seeds > 1 else np.zeros_like(mean))
            style = METHOD_STYLE.get(method, {"color": "grey", "ls": "-",
                                              "lw": 1.5, "label": method, "zorder": 0})
            line, = ax.plot(iters, mean, color=style["color"], ls=style["ls"],
                            lw=style["lw"], label=style["label"], zorder=style["zorder"])
            ax.fill_between(iters, mean - 1.96 * sem, mean + 1.96 * sem,
                            color=style["color"], alpha=0.08,
                            zorder=style["zorder"] - 1, linewidth=0)
            if style["label"] not in labels:
                handles.append(line); labels.append(style["label"])
        ax.axhline(GLOBAL_MAX, color="gold", ls=":", lw=1.2, zorder=20)
        ax.set_title(f"$N$={N}", fontsize=12)
        ax.set_xlabel("Evaluation", fontsize=11)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Best Hartmann6 Value Found", fontsize=11)
    fig.suptitle(title, fontsize=13, y=0.99)
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.10, 1, 0.96])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# Fig 4d: active-set size |S_q| trajectory
# ---------------------------------------------------------------------------

def plot_active_set_trajectory(
    qsweep_paths: List[str], out_path: str,
    methods: Tuple[str, str] = ("egbo_strong", "egbo_weak"),
    qs_to_plot: Tuple[int, ...] = (1, 4, 16),
    title: str = "Active-set size $|S_q|$ over BO iterations",
) -> None:
    """Reads `active_set_all_traces` and `relevant_all_traces` (the
    per-iteration |S_q| and |S_q ∩ relevant| sequences emitted by run_egbo)
    from Q-sweep raw.json files. Two panels (one per method); curves per Q."""
    by_method: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {m: {} for m in methods}
    for path in qsweep_paths:
        d = _load_raw(path)
        q = d["config"].get("n_rounds")
        if q is None:
            q = 1 if d["config"].get("expert_mode") == "oneshot" else int(d["config"].get("budget", 0))
        if int(q) not in qs_to_plot:
            continue
        for method in methods:
            r = d["results"].get(method)
            if r is None or "active_set_all_traces" not in r:
                continue
            active = np.array(r["active_set_all_traces"], dtype=float)
            relevant = np.array(r["relevant_all_traces"], dtype=float) if "relevant_all_traces" in r else None
            by_method[method][int(q)] = {"active": active, "relevant": relevant}

    fig, axes = plt.subplots(1, len(methods), figsize=(5.5 * len(methods), 4.0),
                             sharey=True)
    if len(methods) == 1:
        axes = [axes]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for ax, method in zip(axes, methods):
        cell = by_method[method]
        title_m = METHOD_STYLE.get(method, {"label": method})["label"]
        for color, q in zip(palette, sorted(cell.keys())):
            active = cell[q]["active"]
            relevant = cell[q]["relevant"]
            iters = np.arange(1, active.shape[1] + 1)
            mean_active = active.mean(axis=0)
            ax.plot(iters, mean_active, color=color, lw=2.0, label=f"$|S_q|$, $Q$={q}")
            if relevant is not None:
                mean_relevant = relevant.mean(axis=0)
                ax.plot(iters, mean_relevant, color=color, lw=1.2, ls="--",
                        label=f"$|S_q \\cap \\mathcal{{U}}_\\eta|$, $Q$={q}")
        ax.set_xlabel("Evaluation", fontsize=11)
        ax.set_title(title_m, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, frameon=True)
    axes[0].set_ylabel("Active-set size", fontsize=11)
    fig.suptitle(title, fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# Path discovery: expand a glob into raw.json paths.
# ---------------------------------------------------------------------------

def _expand_glob(pattern: str) -> List[str]:
    matches = sorted(glob.glob(pattern))
    paths: List[str] = []
    for m in matches:
        if os.path.isdir(m):
            cand = os.path.join(m, "raw.json")
            if os.path.exists(cand):
                paths.append(cand)
        elif m.endswith(".json"):
            paths.append(m)
    return paths


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default=os.path.join(here, "results"))
    p.add_argument("--name", default=None,
                   help="Run directory name under --results-root (required for convergence mode).")
    p.add_argument("--mode", default="convergence",
                   choices=["convergence", "q-sweep", "pq-heatmap", "n-panel", "active-set"],
                   help="Which figure to produce. 'convergence' is the default "
                        "(paper Fig 1); the other modes produce Fig 4a-d.")
    p.add_argument("--separate", action="store_true",
                   help="(convergence mode) Also write per-mode single-panel figures.")
    p.add_argument("--hero-only", action="store_true",
                   help="(convergence mode) Show CI bands only for hero methods.")
    p.add_argument("--out-name", default=None,
                   help="Filename for the figure (mode-specific default).")
    p.add_argument("--out-path", default=None,
                   help="Full output path (overrides --results-root/--name/--out-name).")
    p.add_argument("--right-panel-label", default="(b) adaptive ($Q$=$T$)",
                   help="(convergence mode) Title for the right (adaptive) panel.")
    p.add_argument("--qsweep-glob", default=None,
                   help="(q-sweep / active-set modes) Glob expanding to either "
                        "raw.json files or directories that contain raw.json. "
                        "Example: 'results/qsweep_q*' or 'results/qsweep_q*/raw.json'.")
    p.add_argument("--pq-glob", default=None,
                   help="(pq-heatmap mode) Glob expanding to (p, q) cell raw.json files.")
    p.add_argument("--n-glob", default=None,
                   help="(n-panel mode) Glob expanding to N=... raw.json files.")
    args = p.parse_args()

    if args.mode == "convergence":
        if args.name is None:
            raise SystemExit("--mode convergence requires --name.")
        base = os.path.join(args.results_root, args.name)
        out = args.out_path or os.path.join(base, args.out_name or "convergence.png")
        plot_both(
            os.path.join(base, "oneshot", "raw.json"),
            os.path.join(base, "adaptive", "raw.json"),
            out,
            hero_only=args.hero_only,
            right_panel_label=args.right_panel_label,
        )
        if args.separate:
            plot_one(
                os.path.join(base, "oneshot", "raw.json"),
                os.path.join(base, "oneshot_convergence.png"),
                "EGBO vs Baselines on Hartmann6 ($d$=200), one-shot ($Q$=1)",
            )
            plot_one(
                os.path.join(base, "adaptive", "raw.json"),
                os.path.join(base, "adaptive_convergence.png"),
                "EGBO vs Baselines on Hartmann6 ($d$=200), adaptive ($Q$=$T$)",
            )
        return

    if args.mode == "q-sweep":
        if not args.qsweep_glob:
            raise SystemExit("--mode q-sweep requires --qsweep-glob.")
        paths = _expand_glob(args.qsweep_glob)
        if not paths:
            raise SystemExit(f"No raw.json files matched {args.qsweep_glob!r}")
        out = args.out_path or os.path.join(
            args.results_root, args.name or "qsweep", args.out_name or "qsweep.png",
        )
        plot_q_sweep(paths, out)
        return

    if args.mode == "pq-heatmap":
        if not args.pq_glob:
            raise SystemExit("--mode pq-heatmap requires --pq-glob.")
        paths = _expand_glob(args.pq_glob)
        if not paths:
            raise SystemExit(f"No raw.json files matched {args.pq_glob!r}")
        out = args.out_path or os.path.join(
            args.results_root, args.name or "pq", args.out_name or "pq_heatmap.png",
        )
        plot_pq_heatmap(paths, out)
        return

    if args.mode == "n-panel":
        if not args.n_glob:
            raise SystemExit("--mode n-panel requires --n-glob.")
        paths = _expand_glob(args.n_glob)
        if not paths:
            raise SystemExit(f"No raw.json files matched {args.n_glob!r}")
        out = args.out_path or os.path.join(
            args.results_root, args.name or "n_panel", args.out_name or "n_panel.png",
        )
        plot_n_panel(paths, out)
        return

    if args.mode == "active-set":
        if not args.qsweep_glob:
            raise SystemExit("--mode active-set requires --qsweep-glob.")
        paths = _expand_glob(args.qsweep_glob)
        if not paths:
            raise SystemExit(f"No raw.json files matched {args.qsweep_glob!r}")
        out = args.out_path or os.path.join(
            args.results_root, args.name or "qsweep", args.out_name or "active_set.png",
        )
        plot_active_set_trajectory(paths, out)
        return


if __name__ == "__main__":
    main()
