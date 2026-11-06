"""Merge per-job Hartmann6 results into per-mode raw.json files.

run_hartmann6.sh fans out the following jobs (with SEEDS=5):
    <NAME>__cheap                        (oneshot, 6 baselines, 5 seeds)
    <NAME>__saasbo_seed{0,1,2,3,4}       (oneshot, 1 seed each)
    <NAME>__sebo_seed{0,1,2,3,4}         (oneshot, 1 seed each)
    <NAME>__egbo_oneshot                 (oneshot, egbo_strong + egbo_weak)
    <NAME>__egbo_adaptive                (adaptive, egbo_strong + egbo_weak)

This script:
  1. Combines per-seed SAASBO and SEBO outputs back into 5-seed entries.
  2. Unions baselines + EGBO into per-mode raw.jsons:
        results/<NAME>/oneshot/raw.json   (8 baselines + 2 EGBO oneshot)
        results/<NAME>/adaptive/raw.json  (8 baselines + 2 EGBO adaptive)

Non-EGBO baselines are mode-independent; their results are reused across
both output JSONs.
"""

import argparse
import json
import os

import numpy as np


def _load(name: str, slug: str, root: str) -> dict:
    path = os.path.join(root, f"{name}__{slug}", "raw.json")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} — did all jobs finish?")
    with open(path) as f:
        return json.load(f)


def _atomic_write_json(path: str, obj) -> None:
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _combine_seed_jobs(name: str, slug_prefix: str, n_seeds: int, root: str) -> dict:
    """Combine N single-seed jobs <name>__<prefix>_seed{0..N-1} into one entry.

    Returns {method_name: combined_method_result} where the combined
    method_result has all N seeds' traces concatenated and stats recomputed.
    """
    method_traces: dict = {}  # method_name -> list of trace lists
    for s in range(n_seeds):
        d = _load(name, f"{slug_prefix}_seed{s}", root)
        for method, mr in d["results"].items():
            method_traces.setdefault(method, []).extend(mr["all_traces"])

    combined = {}
    for method, traces in method_traces.items():
        arr = np.array(traces)
        combined[method] = {
            "all_traces": traces,
            "mean_trace": arr.mean(axis=0).tolist(),
            "std_trace": arr.std(axis=0).tolist(),
            "mean_final": float(arr[:, -1].mean()),
            "std_final": float(arr[:, -1].std()),
        }
    return combined


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base", required=True, help="run name prefix, e.g. release_v1")
    p.add_argument("--seeds", type=int, default=5,
                   help="Number of seeds (must match what run_hartmann6.sh used)")
    p.add_argument("--results-root", default=None,
                   help="default: results/ next to this script")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = args.results_root or os.path.join(here, "results")

    cheap = _load(args.base, "cheap", root)
    saasbo = _combine_seed_jobs(args.base, "saasbo", args.seeds, root)
    sebo = _combine_seed_jobs(args.base, "sebo", args.seeds, root)
    egbo_one = _load(args.base, "egbo_oneshot", root)
    egbo_adp = _load(args.base, "egbo_adaptive", root)

    # Baselines (mode-independent)
    baselines = {**cheap["results"], **saasbo, **sebo}

    base_config = dict(cheap["config"])
    for mode_name, egbo_payload in [("oneshot", egbo_one), ("adaptive", egbo_adp)]:
        results = {**baselines, **egbo_payload["results"]}
        config = dict(base_config)
        config["expert_mode"] = mode_name
        config["methods"] = list(results.keys())
        config["n_seeds"] = args.seeds
        out_path = os.path.join(root, args.base, mode_name, "raw.json")
        _atomic_write_json(out_path, {"config": config, "results": results})
        print(f"wrote {out_path}  ({len(results)} methods)")


if __name__ == "__main__":
    main()
