#!/usr/bin/env python3
"""Minimal template for evaluating a new optimization method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from formulatebench import get_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="dairy/yogurt")
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("method_result.json"))
    args = parser.parse_args()

    task = get_task(args.task)
    rng = np.random.default_rng(args.seed)
    best = float("inf")
    trajectory = []

    for evaluation in range(1, args.budget + 1):
        # Replace this proposal rule with your method. The benchmark only
        # requires a nonnegative mixture over allowed ingredients that sums
        # to one.
        weights = rng.dirichlet(np.ones(task.dimension))
        candidate = dict(zip(task.ingredient_names, map(float, weights)))
        loss = task.evaluate(candidate).loss
        best = min(best, loss)
        trajectory.append(
            {"evaluation": evaluation, "loss": loss, "best_so_far": best}
        )

    result = {
        "schema_version": "1.0",
        "task": task.id,
        "method": "replace-with-your-method-name",
        "budget": args.budget,
        "seed": args.seed,
        "trajectory": trajectory,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}; final best loss = {best:.6f}")


if __name__ == "__main__":
    main()
