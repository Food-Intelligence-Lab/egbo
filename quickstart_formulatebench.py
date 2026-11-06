#!/usr/bin/env python3
"""Dependency-light FormulateBench quickstart.

This script demonstrates the public evaluator with random Dirichlet search
and never uses an LLM, API key, BoTorch, or a pre-existing output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "formulateBench"))

from formulatebench import get_task  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small, API-free FormulateBench example."
    )
    parser.add_argument("--task", default="dairy/yogurt")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "quickstart_results" / "random_search.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.budget < 1:
        raise SystemExit("--budget must be at least 1")

    task = get_task(args.task)
    rng = np.random.default_rng(args.seed)
    best_loss = float("inf")
    best_recipe = None
    trajectory = []

    for evaluation in range(1, args.budget + 1):
        weights = rng.dirichlet(np.ones(task.dimension))
        recipe = dict(zip(task.ingredient_names, map(float, weights)))
        loss = task.evaluate(recipe).loss
        if loss < best_loss:
            best_loss = loss
            best_recipe = recipe
        trajectory.append(
            {
                "evaluation": evaluation,
                "loss": loss,
                "best_so_far": best_loss,
            }
        )

    payload = {
        "task": task.id,
        "method": "random Dirichlet search",
        "budget": args.budget,
        "seed": args.seed,
        "best_loss": best_loss,
        "best_recipe": best_recipe,
        "trajectory": trajectory,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Task: {task.id} (d={task.dimension})")
    print(f"Best loss after {args.budget} evaluations: {best_loss:.6f}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
