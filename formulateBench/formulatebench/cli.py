"""Command-line interface for FormulateBench."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .core import DOMAINS, RecipeError, TaskNotFoundError, get_task, list_tasks


def _write_json(payload: object, output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if output is None:
        print(rendered)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {output}")


def _load_recipe(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "recipe" in payload:
        return payload["recipe"]
    return payload


def _command_list(args: argparse.Namespace) -> None:
    for identifier in list_tasks(args.domain):
        task = get_task(identifier)
        print(
            f"{identifier}\td={task.dimension}\t"
            f"nutrients={','.join(task.measured_nutrients)}"
        )


def _command_describe(args: argparse.Namespace) -> None:
    _write_json(get_task(args.task).metadata(), args.output)


def _command_evaluate(args: argparse.Namespace) -> None:
    result = get_task(args.task).evaluate(_load_recipe(args.recipe))
    _write_json(result.as_dict(), args.output)


def _command_demo(args: argparse.Namespace) -> None:
    task = get_task(args.task)
    rng = np.random.default_rng(args.seed)
    best_loss = float("inf")
    best_recipe: dict[str, float] | None = None
    trajectory: list[dict] = []
    for evaluation in range(1, args.budget + 1):
        weights = rng.dirichlet(np.ones(task.dimension))
        recipe = {
            ingredient: float(weight)
            for ingredient, weight in zip(task.ingredient_names, weights)
            if weight > 0
        }
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
        "seed": args.seed,
        "budget": args.budget,
        "best_loss": best_loss,
        "best_recipe": best_recipe,
        "trajectory": trajectory,
    }
    _write_json(payload, args.output)


def _command_validate_result(args: argparse.Namespace) -> None:
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    required = {"schema_version", "task", "method", "budget", "seed", "trajectory"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")
    if payload["schema_version"] != "1.0":
        raise ValueError("schema_version must be `1.0`")
    get_task(str(payload["task"]))
    budget = int(payload["budget"])
    trajectory = payload["trajectory"]
    if budget < 1 or len(trajectory) != budget:
        raise ValueError(
            f"trajectory must contain exactly budget={budget} evaluations"
        )
    running_best = float("inf")
    for expected_evaluation, row in enumerate(trajectory, start=1):
        if int(row.get("evaluation", -1)) != expected_evaluation:
            raise ValueError("evaluation numbers must be consecutive and start at 1")
        loss = float(row["loss"])
        best = float(row["best_so_far"])
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("loss values must be finite and nonnegative")
        running_best = min(running_best, loss)
        if not math.isclose(best, running_best, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"best_so_far at evaluation {expected_evaluation} is inconsistent"
            )
    print(
        f"VALID  {args.result}: {payload['task']}, {payload['method']}, "
        f"budget={budget}, final_best={running_best:.6f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="formulatebench",
        description="Evaluate plant-based formulations on FormulateBench.",
    )
    parser.add_argument("--version", action="version", version="FormulateBench 1.0.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list benchmark tasks")
    list_parser.add_argument("--domain", choices=DOMAINS)
    list_parser.set_defaults(handler=_command_list)

    describe = subparsers.add_parser("describe", help="show task metadata")
    describe.add_argument("task", help="task identifier, e.g. dairy/yogurt")
    describe.add_argument("--output", type=Path)
    describe.set_defaults(handler=_command_describe)

    evaluate = subparsers.add_parser("evaluate", help="score one recipe JSON file")
    evaluate.add_argument("task", help="task identifier, e.g. dairy/yogurt")
    evaluate.add_argument("recipe", type=Path)
    evaluate.add_argument("--output", type=Path)
    evaluate.set_defaults(handler=_command_evaluate)

    demo = subparsers.add_parser("demo", help="run a small random-search example")
    demo.add_argument("task", nargs="?", default="dairy/yogurt")
    demo.add_argument("--budget", type=int, default=8)
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--output", type=Path, default=Path("formulatebench_demo.json"))
    demo.set_defaults(handler=_command_demo)

    validate = subparsers.add_parser(
        "validate-result", help="validate a method-result JSON file"
    )
    validate.add_argument("result", type=Path)
    validate.set_defaults(handler=_command_validate_result)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "budget", 1) < 1:
        parser.error("--budget must be at least 1")
    try:
        args.handler(args)
    except (
        RecipeError,
        TaskNotFoundError,
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        parser.error(str(exc))
    return 0
