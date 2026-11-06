"""Stable, dependency-light FormulateBench evaluator.

The evaluator intentionally contains no Bayesian-optimization or LLM code.
It defines the frozen task data, candidate validation, and objective used by
the paper. Researchers can use any optimizer and call :meth:`evaluate` for
each proposed formulation.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Iterable, Mapping, Sequence

import numpy as np


NUTRIENT_COLUMNS = (
    "energy_kcal_per_100g",
    "protein_g_per_100g",
    "fat_g_per_100g",
    "saturated_fat_g_per_100g",
    "carbs_g_per_100g",
    "fiber_g_per_100g",
    "sugars_g_per_100g",
    "sodium_mg_per_100g",
)

NUTRIENT_LABELS = (
    "energy",
    "protein",
    "fat",
    "saturated_fat",
    "carbohydrates",
    "fiber",
    "sugars",
    "sodium",
)

DOMAINS = ("meat", "dairy")
DOMAIN_COLUMNS = {"meat": "in_meat", "dairy": "in_dairy"}
EXCLUDED_INGREDIENTS = {
    "meat": frozenset({"egg white", "egg whites", "fat", "milk", "milk proteins"}),
    "dairy": frozenset({"mozzarella"}),
}


class RecipeError(ValueError):
    """Raised when a candidate formulation is invalid."""


class TaskNotFoundError(KeyError):
    """Raised when a domain/category pair is not in the benchmark."""


@dataclass(frozen=True)
class Evaluation:
    """Result returned by :meth:`FormulationTask.evaluate`."""

    loss: float
    normalized_loss: float
    profile: Mapping[str, float]
    target: Mapping[str, float]
    measured_nutrients: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "loss": self.loss,
            "normalized_loss": self.normalized_loss,
            "profile": dict(self.profile),
            "target": dict(self.target),
            "measured_nutrients": list(self.measured_nutrients),
        }


def _data_path(filename: str):
    return files("formulatebench").joinpath("data", filename)


def _number(value: str | None, *, missing: float = 0.0) -> float:
    if value is None or not str(value).strip():
        return missing
    try:
        return float(value)
    except ValueError:
        return missing


def _ingredient_rows(domain: str) -> list[dict[str, str]]:
    if domain not in DOMAINS:
        raise TaskNotFoundError(f"unknown domain {domain!r}; choose from {DOMAINS}")
    with _data_path("usda_nutrition_data.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    column = DOMAIN_COLUMNS[domain]
    excluded = EXCLUDED_INGREDIENTS[domain]
    return [
        row
        for row in rows
        if row.get(column) == "Y"
        and row.get("ingredient", "").strip().casefold() not in excluded
    ]


def _target_rows(domain: str) -> list[dict[str, str]]:
    with _data_path("animal_products_nutrition.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [row for row in rows if row.get("category") == domain]


def list_tasks(domain: str | None = None) -> tuple[str, ...]:
    """Return task identifiers in ``domain/category`` form."""

    domains: Iterable[str] = DOMAINS if domain is None else (domain,)
    identifiers: list[str] = []
    for current_domain in domains:
        if current_domain not in DOMAINS:
            raise TaskNotFoundError(
                f"unknown domain {current_domain!r}; choose from {DOMAINS}"
            )
        identifiers.extend(
            f"{current_domain}/{row['ingredient']}"
            for row in _target_rows(current_domain)
        )
    return tuple(identifiers)


class FormulationTask:
    """One frozen FormulateBench optimization task.

    A candidate is a mapping from allowed ingredient names to nonnegative
    mixture fractions summing to one. The returned loss is z-score RMSE to
    the animal-product target over the nutrients reported for that category;
    lower is better.
    """

    def __init__(self, domain: str, category: str):
        matches = [
            row
            for row in _target_rows(domain)
            if row.get("ingredient", "").casefold() == category.casefold()
        ]
        if not matches:
            raise TaskNotFoundError(
                f"task {domain}/{category} not found; run `formulatebench list`"
            )

        rows = _ingredient_rows(domain)
        self.domain = domain
        self.category = matches[0]["ingredient"]
        self.id = f"{domain}/{self.category}"
        self.ingredient_names = tuple(row["ingredient"] for row in rows)
        self._lookup = {name.casefold(): i for i, name in enumerate(self.ingredient_names)}
        self._matrix = np.asarray(
            [[_number(row.get(column)) for column in NUTRIENT_COLUMNS] for row in rows],
            dtype=float,
        )

        target_row = matches[0]
        target_values: list[float] = []
        mask: list[bool] = []
        for column in NUTRIENT_COLUMNS:
            raw = target_row.get(column)
            present = raw is not None and bool(str(raw).strip())
            mask.append(present)
            target_values.append(_number(raw))
        self._target = np.asarray(target_values, dtype=float)
        self._mask = np.asarray(mask, dtype=bool)
        if not self._mask.any():
            raise ValueError(f"task {self.id} has no measured target nutrients")

        self._mean = self._matrix.mean(axis=0)
        sigma = self._matrix.std(axis=0)
        self._sigma = np.where(sigma > 0, sigma, 1.0)
        self._z_target = (self._target - self._mean) / self._sigma
        with _data_path("normalizers.json").open("r", encoding="utf-8") as stream:
            normalizers = json.load(stream)
        self.normalizer = float(normalizers[self.domain][self.category])

    @property
    def dimension(self) -> int:
        return len(self.ingredient_names)

    @property
    def measured_nutrients(self) -> tuple[str, ...]:
        return tuple(
            label for label, measured in zip(NUTRIENT_LABELS, self._mask) if measured
        )

    def _vectorize(
        self,
        recipe: Mapping[str, float] | Sequence[Mapping[str, object]],
    ) -> np.ndarray:
        if isinstance(recipe, Mapping):
            items = recipe.items()
        else:
            parsed: list[tuple[str, float]] = []
            for item in recipe:
                if not isinstance(item, Mapping):
                    raise RecipeError("recipe entries must be JSON objects")
                ingredient = item.get("ingredient")
                fraction = item.get("fraction")
                if ingredient is None or fraction is None:
                    raise RecipeError(
                        "each recipe entry needs `ingredient` and `fraction`"
                    )
                parsed.append((str(ingredient), float(fraction)))
            items = parsed

        vector = np.zeros(self.dimension, dtype=float)
        seen = False
        for ingredient, fraction_value in items:
            seen = True
            key = str(ingredient).strip().casefold()
            if key not in self._lookup:
                raise RecipeError(
                    f"ingredient {ingredient!r} is not allowed for {self.id}"
                )
            try:
                fraction = float(fraction_value)
            except (TypeError, ValueError) as exc:
                raise RecipeError(
                    f"fraction for {ingredient!r} must be numeric"
                ) from exc
            if not np.isfinite(fraction) or fraction < 0:
                raise RecipeError(
                    f"fraction for {ingredient!r} must be finite and nonnegative"
                )
            vector[self._lookup[key]] += fraction

        if not seen:
            raise RecipeError("recipe must contain at least one ingredient")
        total = float(vector.sum())
        if not np.isclose(total, 1.0, atol=1e-6, rtol=0):
            raise RecipeError(f"recipe fractions must sum to 1.0; received {total:.8f}")
        return vector

    def evaluate(
        self,
        recipe: Mapping[str, float] | Sequence[Mapping[str, object]],
    ) -> Evaluation:
        """Validate and score one formulation. Lower loss is better."""

        vector = self._vectorize(recipe)
        profile = vector @ self._matrix
        z_profile = (profile - self._mean) / self._sigma
        difference = (z_profile - self._z_target)[self._mask]
        loss = float(np.sqrt(np.mean(difference**2)))
        return Evaluation(
            loss=loss,
            normalized_loss=loss / self.normalizer,
            profile={
                label: float(value)
                for label, value in zip(NUTRIENT_LABELS, profile)
            },
            target={
                label: float(value)
                for label, value, measured in zip(
                    NUTRIENT_LABELS, self._target, self._mask
                )
                if measured
            },
            measured_nutrients=self.measured_nutrients,
        )

    def metadata(self) -> dict:
        return {
            "id": self.id,
            "domain": self.domain,
            "category": self.category,
            "dimension": self.dimension,
            "measured_nutrients": list(self.measured_nutrients),
            "objective": "z-score RMSE (lower is better)",
            "normalizer": self.normalizer,
            "normalized_objective": "loss / mean first random-search loss",
            "constraint": "nonnegative mixture fractions summing to one",
            "excluded_ingredients": sorted(EXCLUDED_INGREDIENTS[self.domain]),
        }


def get_task(identifier: str, category: str | None = None) -> FormulationTask:
    """Load a task by ``domain/category`` or by two arguments."""

    if category is None:
        if "/" not in identifier:
            raise TaskNotFoundError(
                "task identifier must have the form `domain/category`"
            )
        domain, category = identifier.split("/", 1)
    else:
        domain = identifier
    return FormulationTask(domain.strip().casefold(), category.strip())
