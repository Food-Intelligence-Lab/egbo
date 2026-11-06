import json

import numpy as np
import pytest

from formulatebench import RecipeError, get_task, list_tasks


def test_release_contains_24_tasks():
    tasks = list_tasks()
    assert len(tasks) == 24
    assert len(list_tasks("meat")) == 14
    assert len(list_tasks("dairy")) == 10


def test_final_protocol_pool_sizes():
    assert get_task("meat/bacon").dimension == 106
    assert get_task("dairy/yogurt").dimension == 90


def test_one_hot_recipe_matches_manual_objective():
    task = get_task("dairy/yogurt")
    result = task.evaluate({"oat milk": 1.0})
    assert np.isfinite(result.loss)
    assert result.loss > 0
    assert set(result.measured_nutrients) == set(result.target)


def test_list_recipe_and_mapping_recipe_are_equivalent():
    task = get_task("dairy/yogurt")
    mapping = task.evaluate({"oat milk": 0.75, "coconut oil": 0.25})
    sequence = task.evaluate(
        [
            {"ingredient": "oat milk", "fraction": 0.75},
            {"ingredient": "coconut oil", "fraction": 0.25},
        ]
    )
    assert mapping.loss == pytest.approx(sequence.loss)


@pytest.mark.parametrize(
    "recipe",
    [
        {"oat milk": 0.5},
        {"not an ingredient": 1.0},
        {"oat milk": -0.1, "coconut oil": 1.1},
        {},
    ],
)
def test_invalid_recipes_are_rejected(recipe):
    with pytest.raises(RecipeError):
        get_task("dairy/yogurt").evaluate(recipe)


def test_metadata_is_json_serializable():
    json.dumps(get_task("meat/burgers").metadata())
