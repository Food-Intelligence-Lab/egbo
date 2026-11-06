import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "formulateBench" / "results"
CATEGORY_COUNTS = {"meat": 14, "dairy": 10}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def assert_finite_trajectory(values, *, expected_budget=20):
    array = np.asarray(values, dtype=float)
    assert array.shape[-1] == expected_budget
    assert np.isfinite(array).all()


def test_conventional_result_shards_are_complete():
    groups = (
        "final_protocol_core_v1",
        "final_protocol_saasbo_v1",
        "final_protocol_sebo_v1",
    )
    for group in groups:
        for domain, count in CATEGORY_COUNTS.items():
            payload = load(RESULTS / group / domain / "raw.json")
            assert payload["config"]["n_seeds"] == 10
            assert payload["config"]["n_ingredients"] == (106 if domain == "meat" else 90)
            assert len(payload["results"]) == count
            for result in payload["results"].values():
                assert len(result["selected_ingredients"]) <= 9
                for traces in result["traces"].values():
                    assert len(traces) == 10
                    assert_finite_trajectory(traces)


def test_egbo_result_shards_are_complete():
    groups = {
        "final_protocol_egbo_default_v1": "EGBO (default BO)",
        "final_protocol_egbo_llm_config_v1": "EGBO (LLM-configured BO)",
    }
    for group, method in groups.items():
        for domain, count in CATEGORY_COUNTS.items():
            raw_paths = sorted((RESULTS / group / domain).glob("*/raw.json"))
            assert len(raw_paths) == count
            for path in raw_paths:
                payload = load(path)
                assert payload["config"]["status"] == "complete"
                assert payload["config"]["bo_seeds"] == 10
                assert len(payload["config"]["shared_support"]) <= 9
                assert len(payload["methods"][method]) == 10
                for record in payload["methods"][method]:
                    assert_finite_trajectory(record["best_values"])


def test_llm_and_adaptive_records_are_complete():
    sources = {
        "meat": RESULTS / "meat_semantic_all14_v1",
        "dairy": RESULTS / "dairy_semantic_all10_v3",
    }
    for domain, root in sources.items():
        paths = sorted(root.glob("*/raw.json"))
        assert len(paths) == CATEGORY_COUNTS[domain]
        for path in paths:
            payload = load(path)
            assert payload["config"]["status"] == "complete"
            for method in ("Restricted iterative LLM", "Unrestricted iterative LLM"):
                records = payload["methods"][method]
                assert len(records) == 1
                assert_finite_trajectory(records[0]["best_values"])

        adaptive_paths = sorted(
            (RESULTS / "final_protocol_adaptive_q2_v1" / domain).glob("*/raw.json")
        )
        assert len(adaptive_paths) == CATEGORY_COUNTS[domain]
        for path in adaptive_paths:
            payload = load(path)
            assert payload["config"]["status"] == "complete"
            records = payload["methods"]["Adaptive semantic-only EGBO (Q=2)"]
            assert len(records) == 10
            for record in records:
                assert_finite_trajectory(record["best_values"])


def test_release_records_do_not_contain_local_absolute_paths():
    forbidden = ("/Users/", "GoogleDrive", "OneDrive", "Dropbox")
    for path in RESULTS.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path
