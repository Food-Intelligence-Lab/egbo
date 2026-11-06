#!/usr/bin/env python3
"""Verify the camera-ready paper's principal claims from released records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
FAILURES: list[str] = []


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_close(label: str, actual: float, expected: float, tolerance: float) -> None:
    ok = abs(actual - expected) <= tolerance
    status = "PASS" if ok else "FAIL"
    print(
        f"{status}  {label}: {actual:.4f} "
        f"(expected {expected:.4f} +/- {tolerance:g})"
    )
    if not ok:
        FAILURES.append(f"{label}: {actual:.4f} vs {expected:.4f}")


def check_equal(label: str, actual, expected) -> None:
    ok = actual == expected
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {actual} (expected {expected})")
    if not ok:
        FAILURES.append(f"{label}: {actual!r} vs {expected!r}")


def hartmann_checks() -> None:
    base = ROOT / "hartmann6" / "results"
    main = read_json(base / "release_v1_10seed" / "oneshot" / "raw.json")
    finals = {name: values["mean_trace"][-1] for name, values in main["results"].items()}
    print("\nHartmann6")
    check_close("one-shot oracle value", finals["oracle"], 2.95, 0.05)
    check_close("one-shot strong-EGBO value", finals["egbo_strong"], 2.54, 0.05)
    check_close("one-shot weak-EGBO value", finals["egbo_weak"], 0.18, 0.05)
    q1 = read_json(base / "revision_v1__qsweep_q1" / "raw.json")
    q20 = read_json(base / "revision_v1__qsweep_q20" / "raw.json")
    optimum = float(q1["config"]["global_max"])
    check_close(
        "weak-expert regret Q=1",
        optimum - q1["results"]["egbo_weak"]["mean_final"],
        3.20,
        0.01,
    )
    check_close(
        "weak-expert regret Q=16 (Q=20 run; identical)",
        optimum - q20["results"]["egbo_weak"]["mean_final"],
        2.68,
        0.01,
    )


def _core_payload(domain: str):
    return read_json(
        ROOT
        / "formulateBench"
        / "results"
        / "final_protocol_core_v1"
        / domain
        / "raw.json"
    )


def _random_normalizers(domain: str) -> tuple[list[str], dict[str, float]]:
    payload = _core_payload(domain)
    categories = list(payload["results"])
    normalizers = {}
    for category, result in payload["results"].items():
        label = next(label for label in result["traces"] if "Random Search" in label)
        traces = -np.asarray(result["traces"][label], dtype=float)
        normalizers[category] = float(traces[:, 0].mean())
    return categories, normalizers


def _suite_records(root: Path, method: str) -> dict[str, np.ndarray]:
    records = {}
    for raw_path in sorted(root.glob("*/raw.json")):
        payload = read_json(raw_path)
        category = payload["config"]["category"]
        values = payload["methods"][method]
        records[category] = -np.asarray(
            [record["best_values"] for record in values], dtype=float
        )
    return records


def _normalized_category_curves(
    records: dict[str, np.ndarray],
    categories: list[str],
    normalizers: dict[str, float],
) -> dict[str, np.ndarray]:
    return {
        category: records[category].mean(axis=0) / normalizers[category]
        for category in categories
    }


def formulatebench_checks() -> None:
    base = ROOT / "formulateBench" / "results"
    expected_means = {
        "meat": {"egbo10": 0.46, "llm10": 0.58, "egbo20": 0.35, "llm20": 0.52},
        "dairy": {"egbo10": 0.45, "llm10": 0.44, "egbo20": 0.36, "llm20": 0.41},
    }
    print("\nFormulateBench")
    pooled_counts = {"fixed10": 0, "fixed20": 0, "free10": 0, "free20": 0}
    for domain in ("meat", "dairy"):
        categories, normalizers = _random_normalizers(domain)
        check_equal(f"{domain} category count", len(categories), 14 if domain == "meat" else 10)
        check_equal(
            f"{domain} filtered dimension",
            int(_core_payload(domain)["config"]["n_ingredients"]),
            106 if domain == "meat" else 90,
        )

        egbo = _suite_records(
            base / "final_protocol_egbo_llm_config_v1" / domain,
            "EGBO (LLM-configured BO)",
        )
        llm_root = base / ("meat_semantic_all14_v1" if domain == "meat" else "dairy_semantic_all10_v3")
        unrestricted = _suite_records(llm_root, "Unrestricted iterative LLM")
        fixed = _suite_records(llm_root, "Restricted iterative LLM")

        egbo_curves = _normalized_category_curves(egbo, categories, normalizers)
        unrestricted_curves = _normalized_category_curves(
            unrestricted, categories, normalizers
        )
        fixed_curves = _normalized_category_curves(fixed, categories, normalizers)

        means = expected_means[domain]
        check_close(
            f"{domain} EGBO normalized distance at evaluation 10",
            float(np.mean([egbo_curves[c][9] for c in categories])),
            means["egbo10"],
            0.01,
        )
        check_close(
            f"{domain} unrestricted LLM at evaluation 10",
            float(np.mean([unrestricted_curves[c][9] for c in categories])),
            means["llm10"],
            0.01,
        )
        check_close(
            f"{domain} EGBO normalized distance at evaluation 20",
            float(np.mean([egbo_curves[c][19] for c in categories])),
            means["egbo20"],
            0.01,
        )
        check_close(
            f"{domain} unrestricted LLM at evaluation 20",
            float(np.mean([unrestricted_curves[c][19] for c in categories])),
            means["llm20"],
            0.01,
        )

        wins = {
            "fixed10": sum(egbo_curves[c][9] < fixed_curves[c][9] for c in categories),
            "fixed20": sum(egbo_curves[c][19] < fixed_curves[c][19] for c in categories),
            "free10": sum(
                egbo_curves[c][9] < unrestricted_curves[c][9] for c in categories
            ),
            "free20": sum(
                egbo_curves[c][19] < unrestricted_curves[c][19] for c in categories
            ),
        }
        for key, value in wins.items():
            print(f"INFO  {domain} category wins ({key}): {value}/{len(categories)}")
            pooled_counts[key] += value

    for key, expected in {"fixed10": 18, "fixed20": 23, "free10": 16, "free20": 20}.items():
        check_equal(f"pooled category wins ({key})", pooled_counts[key], expected)

    adaptive = read_json(
        base / "final_protocol_figures_v1" / "formulatebench_adaptive_q2.json"
    )
    for domain, expected_wins_value in (("meat", 4), ("dairy", 2)):
        check_equal(
            f"adaptive EGBO category wins ({domain})",
            adaptive[domain]["adaptive_category_wins"],
            expected_wins_value,
        )
        if adaptive[domain]["adaptive_eval20"] <= adaptive[domain]["one_shot_eval20"]:
            FAILURES.append(f"adaptive EGBO unexpectedly beats one-shot on {domain}")
            print(f"FAIL  adaptive EGBO does not improve on average ({domain})")
        else:
            print(f"PASS  adaptive EGBO does not improve on average ({domain})")


def regime_checks() -> None:
    """Regime analysis (formulateBench/analyze_regimes.py) from cached records."""
    import contextlib
    import importlib.util
    import io
    import sys

    from scipy.stats import spearmanr

    print("\nFormulateBench regime analysis")
    path = ROOT / "formulateBench" / "analyze_regimes.py"
    spec = importlib.util.spec_from_file_location("analyze_regimes", path)
    regimes = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("analyze_regimes", regimes)
    spec.loader.exec_module(regimes)
    with contextlib.redirect_stdout(io.StringIO()):
        table = regimes.build_table(regimes.build_rows(ROOT / "formulateBench" / "results"))
    check_equal("regime table category count", len(table), 24)
    for evaluation, expected in ((5, (14, 5, 5)), (10, (11, 2, 11)), (20, (4, 2, 18))):
        winners = [row[f"winner{evaluation}"] for row in table]
        check_equal(
            f"best family at evaluation {evaluation} (LLM/BO/EGBO)",
            tuple(winners.count(k) for k in ("LLM", "BO", "EGBO")),
            expected,
        )
    for x_key, y_key, rho_expected, p_expected in (
        ("init_gap", "adv_llm20", 0.54, 0.007),
        ("sup_gap", "adv_bo20", -0.51, 0.011),
    ):
        rho, pval = spearmanr([r[x_key] for r in table], [r[y_key] for r in table])
        check_close(f"Spearman rho {x_key} vs {y_key}", float(rho), rho_expected, 0.005)
        check_close(f"Spearman p {x_key} vs {y_key}", float(pval), p_expected, 0.0005)
    for arm, name, expected in (("llm_fixed", "fixed-support", (18, 23)),
                                ("llm_unres", "unrestricted", (16, 20))):
        wins = tuple(int(sum(r[f"egbo{e}"] < r[f"{arm}{e}"] for r in table)) for e in (10, 20))
        check_equal(f"regime table: EGBO beats {name} LLM (eval 10, eval 20)", wins, expected)


def _noise_run_files() -> list[Path]:
    sweep = ROOT / "formulateBench" / "results" / "noise_sweep_v1"
    return sorted(p for domain in ("meat", "dairy") for p in (sweep / domain).glob("*/*.json"))


def _noise_curves(field: str) -> dict:
    """curves[level][arm][(domain, category)] -> mean normalized curve, with
    the noiseless paper records as level 0 (as in analyze_noise_sweep.py)."""
    base = ROOT / "formulateBench" / "results"
    semantic = {"meat": base / "meat_semantic_all14_v1", "dairy": base / "dairy_semantic_all10_v3"}
    curves: dict = {}
    for domain in ("meat", "dairy"):
        core = read_json(base / "final_protocol_core_v1" / domain / "summary.json")
        for category, entry in core.items():
            rs = next(k for k in entry if k.startswith("Random Search"))
            norm = float(np.mean(-np.asarray(entry[rs])[:, 0]))
            folder = category.replace(" ", "_")
            sem = read_json(semantic[domain] / folder / "raw.json")["methods"]
            egbo = read_json(
                base / "final_protocol_egbo_llm_config_v1" / domain / folder / "raw.json"
            )["methods"]
            for arm, records in (("llm_unres", sem["Unrestricted iterative LLM"]),
                                 ("llm_fixed", sem["Restricted iterative LLM"]),
                                 ("egbo", egbo["EGBO (LLM-configured BO)"])):
                curve = (-np.asarray([r["best_values"] for r in records], dtype=float)).mean(0) / norm
                curves.setdefault(0.0, {}).setdefault(arm, {})[(domain, category)] = curve
    acc: dict = {}
    for path in _noise_run_files():
        d = read_json(path)
        key = (d["level"], d["arm"], d["domain"], d["category"])
        acc.setdefault(key, []).append(np.asarray(d[field], dtype=float) / d["norm"])
    for (level, arm, domain, category), values in acc.items():
        curves.setdefault(level, {}).setdefault(arm, {})[(domain, category)] = np.mean(values, 0)
    return curves


def noise_checks() -> None:
    """Observation-noise sweep (formulateBench/analyze_noise_sweep.py)."""
    from scipy.stats import wilcoxon

    print("\nFormulateBench observation-noise sweep")
    n_files = len(_noise_run_files())
    check_equal("noise-sweep run files", n_files, 768)
    curves = _noise_curves("recommended_true_loss")
    check_equal("noise levels", sorted(curves), [0.0, 0.05, 0.15])
    expected_wins = {0.0: 20, 0.05: 21, 0.15: 19}
    expected_improvement = {
        "llm_unres": {0.0: 0.139, 0.05: 0.052, 0.15: 0.000},
        "egbo": {0.0: 0.302, 0.05: 0.262, 0.15: 0.199},
    }
    best_true = _noise_curves("best_true_so_far")
    for level in sorted(curves):
        keys = sorted(curves[level]["egbo"])
        check_equal(f"noise {level:g}: complete categories", len(keys), 24)

        def best_llm(cv, k, e):
            return min(cv[level]["llm_unres"][k][e - 1], cv[level]["llm_fixed"][k][e - 1])

        llm20 = np.array([best_llm(curves, k, 20) for k in keys])
        egbo20 = np.array([curves[level]["egbo"][k][19] for k in keys])
        check_equal(f"noise {level:g}: EGBO beats best LLM arm at eval 20",
                    int((egbo20 < llm20).sum()), expected_wins[level])
        for e in (5, 20):
            diff = np.array([best_llm(curves, k, e) - curves[level]["egbo"][k][e - 1] for k in keys])
            print(f"INFO  noise {level:g}: Wilcoxon p (best LLM - EGBO) at eval {e}: "
                  f"{wilcoxon(diff).pvalue:.2g}")
        diff20 = llm20 - egbo20
        p20 = float(wilcoxon(diff20).pvalue)
        ok = p20 < 0.01 and np.median(diff20) > 0
        print(f"{'PASS' if ok else 'FAIL'}  noise {level:g}: EGBO better than best LLM at "
              f"eval 20 (Wilcoxon p = {p20:.2g} < 0.01)")
        if not ok:
            FAILURES.append(f"noise {level:g} eval-20 Wilcoxon p = {p20:.2g}")
        for arm, expected in expected_improvement.items():
            m = np.array([curves[level][arm][k] for k in keys])
            check_close(f"noise {level:g}: {arm} mean improvement eval 5 -> 20",
                        float(m[:, 4].mean() - m[:, 19].mean()), expected[level], 0.002)
        bt_llm = np.array([best_llm(best_true, k, 20) for k in keys])
        bt_egbo = np.array([best_true[level]["egbo"][k][19] for k in keys])
        print(f"INFO  noise {level:g}: best-true-ever-evaluated variant, EGBO beats best LLM "
              f"at eval 20 in {int((bt_egbo < bt_llm).sum())}/24")


def _wilcoxon_p(diff) -> float:
    from scipy.stats import wilcoxon
    return float(wilcoxon(diff).pvalue)


def noise_aware_checks() -> None:
    """Iterative LLM told the noise level (Appendix F.6, 'Telling the LLM the noise level')."""
    print("\nNoise-aware prompt arm")
    res = ROOT / "formulateBench" / "results"

    def load(root, arms, seed0):
        acc = {}
        for path in (res / root).glob("*/*/*.json"):
            d = read_json(path)
            if d.get("level") != 0.15 or d["arm"] not in arms or (seed0 and d["seed"] != 0):
                continue
            acc.setdefault((d["arm"], d["domain"], d["category"]), []).append(d["recommended_true_loss"][19] / d["norm"])
        return {k: float(np.mean(v)) for k, v in acc.items()}

    for name, base, aware, exp in (
            ("Claude Opus 4.7", "noise_sweep_v1", "noise_aware_opus_v1", ((0.68, 7), (0.70, 6))),
            ("GPT-6.1 Sol", "noise_sweep_gpt61sol_v1", "noise_aware_gpt61sol_v1", ((0.568, 6), (0.575, 4)))):
        egbo = load(base, {"egbo"}, False)
        for label, root, (e_mean, e_ties) in (("without note", base, exp[0]), ("with note", aware, exp[1])):
            llm = load(root, {"llm_unres", "llm_fixed"}, True)
            cats = sorted({k[1:] for k in egbo})
            b = np.array([min(llm[("llm_unres", *c)], llm[("llm_fixed", *c)]) for c in cats])
            e = np.array([egbo[("egbo", *c)] for c in cats])
            check_close(f"noise-aware {name} {label}: better LLM arm", float(b.mean()), e_mean, 0.005)
            check_equal(f"noise-aware {name} {label}: LLM ties or beats EGBO", int((b <= e).sum()), e_ties)
    own = read_json(res / "noise_sweep_v1" / "own_pick_summary.json")["0.15"]
    check_equal("own-pick scoring, noise 0.15: EGBO better at eval 20", own["egbo_beats_best_llm_e20"], 20)
    check_close("own-pick scoring, noise 0.15: EGBO own pick at eval 20", own["egbo"]["own"][2], 0.489, 0.0005)


def batched_checks() -> None:
    """Batched FormulateBench, 10 batches of 3 (main-text Fig. 2, Table 6)."""
    from collections import defaultdict
    root = ROOT / "formulateBench" / "results" / "batched_r10_v1"
    print("\nFormulateBench batched (10 x 3)")
    acc = defaultdict(list)
    first = defaultdict(list)
    div = defaultdict(list)
    for path in root.glob("*/*/*/*.json"):
        if "llm_cache" in path.parts:
            continue
        d = read_json(path)
        model = "baselines" if "baselines" in path.parts else d["model"]
        key = (model, d["level"], d["arm"], d["domain"], d["category"])
        rec = np.asarray(d["recommended_true_loss"]) / d["norm"]
        acc[key].append(rec[-1])
        first[key].append(rec[5])  # after batch 2
        if d["arm"] != "random":
            pairs = []
            for b in d["batches"][1:]:
                recs = [d["recipes"][i] for i in b]
                names = sorted(set().union(*recs))
                v = np.array([[r.get(n, 0.0) for n in names] for r in recs])
                pairs.append(np.mean([np.abs(v[i] - v[j]).sum() for i in range(3) for j in range(i + 1, 3)]))
            div[key].append(float(np.mean(pairs)))
    mean = {k: float(np.mean(v)) for k, v in acc.items()}
    # Table 6: (better LLM arm per category, EGBO, EGBO wins)
    expected = {
        ("anthropic/claude-opus-4.7", 0.0): (0.44, 0.34, 19),
        ("openai/gpt-6.1-sol", 0.0): (0.31, 0.32, 9),
        ("google/gemini-4-argon", 0.0): (0.32, 0.35, 10),
        ("anthropic/claude-opus-4.7", 0.15): (0.68, 0.48, 19),
        ("openai/gpt-6.1-sol", 0.15): (0.56, 0.47, 18),
        ("google/gemini-4-argon", 0.15): (0.58, 0.50, 15),
    }
    for (model, level), (e_llm, e_egbo, e_wins) in expected.items():
        cats = sorted({k[3:] for k in mean if k[0] == model and k[1] == level and k[2] == "egbo"})
        check_equal(f"batched {model} noise {level:g}: categories", len(cats), 24)
        llm = np.array([min(mean[(model, level, "llm_unres", *c)], mean[(model, level, "llm_fixed", *c)]) for c in cats])
        egbo = np.array([mean[(model, level, "egbo", *c)] for c in cats])
        check_close(f"batched {model} noise {level:g}: better LLM arm", float(llm.mean()), e_llm, 0.005)
        check_close(f"batched {model} noise {level:g}: EGBO", float(egbo.mean()), e_egbo, 0.005)
        check_equal(f"batched {model} noise {level:g}: EGBO better (categories)", int((egbo < llm).sum()), e_wins)
        print(f"INFO  batched {model} noise {level:g}: Wilcoxon p = {_wilcoxon_p(llm - egbo):.2g}")
    opus = "anthropic/claude-opus-4.7"
    cats = sorted({k[3:] for k in mean if k[0] == opus and k[2] == "egbo"})
    for level, e_vbo, e_tur, e_vw, e_tw in ((0.0, 0.52, 0.51, 20, 20), (0.15, 0.70, 0.73, 21, 22)):
        egbo = np.array([mean[(opus, level, "egbo", *c)] for c in cats])
        vbo = np.array([mean[("baselines", level, "vanilla_bo", *c)] for c in cats])
        tur = np.array([mean[("baselines", level, "turbo", *c)] for c in cats])
        check_close(f"batched noise {level:g}: vanilla BO", float(vbo.mean()), e_vbo, 0.005)
        check_close(f"batched noise {level:g}: TuRBO", float(tur.mean()), e_tur, 0.005)
        check_equal(f"batched noise {level:g}: EGBO (Opus) beats vanilla BO", int((egbo < vbo).sum()), e_vw)
        check_equal(f"batched noise {level:g}: EGBO (Opus) beats TuRBO", int((egbo < tur).sum()), e_tw)
    # "both iterative LLM arms stop improving after the first batch" (noise, Opus)
    for arm in ("llm_unres", "llm_fixed"):
        late = np.mean([np.mean(first[(opus, 0.15, arm, *c)]) - mean[(opus, 0.15, arm, *c)] for c in cats])
        ok = abs(late) < 0.03
        print(f"{'PASS' if ok else 'FAIL'}  batched noise 0.15: Opus {arm} improvement batch 2 -> 10 = {late:.3f} (< 0.03)")
        if not ok:
            FAILURES.append(f"batched Opus {arm} still improving under noise ({late:.3f})")
    llm_div = [np.mean([np.mean(div[(opus, lvl, arm, *c)]) for c in cats]) for lvl in (0.0, 0.15) for arm in ("llm_unres", "llm_fixed")]
    egbo_div = [np.mean([np.mean(div[(opus, lvl, "egbo", *c)]) for c in cats]) for lvl in (0.0, 0.15)]
    ok = 0.035 <= min(llm_div) and max(llm_div) <= 0.085 and 0.42 <= min(egbo_div) and max(egbo_div) <= 0.475
    print(f"{'PASS' if ok else 'FAIL'}  batched batch diversity (Opus): LLM {min(llm_div):.3f}-{max(llm_div):.3f} "
          f"(paper 0.04-0.08), EGBO {min(egbo_div):.3f}-{max(egbo_div):.3f} (paper 0.43-0.47)")
    if not ok:
        FAILURES.append("batched batch-diversity range")


def crossmodel_checks() -> None:
    """Other LLMs, one recipe at a time (Table 7, noiseless columns)."""
    import sys
    sys.path.insert(0, str(ROOT / "formulateBench"))
    import analyze_multi_llm as aml
    print("\nFormulateBench other LLMs (one recipe at a time)")
    models = {"anthropic/claude-opus-4.7": read_json(ROOT / "formulateBench" / "results" / "regime_analysis_v1" / "regime_table.json")}
    models.update(aml.load_new(ROOT / "formulateBench" / "results" / "multi_llm_v1", aml.norms()))
    expected = {"anthropic/claude-opus-4.7": (0.42, 0.36, 20), "openai/gpt-6.1-sol": (0.37, 0.37, 13),
                "google/gemini-4-argon": (0.34, 0.39, 9)}
    errors = {"anthropic/claude-opus-4.7": (0.54, 0.21), "openai/gpt-6.1-sol": (0.39, 0.18),
              "google/gemini-4-argon": (0.71, 0.17)}
    for model, (e_conc, e_ingr) in errors.items():
        rows = models[model]
        conc = float(np.mean([r["init_gap"] for r in rows]))
        ingr = float(np.mean([r["sup_gap"] for r in rows]))
        check_close(f"other LLMs {model}: concentration error", conc, e_conc, 0.005)
        check_close(f"other LLMs {model}: ingredient-choice error", ingr, e_ingr, 0.005)
        ok = conc > ingr
        print(f"{'PASS' if ok else 'FAIL'}  other LLMs {model}: concentration error exceeds ingredient-choice error")
        if not ok:
            FAILURES.append(f"{model}: concentration error not larger")
    for model, (e_llm, e_egbo, e_wins) in expected.items():
        rows = models[model]
        llm = np.array([min(r["llm_unres20"], r["llm_fixed20"]) for r in rows])
        egbo = np.array([r["egbo20"] for r in rows])
        check_close(f"other LLMs {model}: better LLM arm (eval 20)", float(llm.mean()), e_llm, 0.005)
        check_close(f"other LLMs {model}: EGBO (eval 20)", float(egbo.mean()), e_egbo, 0.005)
        check_equal(f"other LLMs {model}: EGBO better (categories)", int((egbo < llm).sum()), e_wins)


def agent_checks() -> None:
    """Tool-using-agent test (Table 8, Appendix F.9)."""
    import re
    print("\nTool-using agents")
    gp = re.compile(r"GaussianProcess|gaussian_process|botorch|gpytorch|skopt|bayes_opt|optuna", re.I)
    model_rx = re.compile(r"scipy\.optimize|from scipy import optimize|minimize\(|differential_evolution|nelder|"
                          r"LinearRegression|Ridge|Lasso|RandomForest|GradientBoosting|polyfit|lstsq|PolynomialFeatures|"
                          r"KernelRidge|SVR|MLPRegressor|curve_fit", re.I)
    root = ROOT / "formulateBench" / "results" / "agent_pilot_v1"
    # (no code, any model, GP, agent distance)
    expected = {
        ("anthropic__claude-opus-4.7", "agent"): (0, 0, 0, 0.58), ("anthropic__claude-opus-4.7", "agent_hint"): (0, 0, 0, 0.59),
        ("openai__gpt-6.1-sol", "agent"): (20, 4, 1, 0.48), ("openai__gpt-6.1-sol", "agent_hint"): (15, 7, 4, 0.46),
        ("openai__gpt-6.1-sol", "agent_botorch"): (12, 10, 6, 0.51),
        ("google__gemini-4-argon", "agent"): (0, 24, 20, 0.66), ("google__gemini-4-argon", "agent_hint"): (1, 23, 22, 0.66),
    }
    botorch_imports = 0
    for (slug, arm), (e_none, e_any, e_gp, e_dist) in expected.items():
        runs = [read_json(p) for p in root.glob(f"{slug}/*/*/{arm}_noise0.15_seed0.json")]
        check_equal(f"agents {slug} {arm}: runs", len(runs), 24)
        code = ["\n".join(r["python_code"]) for r in runs]
        n_none = sum(not r["python_code"] for r in runs)
        n_gp = sum(bool(gp.search(c)) for c in code)
        n_any = sum(bool(gp.search(c) or model_rx.search(c)) for c in code)
        botorch_imports += sum(bool(re.search(r"botorch", c, re.I)) for c in code)
        check_equal(f"agents {slug} {arm}: no code / any model / GP", (n_none, n_any, n_gp), (e_none, e_any, e_gp))
        dist = float(np.mean([r["recommended_true_loss"][-1] / r["norm"] for r in runs]))
        check_close(f"agents {slug} {arm}: distance at eval 20", dist, e_dist, 0.006)
    check_equal("agents: runs importing BoTorch", botorch_imports, 0)
    # Agents: Claude Opus 4.7, GPT-6.1 Sol, and Gemini 4 Argon.
    reported = ("anthropic__claude-opus-4.7", "openai__gpt-6.1-sol", "google__gemini-4-argon")
    all_runs = [read_json(p) for slug in reported for p in root.glob(f"{slug}/*/*/agent*_noise0.15_seed0.json")
                if "transcript" not in p.name]
    no_model = sum(not (gp.search("\n".join(r["python_code"])) or model_rx.search("\n".join(r["python_code"]))) for r in all_runs)
    check_equal("agents: runs that fit no model of their data (of all runs)", (no_model, len(all_runs)), (100, 168))
    # "No agent significantly outperformed EGBO" (Section 4.2): one-sided check per arm
    from scipy.stats import wilcoxon
    res = ROOT / "formulateBench" / "results"
    egbo_roots = {"anthropic__claude-opus-4.7": "noise_sweep_v1", "openai__gpt-6.1-sol": "noise_sweep_gpt61sol_v1",
                  "google__gemini-4-argon": "noise_sweep_gemini4argon_v1"}
    worst = []
    for slug, eroot in egbo_roots.items():
        eg = {}
        for path in (res / eroot).glob("*/*/egbo_noise0.15_seed*.json"):
            d = read_json(path)
            eg.setdefault((d["domain"], d["category"]), []).append(d["recommended_true_loss"][19] / d["norm"])
        for arm in ("agent", "agent_hint", "agent_botorch"):
            rs = [read_json(p) for p in root.glob(f"{slug}/*/*/{arm}_noise0.15_seed0.json")]
            if not rs:
                continue
            a_ = np.array([r["recommended_true_loss"][-1] / r["norm"] for r in rs])
            e_ = np.array([np.mean(eg[(r["domain"], r["category"])]) for r in rs])
            p_ = float(wilcoxon(e_ - a_, alternative="greater").pvalue)
            worst.append((p_, slug, arm))
    ok = min(worst)[0] >= 0.05
    print(f"{'PASS' if ok else 'FAIL'}  agents: no agent arm significantly better than EGBO "
          f"(smallest one-sided p = {min(worst)[0]:.2g}, {min(worst)[1]} {min(worst)[2]})")
    if not ok:
        FAILURES.append("an agent arm significantly outperforms EGBO")


def _csv_max(path: Path, column: str) -> float:
    with path.open(newline="", encoding="utf-8") as stream:
        return max(float(row[column]) for row in csv.DictReader(stream))


def deployment_checks() -> None:
    data = ROOT / "deployment" / "data"
    print("\nDeployment")
    check_close(
        "yogurt best utility",
        _csv_max(data / "yogurt_utility_per_iteration.csv", "Best_So_Far"),
        0.992,
        0.001,
    )
    check_close(
        "cottage-cheese best utility",
        _csv_max(data / "cottage_utility_per_iteration.csv", "Best_So_Far"),
        0.975,
        0.001,
    )
    check_close(
        "HFS from-scratch best utility",
        _csv_max(data / "leg1_data.csv", "utility"),
        0.850,
        0.001,
    )
    check_close(
        "HFS same-start best utility",
        _csv_max(data / "leg2_data.csv", "Best_So_Far"),
        0.867,
        0.001,
    )


# ---------------------------------------------------------------------------
# Data-consistency audit
#
# The checks above confirm headline values. The checks below confirm that the
# released records are internally consistent with the paper's definitions
# (Eq. 7 utility, best-recipe identity, Q=T schedules, heatmap reading).
# They report WARN by default and only fail with --consistency-strict.
# ---------------------------------------------------------------------------

WARNINGS: list[str] = []
S_MAX = 15.0


def warn_or_pass(label: str, ok: bool, detail: str) -> None:
    print(f"{'PASS' if ok else 'WARN'}  {label}: {detail}")
    if not ok:
        WARNINGS.append(f"{label}: {detail}")


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _eq7_utilities(sensory_csv: Path, attributes: list[str]) -> dict[str, float]:
    """Uniform-weight utility of Eq. (7): 1 - mean_j |s_j - t_j| / S_max."""
    out = {}
    for row in _read_csv(sensory_csv):
        errs = [abs(float(row[a]) - float(row[f"{a}_target"])) for a in attributes]
        out[row["Batch"]] = 1.0 - sum(errs) / (len(errs) * S_MAX)
    return out


def deployment_consistency() -> None:
    data = ROOT / "deployment" / "data"
    print("\nDeployment consistency (Eq. 7, S_max = 15, weights as entered in the app)")
    # Attribute weights as entered in the deployment app (food_bo.py does not
    # renormalize them). Yogurt used 1/4 each; cottage cheese used 0.17 each
    # (sum 1.02), as disclosed in the paper's Section 5 footnote.
    campaigns = {
        "yogurt": (["Consistency", "Creaminess", "Tangy", "Thickness_Uniformity"], 0.25),
        "cottage": (["Homogeneity", "Dispersion Phase", "Slipperiness", "Sour", "Dairy", "Off-notes"], 0.17),
    }
    for name, (attrs, weight) in campaigns.items():
        eq7 = _eq7_utilities(data / f"{name}_sensory_data.csv", attrs)
        entered = {b: weight * len(attrs) * u for b, u in eq7.items()}
        recorded = {
            r["Batch"]: float(r["Total_Utility_Score"])
            for r in _read_csv(data / f"{name}_utility_per_recipe.csv")
        }
        worst = max(abs(entered[b] - recorded[b]) for b in recorded)
        warn_or_pass(
            f"{name} recorded utilities = Eq. 7 with w_j = {weight}",
            worst < 1e-3,
            f"max |computed - recorded| = {worst:.4f}",
        )
        best_eq7 = max(eq7, key=eq7.get)
        best_rec = max(recorded, key=recorded.get)
        print(
            f"INFO  {name} best recipe: recorded {best_rec} ({recorded[best_rec]:.4f}); "
            f"Eq. 7 with w_j = 1/m: {best_eq7} ({eq7[best_eq7]:.4f})"
        )
        if name == "yogurt":
            warn_or_pass(
                "yogurt best recipe is 7.3 (Table 5, App. G.4)",
                best_rec == "7.3",
                f"records say {best_rec} ({recorded[best_rec]:.4f})",
            )
        if name == "cottage":
            warn_or_pass(
                "cottage best recipe is 10.1 (Table 7, App. G.5)",
                best_rec == "10.1",
                f"records say {best_rec} ({recorded[best_rec]:.4f})",
            )
            warn_or_pass(
                "cottage gain under w_j = 1/6 matches the reported 26%",
                round(100 * (eq7[best_eq7] / eq7["1.1"] - 1)) == 26,
                f"Eq. 7 gives baseline {eq7['1.1']:.4f}, best {eq7[best_eq7]:.4f} "
                f"(gain {eq7[best_eq7] / eq7['1.1'] - 1:.1%})",
            )
            per_iter = {
                int(r["Iteration"]): float(r["Best_So_Far"])
                for r in _read_csv(data / "cottage_utility_per_iteration.csv")
            }
            warn_or_pass(
                "cottage best-so-far at iteration 4 is reported as 0.895 (App. G.5)",
                abs(per_iter[4] - 0.895) < 1e-3,
                f"iteration 4 best-so-far = {per_iter[4]:.4f}",
            )


def hartmann_consistency() -> None:
    base = ROOT / "hartmann6" / "results"
    print("\nHartmann6 consistency")
    adaptive = read_json(base / "release_v1_10seed" / "adaptive" / "raw.json")
    q20 = read_json(base / "revision_v1__qsweep_q20" / "raw.json")
    optimum = float(q20["config"]["global_max"])
    for method in ("egbo_strong", "egbo_weak"):
        fig1 = np.asarray(adaptive["results"][method]["all_traces"], dtype=float)[:, -1]
        fig7 = np.asarray(q20["results"][method]["all_traces"], dtype=float)[:10, -1]
        # The paper (App. E.4) states that the two adaptive schedules differ:
        # Fig. 1(b) queries before all 20 evaluations (adaptive=True), the
        # Q-sweep before the initial design and each BO step (max 16 rounds).
        warn_or_pass(
            f"{method}: Fig. 1(b) 'Q=T' and Fig. 7 'Q=16' schedules differ on shared seeds 0-9 (App. E.4)",
            not bool(np.allclose(fig1, fig7, atol=1e-6)),
            f"regret {optimum - fig1.mean():.3f} vs {optimum - fig7.mean():.3f}",
        )

    baseline = read_json(base / "revision_v1__pq_baseline" / "raw.json")["results"]
    vanilla = float(np.asarray(baseline["vanilla"]["all_traces"])[:, -1].mean())
    oracle = float(np.asarray(baseline["oracle"]["all_traces"])[:, -1].mean())
    grid = {}
    for path in sorted(base.glob("revision_v1__pq_p*_q*/raw.json")):
        payload = read_json(path)
        p, q = payload["config"]["expert_p"], payload["config"]["expert_q"]
        final = float(np.asarray(payload["results"]["egbo_custom"]["all_traces"])[:, -1].mean())
        grid[(p, q)] = (final - vanilla) / (oracle - vanilla)
    ps = sorted({k[0] for k in grid})
    qs = sorted({k[1] for k in grid})
    print(f"INFO  heatmap grid p = {ps}, q_fp = {qs}")
    print("INFO  gap-closed fraction (rows q_fp, columns p):")
    for q in qs:
        print("        q=%-5g " % q + " ".join(f"{grid[(p, q)]:+.2f}" for p in ps))
    # Claims as stated in the Fig. 6 caption.
    major = [(p, q) for (p, q) in grid if (q == 0 and p >= 0.4) or (q == 0.01 and p >= 0.8)]
    failing = [k for k in major if grid[k] <= 0.5]
    warn_or_pass(
        "majority of gap closed for p >= 0.4 at q_fp = 0 and p >= 0.8 at q_fp = 0.01 (Fig. 6 caption)",
        not failing, f"{len(major)} cells checked; failing: {sorted(failing)}",
    )
    high_q = [v for (p, q), v in grid.items() if q >= 0.05]
    warn_or_pass(
        "at q_fp >= 0.05 at most 38% of the gap closed (Fig. 6 caption)",
        max(high_q) <= 0.385, f"max {max(high_q):.4f}",
    )
    low_p = [v for (p, q), v in grid.items() if p <= 0.2]
    warn_or_pass(
        "at p <= 0.2 EGBO underperforms vanilla BO for every q_fp (Fig. 6 caption)",
        max(low_p) < 0, f"max gap-closed {max(low_p):+.3f}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if a headline claim fails")
    parser.add_argument("--consistency-strict", action="store_true",
                        help="also exit non-zero if a data-consistency check warns")
    args = parser.parse_args()
    hartmann_checks()
    formulatebench_checks()
    regime_checks()
    noise_checks()
    noise_aware_checks()
    batched_checks()
    crossmodel_checks()
    agent_checks()
    deployment_checks()
    deployment_consistency()
    hartmann_consistency()
    if WARNINGS:
        print("\nData-consistency warnings (records vs. paper text):")
        for warning in WARNINGS:
            print(f"WARN  {warning}")
    consistency_fail = bool(WARNINGS) and args.consistency_strict
    if FAILURES:
        print("\nVerification failures:")
        for failure in FAILURES:
            print(f"FAIL  {failure}")
        if args.strict:
            raise SystemExit(1)
    else:
        print("\nAll checked headline claims match the released records.")
        if WARNINGS:
            print(f"{len(WARNINGS)} data-consistency warning(s) above need attention "
                  "in the paper text (run with --consistency-strict to fail on them).")
    if consistency_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
