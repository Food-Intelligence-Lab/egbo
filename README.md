# Expert-Guided Bayesian Optimization

Official code and benchmark release for **Expert-Guided Bayesian Optimization
for Sustainable Protein Design** (NeurIPS 2026).

This repository has two purposes:

1. reproduce the numerical claims and figures in the paper from frozen records;
2. provide a small, stable interface for evaluating new methods on
   **FormulateBench**.

No API key is necessary for either the FormulateBench quickstart or exact paper
figure reproduction. The released LLM trajectories and initial recipes are
frozen so that results do not depend on future model or API changes.

## FormulateBench in two minutes

Python 3.11-3.13 is supported for the benchmark. The pinned paper stack
(`.[paper]`, used below to verify the paper) requires Python 3.12 or 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .

formulatebench list
formulatebench describe dairy/yogurt
formulatebench evaluate dairy/yogurt examples/yogurt_recipe.json
formulatebench demo dairy/yogurt --budget 8
```

The evaluator has one required dependency, NumPy. It validates the recipe,
computes its nutrient profile, and returns both raw and paper-normalized loss.
Lower is better.

Python usage:

```python
from formulatebench import get_task

task = get_task("dairy/yogurt")
result = task.evaluate({
    "oat milk": 0.75,
    "coconut oil": 0.15,
    "pea protein": 0.10,
})
print(result.loss, result.normalized_loss)
```

See [`formulateBench/README.md`](formulateBench/README.md) for the full task
definition, evaluation protocol, and a template for testing a new optimizer.

## Verify the paper

Install the exact scientific stack and regenerate all data-driven paper
figures from the released records:

```bash
python3 -m pip install -e '.[paper]'
./reproduce_all.sh
```

The script first runs `verify_paper_claims.py --strict`, then writes figures
to `reproduced/`. The verifier has two parts: headline claims (fail under
`--strict`) and a data-consistency audit that recomputes quantities from their
definitions (Eq. 7 utilities, best-recipe identity, the two Q = T schedules,
the expert-quality heatmap). The audit prints `WARN` lines and fails only
under `--consistency-strict`.

| Output under `reproduced/` | Paper figure |
| --- | --- |
| `hartmann6/figure1_convergence.png` | Fig. 1 (and Fig. 5) |
| `hartmann6/figure6_pq_heatmap.pdf` | Fig. 6 |
| `hartmann6/figure7_qsweep.pdf` | Fig. 7 |
| `hartmann6/figure8_active_set.pdf` | Fig. 8 |
| `formulatebench/figure2_formulatebench_batched.*` | Fig. 2 (batched, 10 × 3 recipes) and Table 6 |
| `formulatebench/figure9_formulatebench_one_at_a_time.*` | Fig. 9 |
| `formulatebench/formulatebench_meat_per_category.*` | Fig. 10 |
| `formulatebench/formulatebench_dairy_per_category.*` | Fig. 11 |
| `formulatebench/figure12_adaptive_q2.*` | Fig. 12 |
| `formulatebench/regime_analysis/formulatebench_regimes.*` | Fig. 13 (frozen copy in `formulateBench/results/regime_analysis_v1/`) |
| `formulateBench/results/noise_sweep_v1/formulatebench_noise.*` (written in place) | Fig. 14 and Table 5 |
| `deployment/figure15_yogurt_sensory.png` … `figure17_yogurt_ingredients.png` | Figs. 15-17 |
| `deployment/figure28_cottage_sensory.png` … `figure32_egbo_vs_hfs.png` | Figs. 28-32 |
| printed summaries + `*_summary.json` in the corresponding `formulateBench/results/` folder | Tables 6-8 (batched, other LLMs, tool-using agents) |

Figs. 7 and 8 use the Hartmann6 Q-sweep with Q ∈ {1, 2, 4, 8, 16} (with
T = 20 and 5 initial points, Q = 16 is the largest distinct schedule; the
archived Q = 20 run duplicates it and is not plotted). Fig. 8 shows
Q ∈ {1, 4, 16}.

`reproduce_all.sh` does not rerun the approximately 180 CPU-hour experimental
sweep. Fresh-run commands are documented in each component README.

The real-world yogurt photographs are archival experimental images rather
than computational plots; the paper assets are under
`deployment/paper_assets/`.

## Regime analysis and noise sweep

Two additional FormulateBench analyses reproduce from cached results with no
API key and no LLM cost (both run in `reproduce_all.sh`):

- **Regime analysis** (`formulateBench/analyze_regimes.py`): which family is
  best (iterative LLM, full-space BO, or EGBO) at each budget, head-to-head
  EGBO-vs-LLM counts, and how EGBO's advantage tracks the LLM's ingredient
  choice error and concentration error (oracle-based, Spearman over 24
  categories). Outputs go to `formulateBench/results/regime_analysis_v1/`.
- **Observation-noise sweep** (`formulateBench/run_noise_sweep.py` runner,
  `formulateBench/analyze_noise_sweep.py` analysis): the iterative LLM and
  one-shot EGBO arms re-run with additive Gaussian observation noise of
  sd = level × the category's mean random-search loss at evaluation 1
  (levels 0.05 and 0.15; the noiseless paper runs are level 0). Each method
  is scored by the true loss of its recommended recipe, i.e. the best observed
  so far. `python3 formulateBench/analyze_noise_sweep.py` recomputes all
  numbers from `formulateBench/results/noise_sweep_v1/`. Rerunning the sweep
  itself needs `OPENROUTER_API_KEY` and about 5,500 LLM calls (288 LLM
  trajectories × 19 calls; 3 noise draws per LLM arm).

Further cached analyses, also run by `reproduce_all.sh` with no API key:

- **Batched experiments** (`run_batched.py`, `analyze_batched.py`,
  `plot_batched_figure.py`): 10 batches of 3 recipes, as in the laboratory
  deployment, with and without noise; three LLMs plus random search, vanilla
  BO, and TuRBO. Main-text Fig. 2 and Table 6.
- **Other LLMs** (`run_multi_llm.py`, `run_multi_llm_selfcfg.py`,
  `analyze_multi_llm.py`, `analyze_noise_other_model.py`): the one-at-a-time
  comparison and noise sweep with GPT-6.1 Sol and Gemini 4 Argon (Table 7).
- **Noise-aware prompt** (`analyze_noise_aware.py`): the iterative LLM told the
  exact noise level.
- **Tool-using agents** (`run_agent_pilot.py`,
  `analyze_agent_pilot.py`): agents with an experiment tool and a sandboxed
  Python tool; does the agent choose to model its data? Claude Opus 4.7,
  GPT-6.1 Sol, and Gemini 4 Argon (Table 8).

Details are in [`formulateBench/README.md`](formulateBench/README.md).

## Repository map

| Path | Contents |
| --- | --- |
| `formulateBench/formulatebench/` | Stable public benchmark API and CLI |
| `formulateBench/formulatebench/data/` | Frozen nutrition tables and task metadata |
| `formulateBench/results/` | Frozen paper trajectories and figure inputs |
| `formulateBench/*.py` | Paper experiment and plotting code |
| `hartmann6/` | Controlled synthetic study |
| `deployment/` | Real-world BO code, aggregate data, and plots |
| `examples/` | Recipe and custom-method examples |
| `verify_paper_claims.py` | Machine-checkable audit of headline results |
| `reproduce_all.sh` | One-command figure reproduction |

## Reproducibility boundaries

- **Exact reproduction:** uses frozen JSON/CSV records and requires no model
  endpoint.
- **Fresh optimization runs:** require the paper dependencies and CPU time.
- **Fresh LLM trajectories:** additionally require `OPENROUTER_API_KEY` and
  access to the recorded model identifier. Model-provider behavior may drift;
  these runs are not expected to be bitwise identical to the frozen records.
- **Real-world deployment:** aggregate measurements and optimization code are
  released. The partner's proprietary ingredient catalog is not.

## Tests

```bash
python3 -m pip install -e '.[paper,test]'
python3 -m pytest -q
python3 verify_paper_claims.py --strict
```

## Data provenance

FormulateBench uses nutrient data derived from USDA FoodData Central and
plant-based ingredient ground sets derived from NECTAR Taste of the Industry
research. Third-party inputs retain their original terms and are not
relicensed by this repository. Deployment files contain only aggregate or
anonymized data. See
[`DATA.md`](DATA.md) for sources and transformations.

## Citation

Please cite the accompanying paper when using EGBO or FormulateBench.
Machine-readable metadata are in [`CITATION.cff`](CITATION.cff). Add the arXiv
identifier after it is assigned.

## License

The code is released under the [MIT License](LICENSE). The data are not: USDA
FoodData Central data are CC0 1.0, the NECTAR-derived ingredient ground sets
keep NECTAR's terms, and the real-world deployment data and photographs are
provided for verifying the paper. The scope section of [`LICENSE`](LICENSE)
and [`DATA.md`](DATA.md) give details.
