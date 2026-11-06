# FormulateBench

FormulateBench is a suite of 24 black-box formulation tasks: 14 plant-based
meat categories and 10 plant-based dairy categories. A method proposes a
nonnegative mixture of allowed ingredients whose fractions sum to one. The
evaluator returns nutritional distance to the corresponding animal product;
lower is better.

The public evaluator is deliberately independent of the paper's BO and LLM
implementations. You can use any optimizer, programming framework, or model.

## Task definition

For each domain, ingredient nutrient vectors are standardized using the mean
and standard deviation of the domain's eligible ingredient pool. The loss is
the root mean squared distance between the candidate and target vectors in
this z-scored nutrient space, over nutrients reported for the target category.

- Nutrients: energy, protein, fat, saturated fat, carbohydrates, fiber,
  sugars, and sodium.
- Meat pool: 106 eligible ingredients.
- Dairy pool: 90 eligible ingredients.
- Candidate constraint: mixture fractions are nonnegative and sum to one.
- Standard paper budget: 20 sequential objective evaluations.
- Repetitions: 10 seeds for stochastic optimizers; one cached trajectory for
  each temperature-zero LLM baseline.

See [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md) for the frozen paper
protocol, including initialization, information given to the LLM, and the
normalization used for cross-category plots.

## Install and inspect the tasks

From the repository root:

```bash
python3 -m pip install -e .
formulatebench list
formulatebench describe dairy/yogurt
```

Score a recipe:

```bash
formulatebench evaluate dairy/yogurt examples/yogurt_recipe.json
```

The command returns:

- `loss`: raw z-score RMSE for the task;
- `normalized_loss`: raw loss divided by that category's frozen random-search
  normalizer used in the paper;
- the candidate nutrient profile and measured target nutrients.

## Use the Python API

```python
from formulatebench import get_task, list_tasks

print(list_tasks("dairy"))
task = get_task("dairy/yogurt")

evaluation = task.evaluate({
    "oat milk": 0.75,
    "coconut oil": 0.15,
    "pea protein": 0.10,
})
print(evaluation.loss)
print(evaluation.normalized_loss)
```

Invalid ingredients, negative quantities, non-finite values, and mixtures
that do not sum to one raise `RecipeError` rather than being silently changed.

## Test a new optimization method

Copy [`../examples/custom_method.py`](../examples/custom_method.py) and replace
the candidate-proposal rule with your optimizer:

```bash
python3 examples/custom_method.py \
  --task dairy/yogurt \
  --budget 20 \
  --seed 0 \
  --output method_result.json

formulatebench validate-result method_result.json
```

For a paper-style comparison:

1. run all 24 tasks with a budget of 20 evaluations;
2. use 10 independent seeds for stochastic methods;
3. report best-so-far loss at every evaluation;
4. divide each task trajectory by `task.normalizer` before averaging across
   categories;
5. use categories—not pooled seed-by-category records—as the uncertainty unit.

The result template uses a small JSON format:

```json
{
  "schema_version": "1.0",
  "task": "dairy/yogurt",
  "method": "my-method",
  "budget": 20,
  "seed": 0,
  "trajectory": [
    {"evaluation": 1, "loss": 0.42, "best_so_far": 0.42}
  ]
}
```

The machine-readable definition is [`result_schema.json`](result_schema.json).

## Reproduce the paper figures

From the repository root, after installing `.[paper]`:

```bash
./reproduce_all.sh
```

The FormulateBench portion reads only the following frozen result groups:

- `final_protocol_core_v1`: random search, vanilla BO, TuRBO, REMBO, and
  random-subset BO;
- `final_protocol_saasbo_v1` and `final_protocol_sebo_v1`;
- `final_protocol_egbo_default_v1`;
- `final_protocol_egbo_llm_config_v1`;
- `meat_semantic_all14_v1` and `dairy_semantic_all10_v3`: shared initial
  recipes and fixed-support/unrestricted iterative LLM trajectories;
- `final_protocol_adaptive_q2_v1`.

Generated plots are written to `reproduced/formulatebench/`.

## Regime analysis and observation-noise sweep

Both analyses read only cached records: no API key, no LLM calls, and each
finishes in seconds. `reproduce_all.sh` runs both.

### Regime analysis (`analyze_regimes.py`)

Asks when each strategy family wins and why. Per category it compares the
best of the two iterative LLM arms (unrestricted and fixed-support), the best
of two full-space BO arms (vanilla BO and warm-start BO), and LLM-configured
EGBO, all on the normalized best-so-far loss. It then relates EGBO's advantage
to two descriptors of the LLM's first recipe, computed with the NNLS simplex
oracle: `sup_gap` (best loss achievable on the LLM's chosen ingredients minus
best achievable on all ingredients) and `init_gap` (the first recipe's loss
minus the best achievable on its ingredients).

```bash
python3 formulateBench/analyze_regimes.py
# defaults: --results-dir formulateBench/results
#           --output-dir  formulateBench/results/regime_analysis_v1
```

It prints the per-category table, family win counts at evaluations 5/10/20,
EGBO-vs-LLM head-to-head counts, crossover evaluations, and Spearman
correlations, and writes `regime_rows.json`, `regime_table.json`, and
`formulatebench_regimes.{pdf,png}` to the output directory.

### Observation-noise sweep (`run_noise_sweep.py`, `analyze_noise_sweep.py`)

Re-runs the unrestricted iterative LLM, fixed-support iterative LLM, and
one-shot EGBO arms when every observed score is corrupted by additive
Gaussian noise, `y_obs = y_true + N(0, sd^2)` with
`sd = level × (category's mean random-search loss at evaluation 1)`, for
levels 0.05 and 0.15. The noiseless paper runs serve as level 0. All arms
start from the cached shared LLM recipe; EGBO keeps its cached ingredient
support and only its BO refinement is re-run. Each evaluation logs its true
(noiseless) loss, and a method is scored by the **true loss of the recipe it
would recommend, i.e. the one with the best observed score so far**, not by
its best noisy observation. Records are in `results/noise_sweep_v1/`
(768 run JSONs: 24 categories × 2 levels × (10 EGBO seeds + 3 noise draws
for each LLM arm)), together with `llm_cache/`, the raw cached LLM responses.

Analysis only (no API):

```bash
python3 formulateBench/analyze_noise_sweep.py
```

This prints mean normalized losses and EGBO-vs-best-LLM counts per noise
level and writes `noise_summary.json` and `formulatebench_noise.{pdf,png}`. A
robustness section adds paired Wilcoxon tests (best LLM − EGBO, evaluations 5
and 20), the mean improvement from evaluation 5 to 20 per arm, and a variant
scored by the best true recipe ever evaluated (`best_true_so_far`), which
separates search from selection under noise; it is saved to
`noise_robustness.json`.

Rerunning the sweep itself requires `OPENROUTER_API_KEY` and, from scratch,
about 5,500 LLM calls (288 LLM trajectories × 19 calls after the shared first
recipe). The runner skips any run JSON that already exists and reads LLM
responses from `results/noise_sweep_v1/llm_cache/` (5,494 released
responses), so to rerun from scratch use a fresh `--out` directory and move
the cache aside. The released sweep used three noise draws per LLM arm:

```bash
export OPENROUTER_API_KEY="..."
python3 formulateBench/run_noise_sweep.py --llm-seeds 3 --egbo-seeds 10 \
  --levels 0.05 0.15 --out formulateBench/results/noise_sweep_v1
```

## Batched experiments, other LLMs, and tool-using agents

All analyses below read cached records only (no API key) and are run by
`reproduce_all.sh`. Rerunning the experiments themselves requires
`OPENROUTER_API_KEY`; set `OPENROUTER_MAX_RPM` (e.g. 18) if your account limits
requests per minute per model. Runners skip results that already exist, and
stopping a runner stops its worker processes.

### Batched experiments (`run_batched.py`, `analyze_batched.py`, `plot_batched_figure.py`)

Mirrors the laboratory deployment: 10 batches of 3 recipes (30 evaluations),
with and without observation noise (`--levels 0 0.15`). For each LLM, the first
batch is the LLM's first recipe plus two seeded logit-normal perturbations;
later batches come from the iterative LLM arms (asked for three formulations at
once), EGBO (qLogNEI with `q = 3`, sequential greedy), or the LLM-free
baselines `random`, `vanilla_bo`, and `turbo` over the full catalog. Scoring
is the true loss of the best observed recipe.

```bash
python3 formulateBench/analyze_batched.py --root formulateBench/results/batched_r10_v1
python3 formulateBench/plot_batched_figure.py --root formulateBench/results/batched_r10_v1 \
  --output reproduced/formulatebench/figure2_formulatebench_batched
```

To rerun (about 2,700 LLM calls; the 10-batch runs replay the first seven
batches from the 7-batch cache):

```bash
python3 formulateBench/run_batched.py --rounds 10 --levels 0 0.15 \
  --out formulateBench/results/batched_r10_v1 \
  --cache-dir formulateBench/results/batched_v1/llm_cache
python3 formulateBench/run_batched.py --rounds 10 --levels 0 0.15 \
  --arms random vanilla_bo turbo --models anthropic/claude-opus-4.7 --seeds 10 \
  --out formulateBench/results/batched_r10_v1
```

### Other LLMs (`run_multi_llm.py`, `run_multi_llm_selfcfg.py`, `analyze_multi_llm.py`, `analyze_noise_other_model.py`)

Repeats the one-at-a-time comparison with GPT-6.1 Sol
and Gemini 4 Argon. Each model proposes its own first recipe (which fixes EGBO's
ingredients), then the two iterative LLM arms and EGBO run as in the paper.
`run_multi_llm_selfcfg.py` lets each model choose its own BO configuration.
The noise sweep is repeated with another model's first recipe via
`run_noise_sweep.py --recipe-root formulateBench/results/multi_llm_v1/<model>`;
`--noise-aware` tells the LLM the exact noise level (`analyze_noise_aware.py`).

### Tool-using agents (`run_agent_pilot.py`, `analyze_agent_pilot.py`)

Does an agent with tools choose to model its experiments? The agent gets the
same task prompt as the unrestricted iterative LLM (only the final paragraph
differs, to state the budget and describe the tools) and two tools:
`evaluate` (one formulation per call, 20 evaluations including the first
recipe) and `python` (code in a sandbox with no network access and no access
to the benchmark code; `evaluations.json` always holds the history). Options:
`--hint` adds one sentence about noisy, expensive measurements;
`--name-botorch` lists BoTorch among the installed libraries. The sandbox uses
macOS `sandbox-exec`. `analyze_agent_pilot.py` classifies each run from the
code the agent executed and compares final losses with the iterative LLM and
EGBO. The runs cover Claude Opus 4.7, GPT-6.1 Sol, and Gemini 4 Argon (about
4,800 LLM calls).

```bash
python3 formulateBench/run_agent_pilot.py --models openai/gpt-6.1-sol --levels 0.15
python3 formulateBench/analyze_agent_pilot.py
```

## Run fresh paper experiments

Fresh conventional-baseline runs:

```bash
cd formulateBench
PYTHON_BIN=python3 ./run_matched_baselines_domain.sh meat
PYTHON_BIN=python3 ./run_matched_baselines_domain.sh dairy
```

Fresh EGBO or iterative-LLM runs require the frozen initial LLM recipes. Cache
misses additionally require:

```bash
export OPENROUTER_API_KEY="..."
```

These expensive commands are for replication research, not for normal use of
the benchmark. Exact paper plots should use the frozen records.

## Data files

- `formulatebench/data/usda_nutrition_data.csv`: eligible plant-ingredient
  nutrition table and domain membership.
- `formulatebench/data/animal_products_nutrition.csv`: target animal-product
  profiles.
- `formulatebench/data/normalizers.json`: frozen category normalizers used for
  aggregate plots.

The packaged files under `formulatebench/data/` are the single source of truth
for both the public evaluator and the paper experiment runners.
