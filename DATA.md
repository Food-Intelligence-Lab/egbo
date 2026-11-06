# Data provenance

## USDA FoodData Central

`formulateBench/formulatebench/data/usda_nutrition_data.csv` and the nutrient
columns in `animal_products_nutrition.csv` are derived from USDA FoodData
Central. The release preserves the matched FoodData Central identifier and
data type where available.

USDA states that FoodData Central data are in the public domain and published
under CC0 1.0. Please cite:

> U.S. Department of Agriculture, Agricultural Research Service. FoodData
> Central, 2019. https://fdc.nal.usda.gov/

## NECTAR ingredient ground sets

The meat and dairy membership columns encode the plant-based ingredient ground
sets constructed from NECTAR's public Taste of the Industry research. The
release contains ingredient labels and domain membership, not NECTAR sensory
panel observations or product-level sensory scores.

NECTAR-derived material is not relicensed under the software license. Users
should cite the accompanying EGBO paper and the corresponding NECTAR Taste of
the Industry report.

## Real-world deployment data

`deployment/data/` holds aggregate panel scores (one aggregated score per
attribute per recipe), the derived utilities, and the ingredient trajectories
for the yogurt and cottage-cheese campaigns and the human-food-scientist
comparison. `deployment/paper_assets/` holds the yogurt photographs shown in
the paper. No individual panelist responses are included. The partner's
proprietary ingredient catalog is not released.

These files are not covered by the MIT license. They are released so that
the paper's deployment results can be checked and its figures regenerated;
contact the authors before using them for other purposes.

## Frozen experiment records and cached LLM responses

`formulateBench/results/` holds the frozen trajectories behind the paper's
FormulateBench results, plus two derived analyses:

- `regime_analysis_v1/`: tables and figure computed by
  `formulateBench/analyze_regimes.py` from the other frozen records; no new
  experiments.
- `noise_sweep_v1/`: the observation-noise sweep (one JSON per category,
  noise level, arm and seed), its summaries and figure, and `llm_cache/`. The cache holds the raw LLM responses (with the
  prompts that produced them) returned by the model endpoint during the
  sweep. It is included for reproducibility, so that the sweep's LLM
  trajectories can be inspected and replayed without API access. It contains
  no API keys or personal data.

- `batched_v1/` (7 batches) and `batched_r10_v1/` (10 batches, used in the
  paper): batched experiments; LLM arms, EGBO, and, in `batched_r10_v1/baselines/`,
  random search, vanilla BO and TuRBO. `batched_v1/llm_cache/` holds the cached
  LLM responses; the 10-batch runs replay the first seven batches from it.
- `multi_llm_v1/`: one-at-a-time comparison with other LLMs (each model's first
  recipe, iterative LLM arms, EGBO with the paper's and the model's own BO
  configuration) and its `llm_cache/`.
- `noise_sweep_gpt61sol_v1/`, `noise_sweep_gemini4argon_v1/`: the noise sweep
  with other LLMs; `noise_aware_opus_v1/`, `noise_aware_gpt61sol_v1/`: the
  iterative LLM told the exact noise level.
- `agent_pilot_v1/`: the tool-using-agent test. For each run, a
  result JSON (scores, every Python cell the agent executed) and a
  `.transcript.json` with the full message history returned by the model,
  including any reasoning text. Transcripts contain no API keys or personal
  data; the sandbox had no network access.

The ingredient names in these records come from the NECTAR-derived ground
sets and carry the terms described above.

## Transformations

The benchmark applies the following deterministic transformations:

1. restrict ingredients to the meat or dairy ground set;
2. remove the exclusions documented in `formulateBench/EXPERIMENT_PROTOCOL.md`;
3. replace missing ingredient nutrient values with zero, matching the released
   paper runner;
4. compute nutrient-wise mean and standard deviation over the eligible domain
   pool;
5. score candidates using RMSE in this standardized nutrient space over the
   nutrients reported for each animal-product target.

The two source CSV files are packaged with FormulateBench so that the benchmark
remains stable if upstream databases change.
