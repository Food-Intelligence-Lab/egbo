#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p reproduced/hartmann6 reproduced/formulatebench reproduced/deployment
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/egbo-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$MPLCONFIGDIR}"
mkdir -p "$MPLCONFIGDIR"

"$PYTHON_BIN" verify_paper_claims.py --strict

"$PYTHON_BIN" hartmann6/plot_hartmann6.py \
  --name release_v1_10seed --hero-only \
  --out-path reproduced/hartmann6/figure1_convergence.png
"$PYTHON_BIN" hartmann6/plot_hartmann6.py --mode pq-heatmap \
  --pq-glob 'hartmann6/results/revision_v1__pq_*' \
  --out-path reproduced/hartmann6/figure6_pq_heatmap.pdf
"$PYTHON_BIN" hartmann6/plot_hartmann6.py --mode q-sweep \
  --qsweep-glob 'hartmann6/results/revision_v1__qsweep_q*' \
  --out-path reproduced/hartmann6/figure7_qsweep.pdf
"$PYTHON_BIN" hartmann6/plot_hartmann6.py --mode active-set \
  --qsweep-glob 'hartmann6/results/revision_v1__qsweep_q*' \
  --out-path reproduced/hartmann6/figure8_active_set.pdf

COMMON_RESULTS=(
  --results-root formulateBench/results
  --legacy-names final_protocol_core_v1 final_protocol_saasbo_v1 final_protocol_sebo_v1
  --meat-semantic formulateBench/results/meat_semantic_all14_v1
  --dairy-semantic formulateBench/results/dairy_semantic_all10_v3
  --meat-default-egbo formulateBench/results/final_protocol_egbo_llm_config_v1/meat
  --dairy-default-egbo formulateBench/results/final_protocol_egbo_llm_config_v1/dairy
  --meat-native-egbo formulateBench/results/final_protocol_egbo_default_v1/meat
  --dairy-native-egbo formulateBench/results/final_protocol_egbo_default_v1/dairy
  --egbo-method-label 'EGBO (LLM-configured BO)'
)

"$PYTHON_BIN" formulateBench/plot_revised_figure2.py \
  "${COMMON_RESULTS[@]}" \
  --output reproduced/formulatebench/figure9_formulatebench_one_at_a_time
"$PYTHON_BIN" formulateBench/plot_revised_per_category.py \
  "${COMMON_RESULTS[@]}" \
  --output-dir reproduced/formulatebench
"$PYTHON_BIN" formulateBench/plot_adaptive_semantic_q2.py \
  --results-root formulateBench/results \
  --legacy-names final_protocol_core_v1 final_protocol_saasbo_v1 final_protocol_sebo_v1 \
  --meat-one-shot formulateBench/results/final_protocol_egbo_llm_config_v1/meat \
  --dairy-one-shot formulateBench/results/final_protocol_egbo_llm_config_v1/dairy \
  --meat-adaptive formulateBench/results/final_protocol_adaptive_q2_v1/meat \
  --dairy-adaptive formulateBench/results/final_protocol_adaptive_q2_v1/dairy \
  --one-shot-method-label 'EGBO (LLM-configured BO)' \
  --output reproduced/formulatebench/figure12_adaptive_q2

# Regime analysis and observation-noise sweep: analysis of cached records only
# (no API calls). The noise-sweep runner (run_noise_sweep.py) is NOT run here.
"$PYTHON_BIN" formulateBench/analyze_regimes.py \
  --results-dir formulateBench/results \
  --output-dir reproduced/formulatebench/regime_analysis
"$PYTHON_BIN" formulateBench/analyze_noise_sweep.py

# Batched experiments (main-text Fig. 2, Table 6): 10 batches of 3 recipes, with and
# without noise, three LLMs plus random search / vanilla BO / TuRBO. Cached records only.
"$PYTHON_BIN" formulateBench/analyze_batched.py --root formulateBench/results/batched_r10_v1
"$PYTHON_BIN" formulateBench/plot_batched_figure.py \
  --root formulateBench/results/batched_r10_v1 \
  --output reproduced/formulatebench/figure2_formulatebench_batched

# Other LLMs (Table 7), noise with other LLMs, the noise-aware prompt arm, and the
# preliminary tool-using-agent test (Table 8). Cached records only; no API calls.
"$PYTHON_BIN" formulateBench/analyze_multi_llm.py
"$PYTHON_BIN" formulateBench/analyze_noise_other_model.py \
  --slug openai__gpt-6.1-sol --noisy formulateBench/results/noise_sweep_gpt61sol_v1
"$PYTHON_BIN" formulateBench/analyze_noise_other_model.py \
  --slug google__gemini-4-argon --noisy formulateBench/results/noise_sweep_gemini4argon_v1
"$PYTHON_BIN" formulateBench/analyze_noise_aware.py
"$PYTHON_BIN" formulateBench/analyze_agent_pilot.py

"$PYTHON_BIN" deployment/plot_results.py --out-dir reproduced/deployment

echo "Reproduced outputs are under reproduced/."
