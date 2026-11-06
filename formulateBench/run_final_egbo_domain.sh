#!/usr/bin/env bash
set -euo pipefail

domain="${1:?usage: run_final_egbo_domain.sh meat|dairy}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir"

python_bin="${PYTHON_BIN:-python3}"
config=results/bo_config_matched_v1.json

if [[ "$domain" == meat ]]; then
  source_dir=results/meat_semantic_all14_v1
  exclusions=(--exclude-ingredients "egg white" "egg whites" "fat" "milk" "milk proteins")
elif [[ "$domain" == dairy ]]; then
  source_dir=results/dairy_semantic_all10_v3
  exclusions=(--exclude-ingredients "mozzarella")
else
  echo "domain must be meat or dairy" >&2
  exit 2
fi

common=(
  --domain "$domain"
  --source-dir "$source_dir"
  --budget 20
  --bo-seeds 10
  --n-init 5
  --local-sigma 0.35
  "${exclusions[@]}"
)

"$python_bin" run_default_egbo_suite.py \
  "${common[@]}" \
  --output-dir "results/final_protocol_egbo_default_v1/$domain" \
  --method-label "EGBO (default BO)"

"$python_bin" run_default_egbo_suite.py \
  "${common[@]}" \
  --output-dir "results/final_protocol_egbo_llm_config_v1/$domain" \
  --bo-config-json "$config" \
  --method-label "EGBO (LLM-configured BO)"

echo "Completed final one-shot EGBO variants for $domain"
