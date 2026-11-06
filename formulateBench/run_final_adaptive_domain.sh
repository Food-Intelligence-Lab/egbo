#!/usr/bin/env bash
set -euo pipefail

domain="${1:?usage: run_final_adaptive_domain.sh meat|dairy}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir"

python_bin="${PYTHON_BIN:-python3}"

if [[ "$domain" == meat ]]; then
  source_dir=results/meat_semantic_all14_v1
  cache_dir=adaptive_semantic_q2_meat_final_v1
  exclusions=(--exclude-ingredients "egg white" "egg whites" "fat" "milk" "milk proteins")
elif [[ "$domain" == dairy ]]; then
  source_dir=results/dairy_semantic_all10_v3
  cache_dir=adaptive_semantic_q2_dairy_final_v1
  exclusions=(--exclude-ingredients "mozzarella")
else
  echo "domain must be meat or dairy" >&2
  exit 2
fi

"$python_bin" run_adaptive_semantic_suite.py \
  --domain "$domain" \
  --source-dir "$source_dir" \
  --output-dir "results/final_protocol_adaptive_q2_v1/$domain" \
  --budget 20 \
  --bo-seeds 10 \
  --n-init 5 \
  --local-sigma 0.35 \
  --checkpoint 10 \
  --max-additions 2 \
  --llm-cache-dir "$cache_dir" \
  --bo-config-json results/bo_config_matched_v1.json \
  "${exclusions[@]}"

echo "Completed final adaptive EGBO for $domain"
