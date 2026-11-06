#!/usr/bin/env bash
set -euo pipefail

domain="${1:?usage: run_matched_baselines_domain.sh meat|dairy}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/egbo-mpl-matched}"
mkdir -p "$MPLCONFIGDIR"

python_bin="${PYTHON_BIN:-python3}"
common=(
  --domain "$domain"
  --budget 20
  --seeds 10
  --n-init 5
  --output-root results
  --no-plot
)

if [[ "$domain" == meat ]]; then
  exclusions=(--exclude-ingredients "egg white" "egg whites" "fat" "milk" "milk proteins")
elif [[ "$domain" == dairy ]]; then
  exclusions=(--exclude-ingredients "mozzarella")
else
  echo "domain must be meat or dairy" >&2
  exit 2
fi

"$python_bin" run_formulatebench.py \
  "${common[@]}" "${exclusions[@]}" \
  --methods random vanilla turbo rembo random_subset \
  --name final_protocol_core_v1

"$python_bin" run_formulatebench.py \
  "${common[@]}" "${exclusions[@]}" \
  --methods saasbo \
  --name final_protocol_saasbo_v1

"$python_bin" run_formulatebench.py \
  "${common[@]}" "${exclusions[@]}" \
  --methods sebo \
  --name final_protocol_sebo_v1

echo "Completed matched-pool baselines for $domain"
