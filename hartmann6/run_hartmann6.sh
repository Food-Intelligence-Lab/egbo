#!/usr/bin/env bash
# Fan-out runner for the Hartmann6 benchmark (paper Figs 1 and 6).
#
# Maximizes parallelism by splitting the slow methods (SAASBO, SEBO) into
# per-seed jobs. Total jobs = 1 (cheap baselines) + SEEDS (saasbo) +
# SEEDS (sebo) + 1 (egbo oneshot) + 1 (egbo adaptive) = 3 + 2*SEEDS.
# With SEEDS=10 that is 23 jobs; with 18 cores, all run in parallel.
#
# Wall-clock floor ≈ one SAASBO seed at d=DIM (~6-15 min at d=200).
#
# Usage:
#   ./run_hartmann6.sh                                  # defaults
#   NAME=release_v2 SEEDS=10 PYTHON=... ./run_hartmann6.sh
#
# Env vars:
#   NAME    output run name             (default: release_v1)
#   SEEDS   seeds per (method, mode)    (default: 10)
#   BUDGET  evaluations T per seed      (default: 20)
#   N_INIT  initial Sobol points        (default: 5)
#   DIM     ambient dimension N         (default: 200)
#   PYTHON  python interpreter          (default: python)
#   PARALLEL  concurrent jobs cap       (default: 18; tune to your core count)

set -euo pipefail

NAME="${NAME:-release_v1}"
SEEDS="${SEEDS:-10}"
BUDGET="${BUDGET:-20}"
N_INIT="${N_INIT:-5}"
DIM="${DIM:-200}"
PYTHON="${PYTHON:-python}"
PARALLEL="${PARALLEL:-18}"

cd "$(dirname "$0")"

echo "Hartmann6 fan-out:"
echo "  NAME=$NAME  DIM=$DIM  SEEDS=$SEEDS  BUDGET=$BUDGET  N_INIT=$N_INIT"
echo "  PARALLEL=$PARALLEL  PYTHON=$PYTHON"
echo

# Each job: <slug>:<mode>:<methods (space-sep)>:<seed_start>:<n_seeds>
JOBS=()
JOBS+=("cheap:oneshot:random vanilla oracle rembo random_subset turbo:0:$SEEDS")
for s in $(seq 0 $((SEEDS - 1))); do
    JOBS+=("saasbo_seed${s}:oneshot:saasbo:${s}:1")
    JOBS+=("sebo_seed${s}:oneshot:sebo:${s}:1")
done
JOBS+=("egbo_oneshot:oneshot:egbo_strong egbo_weak:0:$SEEDS")
JOBS+=("egbo_adaptive:adaptive:egbo_strong egbo_weak:0:$SEEDS")

echo "  jobs (${#JOBS[@]}):"
for j in "${JOBS[@]}"; do echo "    - $j"; done
echo

export NAME PYTHON BUDGET N_INIT DIM
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

printf '%s\0' "${JOBS[@]}" \
  | xargs -0 -n1 -P"$PARALLEL" -I{} bash -c '
        spec="$1"
        IFS=: read -r slug mode methods seed_start n_seeds <<< "$spec"
        echo ">>> [$slug] starting (mode=$mode methods=$methods seeds=${seed_start}..$((seed_start + n_seeds - 1)))"
        "$PYTHON" run_hartmann6.py \
            --dim "$DIM" \
            --budget "$BUDGET" \
            --seeds "$n_seeds" \
            --seed-start "$seed_start" \
            --n-init "$N_INIT" \
            --expert-mode "$mode" \
            --methods $methods \
            --name "${NAME}__${slug}" \
            --no-plot \
          2>&1 | sed -l "s/^/[$slug] /"
        echo ">>> [$slug] done"
    ' _ {}

echo
echo "All jobs finished. Merging..."
"$PYTHON" merge_hartmann6.py --base "$NAME" --seeds "$SEEDS"
echo
echo "Done. Generate plots with:"
echo "  $PYTHON plot_hartmann6.py --name $NAME"
