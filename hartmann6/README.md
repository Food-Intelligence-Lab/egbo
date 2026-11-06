# Hartmann6 benchmark

Hartmann6 is embedded in 200 dimensions by appending 194 inert coordinates. This directory contains the exact main 10-seed traces and the Appendix E Q-sweep and expert-quality sweep.

Regenerate the paper figures from archived traces:

```bash
python3 plot_hartmann6.py --name release_v1_10seed --hero-only --out-path figure1.png
python3 plot_hartmann6.py --mode pq-heatmap --pq-glob 'results/revision_v1__pq_*' --out-path figure6.pdf
python3 plot_hartmann6.py --mode q-sweep --qsweep-glob 'results/revision_v1__qsweep_q*' --out-path figure7.pdf
python3 plot_hartmann6.py --mode active-set --qsweep-glob 'results/revision_v1__qsweep_q*' --out-path figure8.pdf
```

Fresh full run (10 seeds, 9 methods, CPU):

```bash
./run_hartmann6.sh
```

For a quick smoke test:

```bash
python3 run_hartmann6.py --dim 12 --budget 7 --n-init 3 --seeds 1 \
  --methods random oracle egbo_strong --name smoke_ci --no-plot
```

Conventions:

- Inactive coordinates are pinned at 0 (the `supp(x) ⊆ S` convention of
  Assumption 1), for EGBO, the oracle, and random-subset BO.
- Two adaptive schedules are both described as "Q = T" and they differ:
  `--expert-mode adaptive` queries the expert before every one of the T = 20
  evaluations, including the 5 Sobol initial points (Fig. 1(b), Fig. 6);
  `--n-rounds 20` queries once up front and then before each of the 15 BO
  steps, i.e. 16 queries (Figs. 7-8, where Q = 16 and Q = 20 coincide).
  On shared seeds they give different final values, so state which schedule a
  number comes from.
- The expert-quality grid is p in {0.1, 0.2, 0.4, 0.6, 0.8, 0.9} and
  q_fp in {0, 0.01, 0.05, 0.10, 0.20}; main results use 10 seeds, the Q-sweep 20.

`run_hartmann6.py --help` documents all methods and the `--n-rounds`, `--expert-p`, and `--expert-q` controls used by the appendix sweeps.
