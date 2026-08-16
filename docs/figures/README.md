# docs/figures/

Book-ready renders — the ones cited in the manuscript. Versioned, so a
reader who clones the repo sees the same images as a reader who buys the
book.

Regenerate any subset from committed CSVs with:

```
python experiments/chXX_.../run_chXX.py --plot-only --figdir docs/figures/
```

`--plot-only` skips PSRO/training and re-renders directly from the CSVs
under `experiments/chXX/results/`. Runs in seconds.

## What lives where

| File | Chapter | Source data |
|---|---|---|
| `nash_simplex.{pdf,png}` | 2 | `ch02_nash_from_scratch/results/nash_solutions.csv` |
| `fp_convergence.*`, `fp_trajectory.*`, `fp_exploitability.*` | 3 | `ch03_fictitious_play/results/fp_{rps,matching_pennies}.csv` |
| `do_population_vs_size.*`, `do_exploitability.*`, `do_noise_floor.*` | 4 | `ch04_double_oracle/results/*.csv` |
| `kuhn_grid.*` | 6 | `ch06_ppo_oracle/results/kuhn_grid.csv` |
| `leduc_exploitability.*` | 7 | `ch07_kuhn_leduc/results/leduc_nash_ppo.csv` |
| `solver_curves.*`, `solver_runtimes.*` | 8 | `ch08_meta_solvers/results/*.csv` |
| `speedup.*` | 10 | `ch10_pipeline/results/speedup.csv` |
| `diversity_curves.*` | 11 | `ch11_diversity/results/diversity_curves.csv` |
| `shootout.*` | 12 | `ch12_shootout/results/shootout.csv` |
| `trajectories.*`, `meta_support.*`, `exploitability_proxy.*` | 13 | `ch13_pursuit_evasion/results/*.csv` (see caveat) |
| `cache_hit_rate.*`, `cost_breakdown.*` | 14 | `ch14_engineering/results/*.csv` |

## Ch. 13 caveat

`trajectories.pdf` and `meta_support.pdf` are the only two figures that
cannot be fully regenerated from committed CSVs:

- **`trajectories`** needs live policies + the pursuit-evasion env to
  sample episodes; policies aren't checkpointed. `--plot-only` falls
  back to copying the last full-run render from
  `experiments/ch13_pursuit_evasion/figures/`.
- **`meta_support`** needs the per-iteration meta-weight matrix, which
  `psro_metrics.csv` doesn't store (only support counts). Same copy
  fallback applies.

Both are regenerated properly by a full non-smoke `run_ch13.py` run
(≈100 min).

## Distinction from `experiments/*/figures/`

The chapter-local `figures/` directories are **scratch space** — ignored
by git, overwritten by every non-smoke run, and the primary target when
you re-run experiments to iterate on plot code. Only `docs/figures/` is
tracked. The two directories carry identical content for the same seed
and CSV, so promoting a scratch render to a shipped one is a `--figdir
docs/figures/` invocation.
