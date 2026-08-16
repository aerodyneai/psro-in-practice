# Figure manifest

The contract between the manuscript and this repository: every figure and
table in the book maps to one script invocation here. All commands are run
from the repository root with the default `--seed 0` (recorded in each CSV's
provenance) unless noted.

Two figure directories are relevant. **`docs/figures/`** holds the shipped,
book-ready renders — versioned, one directory, flat filenames. **Every
`run_chXX.py` accepts `--figdir docs/figures/`** to write there directly;
combined with `--plot-only` this re-renders from the committed CSVs in
seconds, without retraining. **`experiments/chXX/figures/`** is scratch —
overwritten by non-smoke runs, gitignored. `--smoke` always writes to
`figures_smoke/` regardless of `--figdir`, so CI never touches tracked
paths.

Regenerate every shipped figure from CSVs:

```bash
for f in experiments/ch*/run_ch*.py; do python "$f" --plot-only --figdir docs/figures/; done
```

Approximate runtimes below are for the reference server (see README
hardware notes); every script also accepts `--smoke` for a <60 s CI pass.

Chapter/figure numbers are placeholders (`Fig. X.y`) until the manuscript
freezes; the *slug* column is the stable key.

| Fig. | Slug | What it shows | Command | Outputs (under `experiments/…`) | Runtime |
|---|---|---|---|---|---|
| 2.1 | `nash_simplex` | Strategy simplexes for RPS, weighted RPS, matching pennies with Nash marked by both solvers | `python experiments/ch02_nash_from_scratch/run_ch02.py` | `ch02_nash_from_scratch/figures/nash_simplex.*`, `results/nash_solutions.csv` | seconds |
| 3.1 | `fp_convergence` | FP time-averaged strategy converging to uniform Nash on RPS | `python experiments/ch03_fictitious_play/run_ch03.py` | `ch03_fictitious_play/figures/fp_convergence.*`, `results/fp_rps.csv` | seconds |
| 3.2 | `fp_trajectory` | FP average-strategy spiral on the simplex (instantaneous play cycles) | same command | `ch03_fictitious_play/figures/fp_trajectory.*` | — |
| 3.3 | `fp_exploitability` | FP exploitability ~1/sqrt(t) on RPS and matching pennies (log-log) | same command | `ch03_fictitious_play/figures/fp_exploitability.*`, `results/fp_matching_pennies.csv` | — |
| 4.1 | `do_population_vs_size` | Double-oracle population at convergence vs game size (~2n/3), 5 seeds | `python experiments/ch04_double_oracle/run_ch04.py` | `ch04_double_oracle/figures/do_population_vs_size.*`, `results/convergence_vs_size.csv` | ~30 min |
| 4.2 | `do_exploitability` | Full-game exploitability falls over iterations while restricted reads ~0 — the honest-measurement lesson | same command | `ch04_double_oracle/figures/do_exploitability.*`, `results/exploitability_vs_iteration.csv` | — |
| 4.3 | `do_noise_floor` | Payoff-estimation noise floors full exploitability | same command | `ch04_double_oracle/figures/do_noise_floor.*`, `results/noise_effect.csv` | — |
| 6.1 | `kuhn_grid` | {uniform, Nash} x {tabular-Q, PPO} exploitability curves on Kuhn, 3 seeds | `python experiments/ch06_ppo_oracle/run_ch06.py --device cpu --n-seeds 3` | `ch06_ppo_oracle/figures/kuhn_grid.*`, `results/kuhn_grid.csv` | ~80 min |
| 7.1 | `leduc_exploitability` | Nash+PPO on Leduc: exact full-game exploitability per iteration | `python experiments/ch07_kuhn_leduc/run_ch07.py --device cpu` | `ch07_kuhn_leduc/figures/leduc_exploitability.*`, `results/leduc_nash_ppo.csv` | ~65 min |
| 8.1 | `solver_curves` | Five meta-solvers, same game and oracle, 3 seeds | `python experiments/ch08_meta_solvers/run_ch08.py` | `ch08_meta_solvers/figures/solver_curves.*`, `results/solver_curves.csv` | ~30 min |
| 8.2 (table) | `solver_runtimes` | Runtime-per-solve vs empirical-game size (alpha-Rank's size^2 wall) | same command | `ch08_meta_solvers/figures/solver_runtimes.*`, `results/solver_runtimes.csv` | — |
| 10.1 | `speedup` | Parallel-PSRO wall-clock speedup at 1/2/4/8 Ray workers (Amdahl at 2) with bit-identical results | `python experiments/ch10_pipeline/run_ch10.py` | `ch10_pipeline/figures/speedup.*`, `results/speedup.csv` | ~25 min |
| 11.1 | `diversity_curves` | Diversity bonus on/off, transitive vs cyclic game (honest null result at these settings) | `python experiments/ch11_diversity/run_ch11.py` | `ch11_diversity/figures/diversity_curves.*`, `results/diversity_curves.csv` | ~50 min |
| 12.1 (table) | `shootout_table` | THE flagship: 7 variants x 5 seeds x 2 games, equal budget, mean±std final exploitability, LaTeX emitted | `python experiments/ch12_shootout/run_ch12.py` | `ch12_shootout/results/shootout_table.tex`, `results/shootout.csv`, `results/shootout_summary.csv` | ~8 h |
| 12.2 | `shootout` | Same data as bar chart with per-seed scatter | same command | `ch12_shootout/figures/shootout.*` | — |
| 13.1 | `trajectories` | Pursuit-evasion episodes under the iteration-1/5/15 meta-strategies | `python experiments/ch13_pursuit_evasion/run_ch13.py` | `ch13_pursuit_evasion/figures/trajectories.*`, `results/psro_metrics.csv` | ~3.5 h |
| 13.2 | `meta_support` | Nash meta-strategy support heatmap over iterations (grows 1 to 5) | same command | `ch13_pursuit_evasion/figures/meta_support.*` | — |
| 13.3 | `exploitability_proxy` | Population-tournament proxy + best-response probe (no exact calculator exists; column names say which proxy) | same command | `ch13_pursuit_evasion/figures/exploitability_proxy.*`, `results/br_probe.csv` | — |
| 14.1 | `cache_hit_rate` | Payoff-table cache hit rate (~85% by iteration 12); O(k) border vs O(k^2) table | `python experiments/ch14_engineering/run_ch14.py` | `ch14_engineering/figures/cache_hit_rate.*`, `results/cache_hit_rate.csv` | ~15 min |
| 14.2 | `cost_breakdown` | Per-iteration wall-clock split: best responses / evaluation / meta-solve | same command | `ch14_engineering/figures/cost_breakdown.*`, `results/cost_breakdown.csv` | — |
| 14.x (text) | `resume_check` | Kill-at-iteration-6 checkpoint/resume produces a bit-identical final table | same command | `ch14_engineering/results/resume_check.csv` | — |

Notes:
- Ch. 5 (the loop itself) and Ch. 9 have no generated figures; Ch. 5 prints
  `src/psrolab/psro.py` verbatim and Ch. 6 prints `src/psrolab/oracles/ppo.py`.
- Every experiment directory may carry a `NOTES.md` recording measured
  runtimes, calibration decisions, and any deviation between planned and
  observed results; the manuscript's prose must stay consistent with those.
- Determinism: numpy-only experiments are bit-reproducible; torch experiments
  are reproducible on CPU (the settings used for all book figures).
- **Ch. 8 provenance:** the `nash`/`uniform` rows in `solver_curves.csv` were
  freshly generated for Ch. 8 (Kuhn poker + tabular-Q oracle, 3 seeds, 15
  iterations, commit `2533e88`). They are *not* reused from Ch. 6's
  `kuhn_grid.csv`, which uses different combos and a different seed schedule.
- **Ch. 13 caveat:** `trajectories.{pdf,png}` and `meta_support.{pdf,png}`
  cannot be regenerated from committed CSVs alone — the first needs live
  policies (no on-disk checkpoints), the second needs the per-iteration
  meta-weight matrix (only support counts are in `psro_metrics.csv`).
  `--plot-only` falls back to copying the last full-run renders from
  `experiments/ch13_pursuit_evasion/figures/`. A full non-smoke `run_ch13.py`
  run (~100 min) is the source of truth for those two.
