# Notes — ch07 Leduc

- **Device choice.** Benchmarked one PSRO iteration (PPO, hidden=128, 8000
  episodes) on the reference server: CPU 20s vs V100 31s. Episode collection
  does one net forward per decision, so per-step host↔device transfers dominate
  and the GPU loses at this net size. The book's Leduc figure therefore uses
  `--device cpu` (also making it bit-reproducible). GPUs start paying off with
  batched/vectorized collection — see Ch. 10 (Pipeline PSRO) — or much larger
  networks.
- No PPO hyperparameter tuning beyond the argparse defaults; `hidden=128` and
  `total_episodes=60000` were chosen a priori and not searched.
- **Reference run** (seed 0, CPU, 66 min): full exploitability 2.37 → 0.82
  over 20 iterations, oscillating — consistent with vanilla PSRO + approximate
  RL best responses on Leduc in the literature (Lanctot et al. 2017 report
  NashConv in the 1–2 range at similar depth). Iterations 16/17 and 18/19
  produced identical exploitability: those best responses received ~zero
  meta-strategy mass. Improving this plateau (better BRs, more evaluation
  episodes, diversity bonuses) is exactly Part III's subject; ch07
  deliberately ships the honest vanilla baseline.
