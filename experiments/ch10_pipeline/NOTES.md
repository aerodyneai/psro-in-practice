# Notes — ch10 parallel PSRO

- **Measured speedups** (reference server, 8 iterations, PPO 20k episodes/BR,
  torch pinned to 1 thread per worker): 1 worker 603s, 2 workers 274s (2.20x),
  4 workers 275s (2.19x), 8 workers 260s (2.32x). With two players the BR
  phase parallelizes exactly 2-way; workers beyond 2 only accelerate
  payoff-table evaluation, which is a small share of Leduc wall-clock at these
  settings. This is the Amdahl figure the chapter builds its Pipeline-PSRO
  motivation on.
- **Bit-identical across worker counts**: every run produced the same payoff
  table and final exploitability (1.2681), verified at runtime and recorded in
  the CSV (`results_identical_to_first_run`). Parallelism here is a
  wall-clock-only decision by construction (SeedSequence per cell/BR).
- Final exploitability differs from ch07's 20-iteration run because this
  config uses 8 iterations x 20k episodes — the speedup sweep needs identical
  short runs, not the best possible policy.
- No tuning performed; all knobs are argparse defaults.
