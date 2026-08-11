# Notes — ch12 shootout

- **Cyclic game (9-strategy generalized RPS): the table's clean story.**
  Game-theoretic meta-solvers (nash 0.050, rm 0.054, diverse 0.067 — tied
  within seed noise) beat alpharank (0.144), uniform (0.344), and crush the
  naive baselines (last_k 0.982, self_play 0.990 ≈ maximally exploitable:
  they chase the cycle and never mix).
- **Leduc at this budget: the good variants do NOT separate.** All of
  nash/rm/diverse/self_play land at 1.37–1.41 (±0.04–0.08); only last_k
  (1.83) and uniform (1.56) are clearly worse. self_play is nominally lowest
  (1.367) — within one std of nash — which the chapter should present as-is:
  at 10 iterations x 15k episodes, best-response quality (not the meta-solver)
  is the binding constraint on Leduc. ch07 reached 0.82 with ~4x this budget.
  A larger-budget flagship rerun (e.g. 20 iterations x 60k episodes, ~1-2 days
  serial or ~hours with a parallel launcher) is the obvious camera-ready
  follow-up; this run's numbers stand as the equal-budget result at THIS
  budget, per CLAUDE.md rule 6.
- The `psro_diverse` arm pays its diversity-bonus compute inside the same
  episode budget (wall-clock recorded per run in shootout.csv is ~2x the plain
  arms); under an equal wall-clock budget it would rank lower.
- No hyperparameters tuned; all knobs are argparse defaults, identical across
  variants (hidden=64 cyclic / 128 leduc for every PPO arm).
