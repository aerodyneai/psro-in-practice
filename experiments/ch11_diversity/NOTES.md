# Notes — ch11 diversity ablation

- **Measured result (3 seeds, λ=1, 12 iterations): the diversity bonus did NOT
  improve final exploitability on either game.** Transitive: off 0.0057±0.0005
  vs on 0.0059±0.0003. Cyclic: off 0.053±0.010 vs on 0.069±0.031 (high
  variance, overlapping bands throughout). The bonus also costs ~2.3x
  wall-clock (per-step min-over-population TV distance, cost growing with
  population size) — under an equal wall-clock budget it would be strictly
  worse here. The chapter reports this null result as measured.
- **Why the cyclic one-shot game may be the wrong showcase**: in a one-shot
  cyclic game, PSRO's exact-ish best responses already chase the cycle — each
  iteration adds a different pure strategy, so the population covers the cycle
  for free and there is no BR mode-collapse for a diversity bonus to fix.
  The literature's wins come from sequential games where many distinct
  behaviors earn similar returns. Ch. 12's shootout includes a diverse arm on
  Leduc, which tests exactly that.
- **Caveats**: this is the simplified state-wise bonus (see
  oracles/diverse_ppo.py docstring), one λ value, and small games. A λ sweep
  or the full occupancy-weighted PSD formulation might behave differently; no
  such sweep was run (and none will be run silently — any tuning would be
  recorded here per CLAUDE.md).
- Wall-clock per run: λ=0 ~280-320s, λ=1 ~690-730s (reference server, CPU).
