# Notes — ch08 meta-solver zoo

- **Solver-internal defaults** (constructor defaults, not argparse flags, since
  the zoo's point is comparing solvers as shipped): alpha-Rank `alpha=50,
  m=50`; regret matching `iterations=10000`; projected replicator
  `iterations=20000, dt=1e-3, gamma=1e-10`. All chosen a priori, none tuned
  against the results.
- **Alpha-Rank validation**: `tests/test_alpha_rank_openspiel.py` pins our
  stationary distribution to open_spiel.python.egt.alpharank (same alpha, m)
  to 1e-6 on RPS, a dominance game, and a random non-square game.
- **Runtime table caveat**: alpha-Rank rows stop at size 40 — its state space
  is size² profiles and the dense eigendecomposition is the scaling wall the
  table demonstrates. Uniform's ~0s rows are shown for completeness.
- **RM returns CCE marginals**, not a correlated device; in (projected)
  zero-sum empirical games the averaged marginals approach Nash, which is why
  it is competitive here. Ch. 8 prose covers what marginalization discards.
