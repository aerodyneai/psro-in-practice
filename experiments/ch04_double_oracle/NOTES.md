# Notes — ch04 double oracle

- **Convergence cap heuristic.** `run_psro` has no early stopping (the loop is
  sacred), so convergence runs use a doubling-restart driver. The initial
  iteration cap is `0.75·n + 6`, chosen after a first full run measured the
  Nash support of iid-Gaussian random zero-sum games at ≈ 2n/3 of strategies
  (mean population at convergence: 8.0/10, 18.8/25, 36.2/50, 67.2/100,
  166.2/250, 327.4/500, 5 seeds). An initial cap of `0.55·n + 4` missed at
  most sizes and doubled the wall-clock (87 min vs ~25 min for the default
  config). No game-theoretic hyperparameters were tuned — only this
  compute-saving cap.
- **Runtime estimate constant.** The startup estimate uses
  `cost ≈ 4.7e-6 · cap³` seconds per run, calibrated from the measured 87.6-min
  first full run (cost is dominated by the Nash LP over growing populations).
  Machine-dependent; treat as order-of-magnitude.
- **Zero-sum projection.** With payoff noise the sampled empirical game is not
  exactly zero-sum; `ZeroSumProjectionNash` projects onto `(A0 − A1)/2` before
  the LP. This is a correctness requirement, not tuning.
