# Notes — ch13 pursuit-evasion capstone

- **The first version of this game was degenerate, and the chapter tells that
  story.** With pursuer_speed 0.055 vs evader 0.045, capture_radius 0.08 and
  60 steps, the full 15-iteration PSRO run (2h46m) produced a PURE
  equilibrium: Nash support collapsed to 1 policy per player by iteration 10
  and the best-response probes saturated at ±1.0 (any fresh pursuer beats the
  final evader 100%; no evader survives a fresh pursuer). Scripted probes then
  showed every candidate parameterization with wall-clipping and a
  pursuer-speed edge is capture-dominant (walls corner the evader — the
  discrete lion-and-man effect). Domain design is part of the method.
- **The shipped parameterization** (equal speeds 0.05, capture_radius 0.04,
  max_steps 30) makes capture a prediction problem: a greedy chaser captures a
  random walker 99%, a deterministic fleer 62%, and a wall-aware juking evader
  ~41% (scripted probes in games/pursuit_evasion.py docstring).
- **Reference run** (seed 0, 15 iterations, PPO 20k episodes/BR, 3h35m total):
  Nash support grows 1 → 2 → 3 → 4 → 5 per player (with a dip to 2 at
  iteration 13 — supports fluctuate, the heatmap shows it honestly). Final
  probes are informative, not saturated: a fresh pursuer BR scores −0.872
  against the final meta (the evader mix is nearly unexploitable) while a
  fresh evader BR scores +0.755 (the pursuer mix remains more exploitable);
  the population-tournament proxy ends at 0.035. There is no exact
  exploitability calculator for this game — the CSV column names say exactly
  which proxy each number is.
- **Threading lesson (recorded, since it cost a rerun-length mistake):**
  pinning torch to 1 thread measured 20ms/episode; torch's default threading
  measured ~9ms/episode equivalent. The opposite of the Leduc finding at
  hidden=128 — always measure. `--torch-threads` defaults to 0 (torch default)
  and the runtime estimate uses the measured 9ms constant.
- No game or PPO hyperparameters were tuned against the results beyond the
  domain-balance redesign documented above (which used scripted probes, not
  PSRO outcomes).
