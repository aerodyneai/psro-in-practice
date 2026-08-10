"""Meta-strategy solvers (MSS).

Given the empirical game (a MatrixGame over the current populations), an MSS
returns one mixture per player — the distribution the next best response will
be trained against. This is PSRO's plug-in point for solution concepts:

  UniformSolver        -> recovers fictitious self-play        (implemented here)
  NashSolverLP         -> classic PSRO, 2p zero-sum            (implemented here)
  AlphaRankSolver      -> alpha-Rank PSRO                      (Ch. 8, Claude Code)
  CCESolver            -> correlated-equilibrium meta-solvers  (Ch. 8, Claude Code)
  SelfPlaySolver       -> last-policy-only, recovers self-play (Ch. 12, Claude Code)

The two implemented solvers are enough to run the full loop end-to-end and are
the reference against which all later solvers are tested.
"""

from __future__ import annotations

import abc

import numpy as np
from scipy.optimize import linprog

from psrolab.games.base import MatrixGame


class MetaSolver(abc.ABC):
    @abc.abstractmethod
    def solve(self, game: MatrixGame) -> list[np.ndarray]:
        """Return one mixture (probability vector) per player."""


class UniformSolver(MetaSolver):
    """Uniform over each population. PSRO + UniformSolver == fictitious self-play.
    Deliberately trivial — it exists so Ch. 5 can run the loop before Nash is
    introduced, and so benchmarks always include the dumbest baseline."""

    def solve(self, game: MatrixGame) -> list[np.ndarray]:
        return [np.full(n, 1.0 / n) for n in game.n_strategies]


class NashSolverLP(MetaSolver):
    """Exact Nash equilibrium for 2-player zero-sum games via linear programming.

    Standard maximin formulation: player 0 chooses mixture x to maximize v
    s.t. x^T A >= v across all opponent columns (A = payoffs to player 0).
    Player 1's equilibrium mixture is the LP dual, solved symmetrically.

    Ch. 6 derives this LP from scratch; readers should be able to reproduce
    this class from the chapter prose alone.
    """

    def solve(self, game: MatrixGame) -> list[np.ndarray]:
        assert game.is_zero_sum, "NashSolverLP requires a 2-player zero-sum game"
        A = game.payoffs[0]  # shape (m, n): payoff to player 0
        x = self._maximin(A)          # row player's mixture
        y = self._maximin(-A.T)       # column player: maximin of their own payoffs
        return [x, y]

    @staticmethod
    def _maximin(A: np.ndarray) -> np.ndarray:
        m, n = A.shape
        # Variables: [x_1..x_m, v]. Maximize v  ==  minimize -v.
        c = np.zeros(m + 1)
        c[-1] = -1.0
        # Constraints: -(x^T A)_j + v <= 0  for each column j
        A_ub = np.hstack([-A.T, np.ones((n, 1))])
        b_ub = np.zeros(n)
        # sum(x) == 1
        A_eq = np.hstack([np.ones((1, m)), np.zeros((1, 1))])
        b_eq = np.ones(1)
        bounds = [(0.0, None)] * m + [(None, None)]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method="highs")
        assert res.success, f"LP failed: {res.message}"
        x = np.clip(res.x[:m], 0.0, None)
        return x / x.sum()
