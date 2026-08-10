"""Nash LP on noisy empirical games via zero-sum projection (Ch. 4).

Sampled payoff tables are never *exactly* zero-sum: each player's returns carry
independent estimation noise, and NashSolverLP (rightly) refuses non-zero-sum
input. Averaging the two players' estimates, a_hat = (payoffs_0 - payoffs_1)/2,
projects back onto the zero-sum subspace and halves the noise variance for
free. Introduced in Ch. 4's noise study; every sampled-payoff PSRO run from
Ch. 6 onward uses this as its meta-solver.
"""

from __future__ import annotations

import numpy as np

from psrolab.games.base import MatrixGame
from psrolab.meta_solvers.base import NashSolverLP


class ZeroSumProjectionNash(NashSolverLP):
    """NashSolverLP applied to the nearest zero-sum game."""

    def solve(self, game: MatrixGame) -> list[np.ndarray]:
        a_hat = 0.5 * (game.payoffs[0] - game.payoffs[1])
        return super().solve(MatrixGame(payoffs=np.stack([a_hat, -a_hat])))
