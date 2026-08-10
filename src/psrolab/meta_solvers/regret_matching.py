"""Regret-matching meta-solver (Ch. 8).

Both players run regret matching (Hart & Mas-Colell 2000) against each other
on the empirical game, using exact expected payoffs. The time-averaged joint
play converges to the set of coarse correlated equilibria; in 2-player
zero-sum games the averaged marginals converge to a Nash equilibrium.

As a PSRO meta-solver we return the averaged *marginals* — the loop trains one
best response per player against a product distribution. Ch. 8 discusses what
is lost by marginalizing a correlated device (and why CCE-based PSRO variants
care); the CSV of the ch08 experiment records marginals only, honestly named.
"""

from __future__ import annotations

import numpy as np

from psrolab.games.base import MatrixGame
from psrolab.meta_solvers.base import MetaSolver


class RegretMatchingSolver(MetaSolver):
    """Simultaneous regret matching with averaged strategies.

    Args:
        iterations: RM steps; cost per step is one matrix-vector product per
            player, so this stays cheap even at 10^4 steps.
    """

    def __init__(self, iterations: int = 10000) -> None:
        self.iterations = iterations

    def solve(self, game: MatrixGame) -> list[np.ndarray]:
        assert game.n_players == 2, "RegretMatchingSolver is 2-player"
        payoff_0, payoff_1 = game.payoffs[0], game.payoffs[1]
        m, n = payoff_0.shape
        regrets = [np.zeros(m), np.zeros(n)]
        strategy_sums = [np.zeros(m), np.zeros(n)]
        x, y = np.full(m, 1.0 / m), np.full(n, 1.0 / n)

        for _ in range(self.iterations):
            # Expected payoff of each pure strategy vs the opponent's mixture.
            u_x = payoff_0 @ y
            u_y = x @ payoff_1
            for p, (mix, u) in enumerate([(x, u_x), (y, u_y)]):
                regrets[p] += u - float(mix @ u)
                strategy_sums[p] += mix
            x = _regret_matching_strategy(regrets[0])
            y = _regret_matching_strategy(regrets[1])

        return [s / self.iterations for s in strategy_sums]


def _regret_matching_strategy(regrets: np.ndarray) -> np.ndarray:
    positive = np.clip(regrets, 0.0, None)
    total = positive.sum()
    if total <= 0.0:
        return np.full(len(regrets), 1.0 / len(regrets))
    return positive / total
