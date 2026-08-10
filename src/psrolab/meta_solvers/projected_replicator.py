"""Projected replicator dynamics meta-solver (Ch. 8).

The meta-solver used in the original PSRO paper (Lanctot et al. 2017,
"A Unified Game-Theoretic Approach to Multiagent RL", appendix): discrete-time
replicator dynamics on each player's mixture over the empirical game, with a
projection that keeps every strategy's probability at least `gamma` so no
population member goes fully extinct (exploration for the best-response
trainer). Returns the final point of the dynamics.
"""

from __future__ import annotations

import numpy as np

from psrolab.games.base import MatrixGame
from psrolab.meta_solvers.base import MetaSolver


class ProjectedReplicatorSolver(MetaSolver):
    """Euler-discretized replicator dynamics with an exploration floor.

    Args:
        iterations: Euler steps.
        dt: step size.
        gamma: minimum probability per strategy after projection.
    """

    def __init__(
        self, iterations: int = 20000, dt: float = 1e-3, gamma: float = 1e-10
    ) -> None:
        self.iterations = iterations
        self.dt = dt
        self.gamma = gamma

    def solve(self, game: MatrixGame) -> list[np.ndarray]:
        assert game.n_players == 2, "ProjectedReplicatorSolver is 2-player"
        payoff_0, payoff_1 = game.payoffs[0], game.payoffs[1]
        x = np.full(payoff_0.shape[0], 1.0 / payoff_0.shape[0])
        y = np.full(payoff_0.shape[1], 1.0 / payoff_0.shape[1])

        for _ in range(self.iterations):
            u_x = payoff_0 @ y
            u_y = x @ payoff_1
            x = _project(x + self.dt * x * (u_x - float(x @ u_x)), self.gamma)
            y = _project(y + self.dt * y * (u_y - float(y @ u_y)), self.gamma)
        return [x, y]


def _project(mix: np.ndarray, gamma: float) -> np.ndarray:
    """Exact Euclidean projection onto {x : x_i >= gamma, sum(x) = 1}.

    Substitutes z = x - gamma and runs the standard sort-based simplex
    projection for a simplex of total mass 1 - n*gamma. (Clip-and-renormalize
    is NOT correct here: renormalization can push entries back below gamma.)
    """
    n = len(mix)
    total = 1.0 - n * gamma
    assert total > 0, "gamma too large for this many strategies"
    z = mix - gamma
    u = np.sort(z)[::-1]
    cumulative = np.cumsum(u) - total
    counts = np.arange(1, n + 1)
    rho = counts[u - cumulative / counts > 0][-1]
    theta = cumulative[rho - 1] / rho
    return np.clip(z - theta, 0.0, None) + gamma
