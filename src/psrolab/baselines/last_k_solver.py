"""Last-K uniform meta-solver (Ch. 12).

Uniform over each player's K most recent policies — the "league of recent
opponents" heuristic common in applied self-play systems (fictitious self-play
truncated to a window). Sits between SelfPlaySolver (K=1) and UniformSolver
(K=inf) in the Ch. 12 ablation.
"""

from __future__ import annotations

import numpy as np

from psrolab.games.base import MatrixGame
from psrolab.meta_solvers.base import MetaSolver


class LastKSolver(MetaSolver):
    """Uniform over the last `k` policies per player (all, if fewer exist)."""

    def __init__(self, k: int = 3) -> None:
        assert k >= 1
        self.k = k

    def solve(self, game: MatrixGame) -> list[np.ndarray]:
        mixtures = []
        for n in game.n_strategies:
            mix = np.zeros(n)
            window = min(self.k, n)
            mix[n - window:] = 1.0 / window
            mixtures.append(mix)
        return mixtures
