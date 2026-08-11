"""Self-play as a degenerate meta-solver (Ch. 12).

Plugging SelfPlaySolver into `run_psro` recovers naive self-play exactly: the
next best response trains only against each opponent's most recent policy.
The whole point of the Ch. 12 ablation is that this is PSRO minus the game
theory — same oracle, same budget, only the meta-solver differs — so any gap
in the shootout table is attributable to the solution concept alone.
"""

from __future__ import annotations

import numpy as np

from psrolab.games.base import MatrixGame
from psrolab.meta_solvers.base import MetaSolver


class SelfPlaySolver(MetaSolver):
    """All probability mass on each player's latest policy."""

    def solve(self, game: MatrixGame) -> list[np.ndarray]:
        mixtures = []
        for n in game.n_strategies:
            mix = np.zeros(n)
            mix[-1] = 1.0
            mixtures.append(mix)
        return mixtures
