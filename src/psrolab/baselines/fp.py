"""Fictitious play (Brown 1951) on 2-player matrix games (Ch. 3).

The oldest learning-in-games algorithm, and the book's first 'population'
method: each player best-responds to the opponent's *empirical average* of past
play. The instantaneous strategies oscillate forever on cyclic games like RPS;
the time-averages converge to Nash (for 2-player zero-sum). Both facts get a
figure in Ch. 3, and both foreshadow PSRO: FP is what you get when the 'oracle'
is an exact best response and the 'meta-solver' is the empirical average.

Deterministic: pure numpy, ties broken by lowest strategy index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from psrolab.games.base import MatrixGame


@dataclass
class FPResult:
    """Trajectory of a fictitious-play run.

    Attributes:
        averages: per player, array of shape (T, n_strategies) — the empirical
            average strategy AFTER each iteration. averages[p][-1] is the final
            (Nash-converging) answer.
        best_responses: array of shape (T, 2) — the pure strategy each player
            played at each iteration (the oscillating, non-converging object).
    """

    averages: list[np.ndarray]
    best_responses: np.ndarray

    @property
    def final_strategies(self) -> list[np.ndarray]:
        return [avg[-1] for avg in self.averages]


class FictitiousPlay:
    """Simultaneous fictitious play on a 2-player MatrixGame.

    Both players update at every iteration: play the exact best response to the
    opponent's empirical average so far, then fold that action into their own
    average. Initial play is a fixed pure profile so runs are reproducible.
    """

    def __init__(self, game: MatrixGame, initial_actions: tuple[int, int] = (0, 0)) -> None:
        assert game.n_players == 2, "FictitiousPlay is 2-player only"
        self.game = game
        self.initial_actions = initial_actions

    def run(self, n_iterations: int) -> FPResult:
        """Run for `n_iterations` and return the full trajectory."""
        counts = [np.zeros(n) for n in self.game.n_strategies]
        for p, action in enumerate(self.initial_actions):
            counts[p][action] = 1.0

        averages = [np.empty((n_iterations, n)) for n in self.game.n_strategies]
        best_responses = np.empty((n_iterations, 2), dtype=int)

        for t in range(n_iterations):
            mixtures = [c / c.sum() for c in counts]
            for p in range(2):
                br, _ = self.game.best_response_value(p, mixtures)
                best_responses[t, p] = br
            for p in range(2):
                counts[p][best_responses[t, p]] += 1.0
                averages[p][t] = counts[p] / counts[p].sum()

        return FPResult(averages=averages, best_responses=best_responses)
