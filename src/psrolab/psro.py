"""The PSRO loop.

This file is the heart of the book. It must stay short enough to print in full
in Chapter 5 (~target: under 120 lines including docstrings). Every variant in
Part III is expressed as a different Oracle, MetaSolver, or ProfileEvaluator
plugged into THIS loop — if a variant seems to need loop changes, first ask
whether it can be expressed as a component instead.

The loop, in words (Lanctot et al. 2017):
  repeat:
    1. Evaluate the empirical game (fill payoff table over current populations)
    2. Solve the meta-game -> meta-strategies sigma
    3. For each player: train a best response to sigma via the oracle
    4. Add new policies to the populations
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from psrolab.eval.profiles import ProfileEvaluator
from psrolab.games.base import Population, SimGame
from psrolab.meta_solvers.base import MetaSolver
from psrolab.oracles.base import Oracle


@dataclass
class PSROResult:
    population: Population
    meta_strategies: list[np.ndarray]
    history: list[dict] = field(default_factory=list)  # per-iteration metrics


def run_psro(
    game: SimGame,
    oracle: Oracle,
    meta_solver: MetaSolver,
    evaluator: ProfileEvaluator,
    n_iterations: int,
    seed: int = 0,
    callbacks: list[Callable[[int, Population, list[np.ndarray]], dict]] | None = None,
) -> PSROResult:
    """Run PSRO for `n_iterations`.

    callbacks: optional per-iteration metric functions (e.g. exploitability).
    Each receives (iteration, population, meta_strategies) and returns a dict
    merged into that iteration's history entry. Exploitability tracking lives
    in a callback, not the loop — the loop stays solution-concept-agnostic.
    """
    rng = np.random.default_rng(seed)

    population = Population(policies=[[p] for p in game.initial_policies()])
    history: list[dict] = []
    meta_strategies: list[np.ndarray] = [np.ones(1) for _ in range(game.n_players)]

    for it in range(n_iterations):
        # 1. Evaluate empirical game (evaluator caches already-filled cells).
        evaluator.fill(game, population, rng)

        # 2. Solve the meta-game.
        meta_strategies = meta_solver.solve(population.as_matrix_game())

        # 3-4. Best responses, one per player, appended to populations.
        new_policies = []
        for p in range(game.n_players):
            br = oracle.best_response(game, p, population, meta_strategies, rng)
            new_policies.append(br)
        for p, br in enumerate(new_policies):
            population.policies[p].append(br)

        # Metrics.
        entry: dict = {"iteration": it, "population_sizes": population.sizes()}
        for cb in callbacks or []:
            entry.update(cb(it, population, meta_strategies))
        history.append(entry)

    # Final evaluation + solve so returned meta_strategies cover the last BRs.
    evaluator.fill(game, population, rng)
    meta_strategies = meta_solver.solve(population.as_matrix_game())

    return PSROResult(population=population, meta_strategies=meta_strategies, history=history)
