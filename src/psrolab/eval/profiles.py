"""Empirical payoff-table evaluation.

Fills Population.payoff_table by simulating every strategy profile that hasn't
been evaluated yet. Caching matters: at iteration k with populations of size k,
only the new 'border' cells (profiles involving at least one new policy) need
simulation — O(k^{n-1}) new cells instead of O(k^n) total. Ch. 14 measures this.

DistributedProfileEvaluator (Ray) is added in Part III with the same interface.
"""

from __future__ import annotations

import itertools

import numpy as np

from psrolab.games.base import Population, SimGame


class ProfileEvaluator:
    def __init__(self, n_episodes_per_profile: int = 100) -> None:
        self.n_episodes = n_episodes_per_profile

    def fill(self, game: SimGame, population: Population, rng: np.random.Generator) -> None:
        sizes = population.sizes()
        n = game.n_players
        new_table = np.full([n] + sizes, np.nan)

        # Copy over cached cells from the previous (smaller) table.
        if population.payoff_table is not None:
            old = population.payoff_table
            old_sizes = list(old.shape[1:])
            slices = tuple([slice(None)] + [slice(0, s) for s in old_sizes])
            new_table[slices] = old

        # Simulate every still-NaN profile.
        for profile in itertools.product(*[range(s) for s in sizes]):
            cell = (slice(None), *profile)
            if not np.isnan(new_table[cell]).any():
                continue
            policies = [population.policies[p][profile[p]] for p in range(n)]
            returns = game.sample_returns(policies, self.n_episodes, rng)
            new_table[cell] = returns

        population.payoff_table = new_table
