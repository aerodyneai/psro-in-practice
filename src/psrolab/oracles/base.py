"""Best-response oracles.

An Oracle answers one question: given the current population and a meta-strategy
(a mixture over each opponent's population), produce a new policy for `player`
that does well against that mixture.

The book's oracle progression:
  Ch. 3   ExactMatrixOracle    — argmax over rows of a known payoff matrix
  Ch. 6   PPOOracle            — PPO trained against the sampled mixture opponent
  Ch. 11  DiversePPOOracle     — PPO + behavioral-diversity regularizer
  Ch. 10  (Pipeline PSRO)      — many PPOOracles running concurrently under Ray

Everything downstream (the PSRO loop) is oracle-agnostic. Swapping the oracle
must never require touching the loop — that invariant is enforced by tests.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence

import numpy as np

from psrolab.games.base import Policy, Population, SimGame


class Oracle(abc.ABC):
    @abc.abstractmethod
    def best_response(
        self,
        game: SimGame,
        player: int,
        population: Population,
        meta_strategies: Sequence[np.ndarray],
        rng: np.random.Generator,
    ) -> Policy:
        """Return a (approximate) best response for `player` against opponents
        sampling from `population.policies[q]` with probabilities
        `meta_strategies[q]`, for each q != player.
        """


class MixtureOpponent:
    """Utility: wraps an opponent population + mixture as a single 'opponent
    sampler' an RL oracle can train against. At each episode reset, sample one
    opponent policy per other player from the meta-strategy.

    This tiny class is where 'training against a Nash mixture' becomes concrete
    for readers — highlight it in Ch. 6 prose.
    """

    def __init__(
        self,
        population: Population,
        meta_strategies: Sequence[np.ndarray],
        exclude_player: int,
    ) -> None:
        self.population = population
        self.meta_strategies = meta_strategies
        self.exclude_player = exclude_player

    def sample(self, rng: np.random.Generator) -> dict[int, Policy]:
        """Return {player_index: sampled policy} for every player except the
        one being trained."""
        out: dict[int, Policy] = {}
        for q, (pols, sigma) in enumerate(zip(self.population.policies, self.meta_strategies)):
            if q == self.exclude_player:
                continue
            idx = int(rng.choice(len(pols), p=sigma))
            out[q] = pols[idx]
        return out
