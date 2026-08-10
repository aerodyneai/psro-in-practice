"""Matrix games as SimGames + the exact oracle.

Wrapping a known MatrixGame behind the black-box SimGame interface looks
pointless — that's the point. PSRO run on a wrapped matrix game with an exact
oracle IS the double-oracle algorithm, and must recover the same populations.
This equivalence is the bridge between Part I and Part II, and it gives every
later component an exact-answer test bed. tests/test_psro_smoke.py relies on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from psrolab.games.base import MatrixGame, Policy, Population, SimGame
from psrolab.oracles.base import Oracle


@dataclass(frozen=True)
class PureStrategy:
    """A policy that is just 'always play row/column i of the underlying game'."""

    player: int
    index: int

    def act(self, observation: np.ndarray, legal_actions: Sequence[int]) -> int:
        return self.index


class MatrixGameSim(SimGame):
    """Black-box wrapper around a MatrixGame. Optionally adds Gaussian noise to
    returns so Ch. 4 can show what payoff-estimation error does to the meta-game."""

    def __init__(self, game: MatrixGame, noise_std: float = 0.0) -> None:
        self.game = game
        self.n_players = game.n_players
        self.noise_std = noise_std

    def sample_returns(
        self, policies: Sequence[Policy], n_episodes: int, rng: np.random.Generator
    ) -> np.ndarray:
        profile = tuple(p.index for p in policies)  # type: ignore[attr-defined]
        mean = self.game.payoffs[(slice(None), *profile)].astype(float)
        if self.noise_std == 0.0:
            return mean
        noise = rng.normal(0.0, self.noise_std / np.sqrt(n_episodes), size=mean.shape)
        return mean + noise

    def initial_policies(self) -> list[Policy]:
        return [PureStrategy(player=p, index=0) for p in range(self.n_players)]


class ExactMatrixOracle(Oracle):
    """Exact best response over the FULL underlying matrix game.

    Expands each opponent's meta-strategy (defined over their population) into a
    mixture over the full strategy space, then takes the exact argmax row.
    """

    def __init__(self, game: MatrixGame) -> None:
        self.game = game

    def best_response(
        self,
        game: SimGame,
        player: int,
        population: Population,
        meta_strategies: Sequence[np.ndarray],
        rng: np.random.Generator,
    ) -> Policy:
        full_mixtures: list[np.ndarray] = []
        for q in range(self.game.n_players):
            mix = np.zeros(self.game.n_strategies[q])
            for policy, prob in zip(population.policies[q], meta_strategies[q]):
                mix[policy.index] += prob  # type: ignore[attr-defined]
            full_mixtures.append(mix)
        br_index, _ = self.game.best_response_value(player, full_mixtures)
        return PureStrategy(player=player, index=br_index)
