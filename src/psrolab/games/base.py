"""Core game abstractions.

Design principle for the book: TWO levels of game, kept deliberately separate.

1. `MatrixGame`   — a normal-form game with an explicit payoff tensor.
                    Used in Part I, and as the *empirical/meta game* that PSRO
                    constructs over its strategy populations in Part II+.

2. `SimGame`      — a black-box simulator: give it one policy per player, get
                    back sampled returns. This is the "real" game whose payoff
                    tensor is too large (or undefined) to enumerate. PSRO's whole
                    job is to approximate a SimGame with a growing MatrixGame.

The pedagogical arc of the book is literally the relationship between these two
classes, so keep them small and readable. No cleverness here.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class Policy(Protocol):
    """Anything that maps an observation to an action.

    Part I policies are just row indices wrapped in a class; Part II policies
    are torch modules. The PSRO loop never looks inside a policy — it only
    hands policies to a SimGame or an Oracle. That opacity is the point.
    """

    def act(self, observation: np.ndarray, legal_actions: Sequence[int]) -> int:
        ...


@dataclass
class MatrixGame:
    """An n-player normal-form game.

    payoffs: array of shape (n_players, |S_0|, |S_1|, ..., |S_{n-1}|)
             payoffs[p, i, j, ...] = utility to player p when player 0 plays
             strategy i, player 1 plays j, etc.

    For 2-player zero-sum games, payoffs[1] == -payoffs[0].
    """

    payoffs: np.ndarray
    strategy_labels: list[list[str]] | None = None

    def __post_init__(self) -> None:
        self.n_players = self.payoffs.shape[0]
        self.n_strategies = list(self.payoffs.shape[1:])
        assert len(self.n_strategies) == self.n_players, (
            "payoff tensor rank must be n_players + 1"
        )

    @property
    def is_zero_sum(self) -> bool:
        return self.n_players == 2 and np.allclose(self.payoffs[0], -self.payoffs[1])

    def expected_payoffs(self, mixtures: Sequence[np.ndarray]) -> np.ndarray:
        """Expected utility per player under a mixed-strategy profile.

        mixtures[p] is a probability vector over player p's strategies.
        Returns array of shape (n_players,).
        """
        out = np.empty(self.n_players)
        for p in range(self.n_players):
            u = self.payoffs[p]
            # Contract each player's axis with their mixture, in order.
            for q in range(self.n_players):
                u = np.tensordot(mixtures[q], u, axes=([0], [0]))
            out[p] = float(u)
        return out

    def best_response_value(self, player: int, mixtures: Sequence[np.ndarray]) -> tuple[int, float]:
        """Exact best response for `player` against the others' mixtures.

        Returns (best pure strategy index, its expected value). This is the
        'oracle' of Part I — cheap and exact because the game is enumerable.
        """
        u = self.payoffs[player]
        # Move `player`'s axis to the front, contract everyone else's.
        u = np.moveaxis(u, player, 0)
        others = [q for q in range(self.n_players) if q != player]
        for k, q in enumerate(others):
            # After moveaxis, remaining axes are in original player order minus `player`.
            u = np.tensordot(u, mixtures[q], axes=([1], [0]))
        br = int(np.argmax(u))
        return br, float(u[br])


class SimGame(abc.ABC):
    """A black-box game: policies in, sampled returns out.

    Implementations in the book:
      - Ch. 4:  MatrixGameSim (wraps a MatrixGame — sanity-check that PSRO
                on a wrapped matrix game recovers double oracle exactly)
      - Ch. 7:  KuhnPokerSim, LeducPokerSim (via OpenSpiel)
      - Ch. 13: PursuitEvasionSim (custom 2D capstone domain)
    """

    n_players: int

    @abc.abstractmethod
    def sample_returns(
        self, policies: Sequence[Policy], n_episodes: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Play `n_episodes` with the given joint policy; return mean returns
        per player, shape (n_players,)."""

    @abc.abstractmethod
    def initial_policies(self) -> list[Policy]:
        """One starting policy per player (e.g. uniform random). Seeds the
        PSRO population at iteration 0."""


@dataclass
class Population:
    """Per-player lists of policies plus the empirical payoff tensor over them.

    `payoff_table[p][i0, i1, ...]` mirrors MatrixGame.payoffs but is filled
    lazily by simulation. NaN marks unevaluated cells; the ProfileEvaluator
    (eval/profiles.py) fills them. Caching evaluated cells across iterations
    is one of the big practical wins the book demonstrates in Ch. 14.
    """

    policies: list[list[Policy]] = field(default_factory=list)
    payoff_table: np.ndarray | None = None

    def sizes(self) -> list[int]:
        return [len(pop) for pop in self.policies]

    def as_matrix_game(self) -> MatrixGame:
        assert self.payoff_table is not None and not np.isnan(self.payoff_table).any(), (
            "payoff table has unevaluated cells — run the ProfileEvaluator first"
        )
        return MatrixGame(payoffs=self.payoff_table)
