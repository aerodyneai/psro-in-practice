"""Alpha-Rank meta-solver (Ch. 8).

Implements multi-population alpha-Rank from Omidshafiei et al. (2019),
"α-Rank: Multi-Agent Evaluation by Evolution" (Scientific Reports). States are
joint pure-strategy profiles; single-mutant invasions drive a Markov chain
whose transition probabilities are Fermi-style fixation probabilities with
selection intensity `alpha` and population size `m`. The stationary
distribution ranks profiles; marginalizing it per player yields the mixtures
PSRO trains against (as in Muller et al. 2020, "A Generalized Training
Approach for Multiagent Learning", which introduced alpha-Rank-PSRO).

Pure numpy. Validated against open_spiel.python.egt.alpharank on shared test
cases in tests/test_alpha_rank_openspiel.py.
"""

from __future__ import annotations

import itertools

import numpy as np

from psrolab.games.base import MatrixGame
from psrolab.meta_solvers.base import MetaSolver


class AlphaRankSolver(MetaSolver):
    """Alpha-Rank stationary distribution, marginalized per player.

    Args:
        alpha: selection intensity. Large alpha approaches the "winner takes
            the edge" limit the paper recommends; the default is high enough
            for the small empirical games PSRO builds.
        m: finite-population size in the fixation probabilities.
    """

    def __init__(self, alpha: float = 50.0, m: int = 50) -> None:
        self.alpha = alpha
        self.m = m

    def solve(self, game: MatrixGame) -> list[np.ndarray]:
        stationary, profiles = self.stationary_distribution(game)
        marginals = [np.zeros(n) for n in game.n_strategies]
        for prob, profile in zip(stationary, profiles):
            for p, strategy in enumerate(profile):
                marginals[p][strategy] += prob
        # Guard against tiny negative eigenvector noise.
        return [np.clip(mix, 0.0, None) / np.clip(mix, 0.0, None).sum()
                for mix in marginals]

    def stationary_distribution(
        self, game: MatrixGame
    ) -> tuple[np.ndarray, list[tuple[int, ...]]]:
        """The Markov chain's stationary distribution over pure profiles.

        Exposed separately so tests can compare directly against
        open_spiel.python.egt.alpharank's `pi`.
        """
        profiles = list(itertools.product(*[range(n) for n in game.n_strategies]))
        index = {profile: k for k, profile in enumerate(profiles)}
        n_states = len(profiles)
        n_mutations = sum(n - 1 for n in game.n_strategies)
        if n_mutations == 0:  # 1x1 empirical game (PSRO iteration 0)
            return np.ones(1), profiles
        eta = 1.0 / n_mutations

        transition = np.zeros((n_states, n_states))
        for s, profile in enumerate(profiles):
            for p in range(game.n_players):
                incumbent = game.payoffs[(p, *profile)]
                for mutant_strategy in range(game.n_strategies[p]):
                    if mutant_strategy == profile[p]:
                        continue
                    mutant_profile = list(profile)
                    mutant_profile[p] = mutant_strategy
                    mutant = game.payoffs[(p, *mutant_profile)]
                    rho = _fixation_probability(
                        mutant - incumbent, self.alpha, self.m
                    )
                    transition[s, index[tuple(mutant_profile)]] = eta * rho
            transition[s, s] = 1.0 - transition[s].sum()

        return _stationary(transition), profiles


def _fixation_probability(delta: float, alpha: float, m: int) -> float:
    """P(single mutant with fitness advantage `delta` takes over m incumbents).

    rho = (1 - e^{-alpha*delta}) / (1 - e^{-alpha*m*delta}), with the neutral
    limit 1/m at delta = 0 and asymptotes handled explicitly to avoid overflow.
    """
    x = alpha * delta
    if abs(x) < 1e-12:
        return 1.0 / m
    if x * m > 500.0:  # strongly advantageous: denominator -> 1
        return -np.expm1(-x)
    if x * m < -500.0:  # strongly disadvantageous: rho -> 0
        return 0.0
    return np.expm1(-x) / np.expm1(-x * m)


def _stationary(transition: np.ndarray) -> np.ndarray:
    """Left eigenvector of the transition matrix for eigenvalue 1."""
    eigenvalues, eigenvectors = np.linalg.eig(transition.T)
    k = int(np.argmin(np.abs(eigenvalues - 1.0)))
    pi = np.real(eigenvectors[:, k])
    pi = np.clip(pi, 0.0, None) if pi.sum() >= 0 else np.clip(-pi, 0.0, None)
    return pi / pi.sum()
