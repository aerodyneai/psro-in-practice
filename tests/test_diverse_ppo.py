"""Diverse-PPO oracle invariants (Ch. 11).

Skipped without torch/open_spiel (base CI stays green)."""

import numpy as np
import pytest

pyspiel = pytest.importorskip("pyspiel")
torch = pytest.importorskip("torch")

from psrolab.games.base import Population
from psrolab.games.openspiel_wrap import OpenSpielGame, UniformRandomPolicy
from psrolab.oracles.diverse_ppo import DiversePPOOracle, _total_variation
from psrolab.oracles.ppo import PPOOracle


def fresh_population() -> Population:
    """Opponent policies hold internal rng state, so each training run needs
    its own copies — sharing one Population would leak state between runs."""
    return Population(policies=[[UniformRandomPolicy(0)], [UniformRandomPolicy(1)]])


def test_zero_coef_matches_plain_ppo():
    """lambda=0 must reproduce PPOOracle bit-for-bit (same seeds, same net)."""
    game = OpenSpielGame("kuhn_poker")
    meta = [np.ones(1), np.ones(1)]
    plain = PPOOracle(total_episodes=2000, device="cpu").best_response(
        game, 0, fresh_population(), meta, np.random.default_rng(5)
    )
    diverse = DiversePPOOracle(
        diversity_coef=0.0, total_episodes=2000, device="cpu"
    ).best_response(game, 0, fresh_population(), meta, np.random.default_rng(5))
    obs = np.zeros(game.obs_dim)
    np.testing.assert_allclose(
        plain.action_probabilities(obs, [0, 1], 2),
        diverse.action_probabilities(obs, [0, 1], 2),
    )


def test_diversity_bonus_changes_the_policy():
    game = OpenSpielGame("kuhn_poker")
    meta = [np.ones(1), np.ones(1)]
    plain = DiversePPOOracle(
        diversity_coef=0.0, total_episodes=3000, device="cpu"
    ).best_response(game, 0, fresh_population(), meta, np.random.default_rng(5))
    diverse = DiversePPOOracle(
        diversity_coef=5.0, total_episodes=3000, device="cpu"
    ).best_response(game, 0, fresh_population(), meta, np.random.default_rng(5))
    obs = np.zeros(game.obs_dim)
    tv = _total_variation(
        plain.action_probabilities(obs, [0, 1], 2),
        diverse.action_probabilities(obs, [0, 1], 2),
    )
    assert tv > 0.01, "a large diversity bonus should visibly change the policy"


def test_total_variation_bounds():
    assert _total_variation(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 1.0
    assert _total_variation(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.0
