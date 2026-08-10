"""M2 invariants: the OpenSpiel wrapper, the exploitability converter, and the
CLAUDE.md milestone gate — PSRO + tabular Q on Kuhn reaches full-game
exploitability < 0.05 within 15 iterations.

Skipped entirely when open_spiel is not installed (base CI stays green)."""

import numpy as np
import pytest

pyspiel = pytest.importorskip("pyspiel")

from psrolab import run_psro
from psrolab.eval import ProfileEvaluator
from psrolab.eval.openspiel_exploitability import full_exploitability
from psrolab.games.base import Population
from psrolab.games.openspiel_wrap import OpenSpielGame, UniformRandomPolicy
from psrolab.meta_solvers import ZeroSumProjectionNash
from psrolab.oracles.tabular_q import TabularQOracle


def test_wrapper_returns_are_zero_sum_and_reproducible():
    game = OpenSpielGame("kuhn_poker")
    policies = game.initial_policies()
    returns_a = game.sample_returns(policies, 200, np.random.default_rng(7))
    assert returns_a.shape == (2,)
    assert abs(returns_a.sum()) < 1e-12
    policies_b = game.initial_policies()  # fresh policy rngs, same seeds
    returns_b = game.sample_returns(policies_b, 200, np.random.default_rng(7))
    np.testing.assert_array_equal(returns_a, returns_b)


def test_exploitability_of_uniform_matches_openspiel():
    """Our population->behavior-policy conversion must reproduce open_spiel's
    own number for the uniform random policy (a 1-policy 'population')."""
    from open_spiel.python import policy as os_policy
    from open_spiel.python.algorithms import exploitability as os_exploitability

    game = OpenSpielGame("kuhn_poker")
    population = Population(
        policies=[[UniformRandomPolicy(0)], [UniformRandomPolicy(1)]]
    )
    ours = full_exploitability(game, population, [np.ones(1), np.ones(1)])
    reference = os_exploitability.exploitability(
        game.raw_game, os_policy.UniformRandomPolicy(game.raw_game)
    )
    assert ours == pytest.approx(float(reference), abs=1e-9)


def test_psro_tabular_q_kuhn_milestone():
    """CLAUDE.md M2 gate: full-game exploitability < 0.05 within 15 iterations."""
    game = OpenSpielGame("kuhn_poker")
    result = run_psro(
        game=game,
        oracle=TabularQOracle(n_episodes=20000),
        meta_solver=ZeroSumProjectionNash(),
        evaluator=ProfileEvaluator(n_episodes_per_profile=2000),
        n_iterations=15,
        seed=0,
        callbacks=[
            lambda it, pop, meta: {
                "full_exploitability": full_exploitability(game, pop, meta)
            }
        ],
    )
    exploits = [e["full_exploitability"] for e in result.history]
    assert min(exploits) < 0.05, f"exploitability trajectory: {exploits}"
