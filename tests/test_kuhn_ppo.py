"""PPO oracle invariants on Kuhn (Ch. 6).

Two levels: (1) a single best-response call must approach the exact
best-response value against the uniform-random opponent (oracle quality,
fast); (2) a short PSRO run must drive full-game exploitability well below
the uniform policy's ~0.458 (loop integration, minutes). The tabular oracle's
0.05-in-15-iterations gate is the stronger end-to-end bound; PPO's is looser
because its best responses are approximate (CLAUDE.md Ch. 6 compares the two).

Skipped without torch/open_spiel (base CI stays green)."""

import numpy as np
import pytest

pyspiel = pytest.importorskip("pyspiel")
torch = pytest.importorskip("torch")

from psrolab import run_psro
from psrolab.eval import ProfileEvaluator
from psrolab.eval.openspiel_exploitability import full_exploitability
from psrolab.games.base import Population
from psrolab.games.openspiel_wrap import OpenSpielGame, UniformRandomPolicy
from psrolab.meta_solvers import ZeroSumProjectionNash
from psrolab.oracles.ppo import PPOOracle


def test_ppo_single_best_response_vs_uniform():
    """PPO's BR against the uniform opponent must capture most of the exact
    BR value (computed with open_spiel's BestResponsePolicy)."""
    from open_spiel.python import policy as os_policy
    from open_spiel.python.algorithms import best_response as os_br

    game = OpenSpielGame("kuhn_poker")
    uniform = os_policy.UniformRandomPolicy(game.raw_game)
    exact = os_br.BestResponsePolicy(game.raw_game, 0, uniform)
    exact_value = exact.value(game.raw_game.new_initial_state())

    population = Population(
        policies=[[UniformRandomPolicy(0)], [UniformRandomPolicy(1)]]
    )
    rng = np.random.default_rng(0)
    oracle = PPOOracle(total_episodes=15000, device="cpu")
    br_policy = oracle.best_response(
        game, 0, population, [np.ones(1), np.ones(1)], rng
    )
    achieved = game.sample_returns(
        [br_policy, UniformRandomPolicy(99)], 20000, np.random.default_rng(1)
    )[0]
    # Exact BR on Kuhn vs uniform is ~0.5; require 80% of it.
    assert achieved > 0.8 * exact_value, f"PPO BR {achieved:.3f} vs exact {exact_value:.3f}"


def test_psro_ppo_kuhn_short_run():
    """6 PSRO iterations with the PPO oracle must cut exploitability to less
    than half of the uniform policy's starting point."""
    game = OpenSpielGame("kuhn_poker")
    result = run_psro(
        game=game,
        oracle=PPOOracle(total_episodes=10000, device="cpu"),
        meta_solver=ZeroSumProjectionNash(),
        evaluator=ProfileEvaluator(n_episodes_per_profile=2000),
        n_iterations=6,
        seed=0,
        callbacks=[
            lambda it, pop, meta: {
                "full_exploitability": full_exploitability(game, pop, meta)
            }
        ],
    )
    exploits = [e["full_exploitability"] for e in result.history]
    assert min(exploits) < 0.22, f"exploitability trajectory: {exploits}"
