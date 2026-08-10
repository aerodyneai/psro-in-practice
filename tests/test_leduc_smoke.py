"""Leduc smoke: the wrapper, PPO oracle, and exploitability converter must
work end-to-end on a game bigger than Kuhn. Small config — this is a
does-it-run test, not a does-it-converge test (that's ch07's job).

Skipped without torch/open_spiel (base CI stays green)."""

import pytest

pyspiel = pytest.importorskip("pyspiel")
torch = pytest.importorskip("torch")

from psrolab import run_psro
from psrolab.eval import ProfileEvaluator
from psrolab.eval.openspiel_exploitability import full_exploitability
from psrolab.games.openspiel_wrap import OpenSpielGame
from psrolab.meta_solvers import ZeroSumProjectionNash
from psrolab.oracles.ppo import PPOOracle


def test_leduc_psro_one_iteration_runs():
    game = OpenSpielGame("leduc_poker")
    assert game.obs_dim > 0
    result = run_psro(
        game=game,
        oracle=PPOOracle(total_episodes=500, hidden=32, device="cpu"),
        meta_solver=ZeroSumProjectionNash(),
        evaluator=ProfileEvaluator(n_episodes_per_profile=50),
        n_iterations=1,
        seed=0,
    )
    assert result.population.sizes() == [2, 2]
    exploit = full_exploitability(game, result.population, result.meta_strategies)
    # Any behavior policy on Leduc has exploitability in (0, ~2.4]; this just
    # pins that the converter produces a finite, sane number on a bigger game.
    assert 0.0 < exploit < 5.0
