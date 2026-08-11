"""Pursuit-evasion capstone invariants (Ch. 13) — pure numpy, base CI."""

import numpy as np

from psrolab import run_psro
from psrolab.eval import ProfileEvaluator, restricted_exploitability
from psrolab.games.pursuit_evasion import (
    ACTIONS,
    PursuitEvasionSim,
    RandomWalkPolicy,
)
from psrolab.meta_solvers import ZeroSumProjectionNash
from psrolab.oracles.tabular_q import TabularQOracle


class GreedyChasePolicy:
    """Move along the action best aligned with the opponent direction."""

    def __init__(self, sign: float = 1.0) -> None:
        self.sign = sign  # +1 chase, -1 flee

    def act(self, observation, legal_actions):
        rel = observation[4:6] * self.sign
        return int(np.argmax(ACTIONS @ rel))


def test_returns_are_zero_sum_and_deterministic():
    game = PursuitEvasionSim()
    policies = game.initial_policies()
    a = game.sample_returns(policies, 50, np.random.default_rng(3))
    assert abs(a.sum()) < 1e-12
    b = PursuitEvasionSim().sample_returns(
        PursuitEvasionSim().initial_policies(), 50, np.random.default_rng(3)
    )
    np.testing.assert_array_equal(a, b)


def test_chaser_catches_random_walker():
    game = PursuitEvasionSim()
    returns = game.sample_returns(
        [GreedyChasePolicy(), RandomWalkPolicy(7)], 40, np.random.default_rng(0)
    )
    assert returns[0] > 0.8, f"pursuer should almost always catch: {returns}"


def test_fleeing_evader_survives_stationary_pursuer():
    game = PursuitEvasionSim()

    class Stay:
        def act(self, observation, legal_actions):
            return 0

    returns = game.sample_returns(
        [Stay(), GreedyChasePolicy(sign=-1.0)], 10, np.random.default_rng(0)
    )
    assert returns[1] == 1.0, "evader must always reach the episode cap"


def test_greedy_chase_beats_greedy_flee():
    """Deterministic fleeing corners itself: the chaser holds a clear edge
    even at equal speeds — the evader must mix to do better."""
    game = PursuitEvasionSim()
    returns = game.sample_returns(
        [GreedyChasePolicy(), GreedyChasePolicy(sign=-1.0)],
        100, np.random.default_rng(0),
    )
    assert returns[0] > 0.0


def test_trajectory_recording_shapes():
    game = PursuitEvasionSim()
    _returns, trajectory = game.play_episode(
        game.initial_policies(), np.random.default_rng(1), record=True
    )
    assert trajectory.ndim == 3 and trajectory.shape[1:] == (2, 2)
    assert (trajectory >= 0).all() and (trajectory <= 1).all()


def test_psro_runs_end_to_end():
    """The whole loop on the capstone game, tabular oracle, base deps only."""
    game = PursuitEvasionSim(max_steps=25)
    result = run_psro(
        game=game,
        oracle=TabularQOracle(n_episodes=300),
        meta_solver=ZeroSumProjectionNash(),
        evaluator=ProfileEvaluator(n_episodes_per_profile=30),
        n_iterations=2,
        seed=0,
    )
    assert result.population.sizes() == [3, 3]
    exploit = restricted_exploitability(
        result.population.as_matrix_game(), result.meta_strategies
    )
    assert exploit < 0.05  # Nash of the (noisy) empirical game
