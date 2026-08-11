"""Self-play and last-K meta-solver invariants (Ch. 12 ablation) — pure numpy."""

import numpy as np

from psrolab import run_psro
from psrolab.baselines import LastKSolver, SelfPlaySolver
from psrolab.eval import ProfileEvaluator
from psrolab.games import ExactMatrixOracle, MatrixGame, MatrixGameSim


def rps() -> MatrixGame:
    a = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    return MatrixGame(payoffs=np.stack([a, -a]))


def test_self_play_mass_on_latest():
    mixtures = SelfPlaySolver().solve(rps())
    for mix in mixtures:
        np.testing.assert_array_equal(mix, [0.0, 0.0, 1.0])


def test_last_k_window():
    game = MatrixGame(payoffs=np.zeros((2, 5, 5)))
    mixtures = LastKSolver(k=3).solve(game)
    np.testing.assert_allclose(mixtures[0], [0, 0, 1 / 3, 1 / 3, 1 / 3])
    short = LastKSolver(k=3).solve(rps())  # window larger than population
    np.testing.assert_allclose(short[0], np.full(3, 1 / 3))


def test_self_play_cycles_on_rps():
    """Naive self-play on RPS chases its own tail: the BR to the latest policy
    cycles rock->paper->scissors forever and the population keeps cycling."""
    game = rps()
    result = run_psro(
        game=MatrixGameSim(game),
        oracle=ExactMatrixOracle(game),
        meta_solver=SelfPlaySolver(),
        evaluator=ProfileEvaluator(n_episodes_per_profile=1),
        n_iterations=7,
    )
    indices = [p.index for p in result.population.policies[0]]
    # Start at rock (0); each BR beats the previous: 1, 2, 0, 1, 2, 0, 1.
    assert indices == [0, 1, 2, 0, 1, 2, 0, 1]
