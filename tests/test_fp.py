"""Fictitious play invariants: averages converge on zero-sum games, runs are
bit-reproducible, and the instantaneous play on RPS never settles."""

import numpy as np

from psrolab.baselines import FictitiousPlay
from psrolab.eval import restricted_exploitability
from psrolab.games import MatrixGame


def rps() -> MatrixGame:
    a = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    return MatrixGame(payoffs=np.stack([a, -a]))


def matching_pennies() -> MatrixGame:
    a = np.array([[1, -1], [-1, 1]], dtype=float)
    return MatrixGame(payoffs=np.stack([a, -a]))


def test_averages_converge_on_rps():
    result = FictitiousPlay(rps()).run(n_iterations=3000)
    exploit = restricted_exploitability(rps(), result.final_strategies)
    assert exploit < 0.05
    for mixture in result.final_strategies:
        np.testing.assert_allclose(mixture, np.full(3, 1 / 3), atol=0.05)


def test_averages_converge_on_matching_pennies():
    result = FictitiousPlay(matching_pennies()).run(n_iterations=3000)
    for mixture in result.final_strategies:
        np.testing.assert_allclose(mixture, [0.5, 0.5], atol=0.05)


def test_instantaneous_play_oscillates_on_rps():
    """The pure best responses must keep cycling — all 3 strategies appear in
    the last 100 iterations. (This non-convergence is the Ch. 3 punchline.)"""
    result = FictitiousPlay(rps()).run(n_iterations=1000)
    assert set(result.best_responses[-100:, 0]) == {0, 1, 2}


def test_deterministic():
    a = FictitiousPlay(rps()).run(n_iterations=500)
    b = FictitiousPlay(rps()).run(n_iterations=500)
    np.testing.assert_array_equal(a.best_responses, b.best_responses)
    np.testing.assert_array_equal(a.averages[0], b.averages[0])
