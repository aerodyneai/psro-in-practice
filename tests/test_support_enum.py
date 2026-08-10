"""Support enumeration must agree with the LP solver — the Ch. 2 cross-check."""

import numpy as np

from psrolab.eval import restricted_exploitability
from psrolab.games import MatrixGame
from psrolab.meta_solvers import NashSolverLP, SupportEnumerationSolver, enumerate_nash


def rps() -> MatrixGame:
    a = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    return MatrixGame(payoffs=np.stack([a, -a]))


def matching_pennies() -> MatrixGame:
    a = np.array([[1, -1], [-1, 1]], dtype=float)
    return MatrixGame(payoffs=np.stack([a, -a]))


def test_rps_unique_uniform_nash():
    equilibria = enumerate_nash(rps())
    assert len(equilibria) == 1
    for mixture in equilibria[0]:
        np.testing.assert_allclose(mixture, np.full(3, 1 / 3), atol=1e-9)


def test_matching_pennies_nash():
    x, y = SupportEnumerationSolver().solve(matching_pennies())
    np.testing.assert_allclose(x, [0.5, 0.5], atol=1e-9)
    np.testing.assert_allclose(y, [0.5, 0.5], atol=1e-9)


def test_pure_equilibrium_found_first():
    """Dominant-strategy game: the pure Nash has support size 1 and comes first."""
    a = np.array([[3, 2], [1, 0]], dtype=float)
    x, y = SupportEnumerationSolver().solve(MatrixGame(payoffs=np.stack([a, -a])))
    np.testing.assert_allclose(x, [1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(y, [0.0, 1.0], atol=1e-9)


def test_agrees_with_lp_on_random_zero_sum_games():
    """Both solvers must return zero-exploitability profiles with equal value."""
    rng = np.random.default_rng(0)
    for _ in range(10):
        a = rng.standard_normal((4, 4))
        game = MatrixGame(payoffs=np.stack([a, -a]))
        enum_eq = SupportEnumerationSolver().solve(game)
        lp_eq = NashSolverLP().solve(game)
        assert restricted_exploitability(game, enum_eq) < 1e-6
        assert restricted_exploitability(game, lp_eq) < 1e-6
        enum_value = game.expected_payoffs(enum_eq)[0]
        lp_value = game.expected_payoffs(lp_eq)[0]
        np.testing.assert_allclose(enum_value, lp_value, atol=1e-6)


def test_general_sum_battle_of_sexes():
    """Non-zero-sum sanity check: BoS has 2 pure + 1 mixed equilibrium."""
    payoff_0 = np.array([[2, 0], [0, 1]], dtype=float)
    payoff_1 = np.array([[1, 0], [0, 2]], dtype=float)
    game = MatrixGame(payoffs=np.stack([payoff_0, payoff_1]))
    equilibria = enumerate_nash(game)
    assert len(equilibria) == 3
    mixed = equilibria[2]
    np.testing.assert_allclose(mixed[0], [2 / 3, 1 / 3], atol=1e-9)
    np.testing.assert_allclose(mixed[1], [1 / 3, 2 / 3], atol=1e-9)
