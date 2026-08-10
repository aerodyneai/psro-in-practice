"""Meta-solver zoo invariants (Ch. 8) — pure numpy, run on base installs."""

import numpy as np

from psrolab.eval import restricted_exploitability
from psrolab.games import MatrixGame
from psrolab.meta_solvers import (
    AlphaRankSolver,
    ProjectedReplicatorSolver,
    RegretMatchingSolver,
)


def rps() -> MatrixGame:
    a = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    return MatrixGame(payoffs=np.stack([a, -a]))


def dominance_game() -> MatrixGame:
    """Row 0 / column 1 strictly dominate; unique equilibrium (0, 1)."""
    a = np.array([[3, 2], [1, 0]], dtype=float)
    return MatrixGame(payoffs=np.stack([a, -a]))


def test_alpha_rank_symmetric_cycle_is_uniform():
    """RPS's cyclic response graph has the uniform stationary distribution."""
    for mix in AlphaRankSolver().solve(rps()):
        np.testing.assert_allclose(mix, np.full(3, 1 / 3), atol=1e-8)


def test_alpha_rank_mass_on_dominant_profile():
    stationary, profiles = AlphaRankSolver().stationary_distribution(dominance_game())
    assert profiles[int(np.argmax(stationary))] == (0, 1)
    assert stationary.max() > 0.9


def test_alpha_rank_probabilities_are_valid():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 5))  # non-square: catches profile-index bugs
    game = MatrixGame(payoffs=np.stack([a, -a]))
    mixtures = AlphaRankSolver().solve(game)
    assert [len(m) for m in mixtures] == [4, 5]
    for mix in mixtures:
        assert (mix >= 0).all()
        np.testing.assert_allclose(mix.sum(), 1.0)


def test_regret_matching_reaches_nash_on_zero_sum():
    """Averaged RM marginals approach Nash in 2p zero-sum games."""
    for game, nash in [(rps(), np.full(3, 1 / 3))]:
        mixtures = RegretMatchingSolver(iterations=20000).solve(game)
        assert restricted_exploitability(game, mixtures) < 0.02
        np.testing.assert_allclose(mixtures[0], nash, atol=0.05)


def test_regret_matching_dominance():
    mixtures = RegretMatchingSolver(iterations=5000).solve(dominance_game())
    assert mixtures[0][0] > 0.95
    assert mixtures[1][1] > 0.95


def test_projected_replicator_interior_fixed_point_on_rps():
    mixtures = ProjectedReplicatorSolver().solve(rps())
    for mix in mixtures:
        np.testing.assert_allclose(mix, np.full(3, 1 / 3), atol=0.05)


def test_projected_replicator_respects_floor():
    mixtures = ProjectedReplicatorSolver(gamma=1e-3).solve(dominance_game())
    for mix in mixtures:
        assert (mix >= 1e-3 - 1e-12).all()
        np.testing.assert_allclose(mix.sum(), 1.0)
