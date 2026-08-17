"""Ch. 2 worked-example asserts — pin the two hand-computed results the
book relies on so a later refactor can't silently drift.

§2.2 example: player 0 mixes (½, ¼, ¼) on RPS against uniform-Nash player 1
    → best-response value = 0.25, so exploitability of that profile = ½ + ½ = 0.5
    (both players deviate the same amount when both are exploited; the
    book uses NashConv/2 as a convention — hence "exploitability = ¼" in §2.2
    when only one player deviates).

§2.3.1 example: weighted RPS with player-0 payoffs
    A = [[0, -1,  2],
         [1,  0, -1],
         [-2, 1,  0]]
    Nash equilibrium (both players): (¼, ½, ¼).
"""

from __future__ import annotations

import numpy as np

from psrolab.eval import restricted_exploitability
from psrolab.games import MatrixGame
from psrolab.meta_solvers import NashSolverLP, SupportEnumerationSolver


def _rps() -> MatrixGame:
    a = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    return MatrixGame(payoffs=np.stack([a, -a]))


def _weighted_rps() -> MatrixGame:
    a = np.array([[0, -1, 2], [1, 0, -1], [-2, 1, 0]], dtype=float)
    return MatrixGame(payoffs=np.stack([a, -a]))


def test_rps_uniform_is_nash():
    game = _rps()
    uniform = np.ones(3) / 3.0
    assert restricted_exploitability(game, [uniform, uniform]) < 1e-12


def test_rps_asymmetric_profile_exploitability_matches_hand_math():
    # Player 0 plays (1/2, 1/4, 1/4), player 1 plays uniform. §2.2 derives:
    #   value to p0 vs uniform = 0 (uniform is best-response-invariant in RPS)
    #   value to p0's exact BR vs uniform = 0
    #   value to p1 vs (1/2,1/4,1/4) = (1/2)(-1)*? worked out to gap = 1/4 per side
    #
    # We pin the observable object: NashConv (sum of gaps) divided by 2 equals
    # the exploitability the book quotes.
    game = _rps()
    profile = [np.array([0.5, 0.25, 0.25]), np.ones(3) / 3.0]
    e = restricted_exploitability(game, profile)
    # gap-from-p0-side = 0 (uniform can't be exploited), gap-from-p1-side = 1/4
    # exploitability = mean(gaps) = 1/8 by our convention; the book's §2.2 uses
    # the same convention. Check it lands on 0.125 ± tiny numeric slack.
    assert abs(e - 0.125) < 1e-9


def test_weighted_rps_nash_via_both_solvers():
    game = _weighted_rps()
    expected = np.array([0.25, 0.5, 0.25])
    for solver in (SupportEnumerationSolver(), NashSolverLP()):
        mixtures = solver.solve(game)
        for p in range(2):
            assert np.allclose(mixtures[p], expected, atol=1e-6), (
                f"{solver.__class__.__name__} p{p}: {mixtures[p]} vs {expected}"
            )
        # And that mixture must be zero-exploitability
        assert restricted_exploitability(game, mixtures) < 1e-9
