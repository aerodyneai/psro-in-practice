"""Cross-validation: our alpha-Rank must match open_spiel.python.egt.alpharank
on shared test cases (same alpha, same population size m), per CLAUDE.md M3.

open_spiel's profile ids are row-major over strategy indices — the same order
as AlphaRankSolver's itertools.product enumeration — so the two stationary
distributions are compared entry by entry.

Skipped without open_spiel (base CI stays green)."""

import numpy as np
import pytest

pytest.importorskip("pyspiel")
egt_alpharank = pytest.importorskip("open_spiel.python.egt.alpharank")

from psrolab.games import MatrixGame
from psrolab.meta_solvers import AlphaRankSolver

ALPHA, M = 25.0, 50


@pytest.mark.parametrize("case", ["rps", "dominance", "random_nonsquare"])
def test_stationary_matches_openspiel(case):
    if case == "rps":
        a = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    elif case == "dominance":
        a = np.array([[3, 2], [1, 0]], dtype=float)
    else:
        a = np.random.default_rng(3).standard_normal((3, 4))

    game = MatrixGame(payoffs=np.stack([a, -a]))
    ours, _profiles = AlphaRankSolver(alpha=ALPHA, m=M).stationary_distribution(game)
    _, _, reference, _, _ = egt_alpharank.compute(
        [a, -a], alpha=ALPHA, m=M, use_inf_alpha=False
    )
    np.testing.assert_allclose(ours, reference, atol=1e-6)
