"""Nash equilibria of 2-player games by support enumeration (Ch. 2).

The 'from scratch' solver: guess which strategies each player uses with positive
probability (the *support*), solve the indifference conditions as a linear
system, and keep the solution if it is a valid equilibrium. Exponential in the
game size — the whole point of Ch. 2 is feeling that wall before the LP solver
(meta_solvers/base.py:NashSolverLP) removes it for the zero-sum case.

Handles general bimatrix (not just zero-sum) games. Restricted to nondegenerate
equilibria, which have equal-size supports (Nisan et al., *Algorithmic Game
Theory*, §3.4) — sufficient for every game used in the book.
"""

from __future__ import annotations

import itertools

import numpy as np

from psrolab.games.base import MatrixGame
from psrolab.meta_solvers.base import MetaSolver


def enumerate_nash(game: MatrixGame, tol: float = 1e-9) -> list[list[np.ndarray]]:
    """Find all nondegenerate Nash equilibria of a 2-player MatrixGame.

    Iterates over support pairs (I, J) with |I| == |J|, smallest supports first.
    For each pair, solves the indifference conditions and keeps solutions that
    are valid probability vectors and best responses to each other.

    Args:
        game: 2-player MatrixGame. Intended for small games (< ~10 strategies
            per player); support pairs grow exponentially.
        tol: numerical tolerance for nonnegativity and best-response checks.

    Returns:
        List of equilibria, each a [x, y] pair of mixture vectors, ordered by
        support size then lexicographically. Duplicate supports yielding the
        same equilibrium are not deduplicated across support sizes.
    """
    assert game.n_players == 2, "support enumeration is 2-player only"
    A, B = game.payoffs[0], game.payoffs[1]  # payoffs to player 0 / player 1
    m, n = A.shape
    equilibria: list[list[np.ndarray]] = []

    for k in range(1, min(m, n) + 1):
        for rows in itertools.combinations(range(m), k):
            for cols in itertools.combinations(range(n), k):
                eq = _solve_support(A, B, list(rows), list(cols), tol)
                if eq is not None:
                    equilibria.append(eq)
    return equilibria


class SupportEnumerationSolver(MetaSolver):
    """MetaSolver adapter: returns the first equilibrium found by enumeration.

    Deterministic (smallest support, lexicographic order). Exists so Ch. 2's
    solver can be plugged into `run_psro` and cross-checked against NashSolverLP
    on identical games.
    """

    def solve(self, game: MatrixGame) -> list[np.ndarray]:
        equilibria = enumerate_nash(game)
        assert equilibria, "no nondegenerate equilibrium found (degenerate game?)"
        return equilibria[0]


def _solve_support(
    A: np.ndarray, B: np.ndarray, rows: list[int], cols: list[int], tol: float
) -> list[np.ndarray] | None:
    """Solve the indifference system for one support pair; validate or reject.

    Conditions (x supported on `rows`, y supported on `cols`):
      player 0 indifferent across `rows`:  (A y)_i = v0  for i in rows
      player 1 indifferent across `cols`:  (x^T B)_j = v1 for j in cols
      x, y are probability vectors; no strategy outside the support does better.
    """
    k = len(rows)
    m, n = A.shape

    # Unknowns [y_cols, v0]: A[rows][:, cols] y - v0 = 0, sum(y) = 1.
    y_full = _indifference_solution(A[np.ix_(rows, cols)], k)
    # Unknowns [x_rows, v1]: B[rows][:, cols]^T x - v1 = 0, sum(x) = 1.
    x_full = _indifference_solution(B[np.ix_(rows, cols)].T, k)
    if y_full is None or x_full is None:
        return None

    x = np.zeros(m)
    y = np.zeros(n)
    x[rows] = x_full
    y[cols] = y_full
    if (x < -tol).any() or (y < -tol).any():
        return None
    x = np.clip(x, 0.0, None)
    y = np.clip(y, 0.0, None)
    x /= x.sum()
    y /= y.sum()

    # Best-response check: no strategy outside the support beats the support value.
    payoffs_0 = A @ y
    payoffs_1 = x @ B
    if payoffs_0.max() > payoffs_0[rows].min() + tol:
        return None
    if payoffs_1.max() > payoffs_1[cols].min() + tol:
        return None
    return [x, y]


def _indifference_solution(M: np.ndarray, k: int) -> np.ndarray | None:
    """Solve M z = v·1, sum(z) = 1 for (z, v); return z or None if singular."""
    lhs = np.zeros((k + 1, k + 1))
    lhs[:k, :k] = M
    lhs[:k, k] = -1.0  # -v column
    lhs[k, :k] = 1.0  # sum(z) = 1
    rhs = np.zeros(k + 1)
    rhs[k] = 1.0
    try:
        sol = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return None
    return sol[:k]
