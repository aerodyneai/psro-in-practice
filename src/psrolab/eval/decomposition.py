"""Transitive / cyclic decomposition of a 2-player zero-sum payoff matrix.

Every antisymmetric matrix A ∈ R^{n×n} admits a unique decomposition into a
*skill* part A_skill = s 1ᵀ − 1 sᵀ (representable as a scalar ranking of the
strategies) and a *cyclic* residual A_cyc = A − A_skill (RPS-like circular
preferences that no ranking can flatten). Every zero-sum matrix's payoff
tensor P (player 0's view) has an antisymmetric part (P − Pᵀ)/2 to which
the same decomposition applies; the *symmetric* part represents correlated
outcomes and vanishes for exact zero-sum games.

The scalar the book uses in Ch. 11 is:

    cyclic_mass_ratio(A) = ‖A_cyc‖_F² / ‖A_antisym‖_F²  ∈ [0, 1]

It is 0 for a pure skill ladder and 1 for a pure RPS-like cycle. The
motivating claim of the chapter is that diversity bonuses only earn their
compute on populations whose empirical meta-game keeps a high cyclic mass
across iterations — a diagnostic a reader can run before spending 2×
compute on a diverse oracle.
"""

from __future__ import annotations

import numpy as np


def antisymmetrize(a: np.ndarray) -> np.ndarray:
    """Return (a − aᵀ) / 2. The residual (a + aᵀ)/2 is discarded here on
    purpose — a proper zero-sum payoff has no symmetric part, and any
    non-zero part reveals evaluator sampling noise."""
    return 0.5 * (a - a.T)


def skill_decomposition(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Skill-vs-cycle split of an antisymmetric matrix.

    Args:
        a: any n×n array. Only its antisymmetric part is decomposed.

    Returns:
        (skill_vector, a_skill, a_cyclic) where
          skill_vector[i] = row-mean of the antisymmetric part of a,
          a_skill[i, j] = skill_vector[i] − skill_vector[j],
          a_cyclic = antisymmetrise(a) − a_skill.
    """
    anti = antisymmetrize(a)
    s = anti.mean(axis=1)
    a_skill = s[:, None] - s[None, :]
    a_cyc = anti - a_skill
    return s, a_skill, a_cyc


def cyclic_mass_ratio(a: np.ndarray) -> float:
    """Fraction of an antisymmetric matrix's squared Frobenius norm coming
    from the cyclic residual (the part no strategy ranking can flatten).

    Returns 0.0 for the degenerate all-zero matrix (Frobenius norm zero, no
    structure to attribute).
    """
    anti = antisymmetrize(a)
    total = float(np.sum(anti * anti))
    if total <= 0.0:
        return 0.0
    _, _, a_cyc = skill_decomposition(a)
    return float(np.sum(a_cyc * a_cyc) / total)


def cyclic_mass_from_payoff_tensor(payoff_tensor: np.ndarray) -> float:
    """Convenience wrapper: pull player 0's view out of an (n_players, k, k)
    payoff tensor and return the cyclic-mass ratio of that antisymmetric part.

    In use as a callback around `run_psro`, pass `population.payoff_table`.
    """
    if payoff_tensor.ndim != 3 or payoff_tensor.shape[0] < 1:
        raise ValueError(
            f"expected (n_players, k, k) payoff tensor, got shape {payoff_tensor.shape}"
        )
    return cyclic_mass_ratio(payoff_tensor[0])
