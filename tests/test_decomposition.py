"""Sanity tests for eval/decomposition.py's cyclic-mass ratio."""

from __future__ import annotations

import numpy as np

from psrolab.eval.decomposition import cyclic_mass_ratio, skill_decomposition


def _skill_ladder(n: int) -> np.ndarray:
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return (i - j).astype(float)


def _rps(n: int = 3) -> np.ndarray:
    return np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)


def test_pure_skill_ladder_has_zero_cyclic_mass():
    a = _skill_ladder(7)
    assert cyclic_mass_ratio(a) < 1e-12


def test_pure_rps_is_all_cyclic():
    assert abs(cyclic_mass_ratio(_rps()) - 1.0) < 1e-12


def test_skill_plus_cycle_lies_in_the_middle():
    # 5x5 antisymmetric matrix: transitive skill ladder with a 3-cycle grafted
    # among strategies 0, 1, 2. Cyclic mass should be strictly between 0 and 1.
    mix = _skill_ladder(5)
    mix[0, 1] += 2.0; mix[1, 0] -= 2.0
    mix[1, 2] += 2.0; mix[2, 1] -= 2.0
    mix[2, 0] += 2.0; mix[0, 2] -= 2.0
    ratio = cyclic_mass_ratio(mix)
    assert 0.0 < ratio < 1.0


def test_decomposition_sums_back():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((6, 6))
    _, a_skill, a_cyc = skill_decomposition(a)
    reconstructed = a_skill + a_cyc
    anti = 0.5 * (a - a.T)
    assert np.allclose(reconstructed, anti)


def test_all_zeros_gives_zero_without_dividing_by_zero():
    assert cyclic_mass_ratio(np.zeros((4, 4))) == 0.0
