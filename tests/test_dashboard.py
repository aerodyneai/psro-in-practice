"""Sanity tests for eval/dashboard.py."""

from __future__ import annotations

import numpy as np

from psrolab.eval.dashboard import (
    br_margin,
    entropy_bits,
    make_dashboard_callback,
    newest_member_mass,
    support_size,
)
from psrolab.games.base import Population


class _FakePolicy:
    pass


def test_support_and_entropy():
    assert support_size(np.array([0.5, 0.5])) == 2
    assert support_size(np.array([1.0, 0.0])) == 1
    assert entropy_bits(np.array([1.0, 0.0])) == 0.0
    assert abs(entropy_bits(np.array([0.5, 0.5])) - 1.0) < 1e-12


def test_newest_member_mass():
    prev = np.array([0.6, 0.4])
    new = np.array([0.5, 0.3, 0.2])
    assert abs(newest_member_mass(prev, new) - 0.2) < 1e-12
    # New with no growth = 0
    assert newest_member_mass(prev, prev) == 0.0


def test_br_margin_two_player():
    payoff = np.zeros((2, 3, 3))
    payoff[0] = np.array([[0.0, -1.0, 1.0],
                          [1.0, 0.0, -1.0],
                          [-1.0, 1.0, 0.0]])
    payoff[1] = -payoff[0]
    meta = [np.array([0.5, 0.5, 0.0]), np.array([0.5, 0.5, 0.0])]
    new_v, inc_v = br_margin(payoff, meta, player=0)
    # incumbent value = uniform(rock,paper) vs uniform(rock,paper) = 0
    assert abs(inc_v) < 1e-12
    # newest (scissors) vs uniform(rock,paper) = 0.5*1 + 0.5*(-1) = 0
    assert abs(new_v) < 1e-12


def test_callback_pipeline():
    cb = make_dashboard_callback()
    pop = Population(policies=[[_FakePolicy()], [_FakePolicy()]])
    pop.payoff_table = np.array([[[0.0]], [[0.0]]])
    entry1 = cb(0, pop, [np.array([1.0]), np.array([1.0])])
    assert entry1["support_size_p0"] == 1
    # After one call, previous_meta is set; next call gets a real newest mass.
    entry2 = cb(1, pop, [np.array([0.4, 0.6]), np.array([0.5, 0.5])])
    assert abs(entry2["newest_member_mass_p0"] - 0.6) < 1e-12
    assert abs(entry2["newest_member_mass_p1"] - 0.5) < 1e-12
