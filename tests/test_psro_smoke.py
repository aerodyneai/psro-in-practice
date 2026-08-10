"""End-to-end smoke tests. These must pass on the bare skeleton (numpy+scipy
only, no torch/OpenSpiel) and must keep passing forever — they pin down the
core invariants every later chapter's code depends on."""

import numpy as np
import pytest

from psrolab import run_psro
from psrolab.eval import ProfileEvaluator, restricted_exploitability
from psrolab.games import ExactMatrixOracle, MatrixGame, MatrixGameSim
from psrolab.meta_solvers import NashSolverLP, UniformSolver


def rps() -> MatrixGame:
    a = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    return MatrixGame(payoffs=np.stack([a, -a]))


def test_nash_solver_rps():
    """Nash of RPS is uniform (1/3, 1/3, 1/3)."""
    mixtures = NashSolverLP().solve(rps())
    for m in mixtures:
        np.testing.assert_allclose(m, np.full(3, 1 / 3), atol=1e-6)


def test_psro_recovers_double_oracle_on_rps():
    """PSRO + exact oracle + Nash MSS on a wrapped matrix game == double oracle.
    On RPS it must discover all 3 strategies and reach zero exploitability."""
    game = rps()
    result = run_psro(
        game=MatrixGameSim(game),
        oracle=ExactMatrixOracle(game),
        meta_solver=NashSolverLP(),
        evaluator=ProfileEvaluator(n_episodes_per_profile=1),
        n_iterations=5,
    )
    # Full support discovered:
    assert result.population.sizes() == [6, 6]  # 1 initial + 5 iterations (dups allowed)
    # Meta-strategy over the population must be a Nash of the full game:
    exploit = restricted_exploitability(
        result.population.as_matrix_game(), result.meta_strategies
    )
    assert exploit < 1e-6


def test_uniform_solver_is_fsp_baseline():
    game = rps()
    result = run_psro(
        game=MatrixGameSim(game),
        oracle=ExactMatrixOracle(game),
        meta_solver=UniformSolver(),
        evaluator=ProfileEvaluator(n_episodes_per_profile=1),
        n_iterations=6,
    )
    assert result.population.sizes() == [7, 7]


def test_payoff_table_caching_preserves_values():
    """Cells filled at iteration k must be identical at iteration k+1 (cache hit,
    not re-simulation). Uses a noisy sim: re-simulation would change values."""
    game = rps()
    sim = MatrixGameSim(game, noise_std=1.0)
    evaluator = ProfileEvaluator(n_episodes_per_profile=10)
    rng = np.random.default_rng(0)

    from psrolab.games import Population, PureStrategy

    pop = Population(policies=[[PureStrategy(0, 0)], [PureStrategy(1, 0)]])
    evaluator.fill(sim, pop, rng)
    first_cell = pop.payoff_table[:, 0, 0].copy()

    pop.policies[0].append(PureStrategy(0, 1))
    pop.policies[1].append(PureStrategy(1, 1))
    evaluator.fill(sim, pop, rng)
    np.testing.assert_array_equal(pop.payoff_table[:, 0, 0], first_cell)


def test_exploitability_zero_at_nash():
    game = rps()
    uniform = [np.full(3, 1 / 3)] * 2
    assert restricted_exploitability(game, uniform) == pytest.approx(0.0, abs=1e-12)
