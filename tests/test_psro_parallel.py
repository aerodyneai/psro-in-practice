"""Parallel-loop invariants (Ch. 10).

The serial path (n_workers=0) needs no ray and runs on base installs; it must
reproduce double oracle's convergence like `run_psro` does. The ray path must
produce BIT-IDENTICAL results to the serial path for any worker count — the
SeedSequence-per-cell scheme makes parallelism a wall-clock-only decision."""

import numpy as np
import pytest

from psrolab.eval import restricted_exploitability
from psrolab.games import ExactMatrixOracle, MatrixGame, MatrixGameSim
from psrolab.meta_solvers import ZeroSumProjectionNash
from psrolab.psro_pipeline import run_psro_parallel


def rps() -> MatrixGame:
    a = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    return MatrixGame(payoffs=np.stack([a, -a]))


def _run(n_workers: int, noise_std: float = 0.0):
    game = rps()
    return run_psro_parallel(
        game_factory=lambda: MatrixGameSim(game, noise_std=noise_std),
        oracle_factory=lambda: ExactMatrixOracle(game),
        meta_solver=ZeroSumProjectionNash(),
        n_iterations=5,
        n_workers=n_workers,
        episodes_per_profile=10,
        seed=0,
    )


def test_serial_path_recovers_double_oracle():
    result = _run(n_workers=0)
    assert result.population.sizes() == [6, 6]
    exploit = restricted_exploitability(
        result.population.as_matrix_game(), result.meta_strategies
    )
    assert exploit < 1e-6


def test_serial_path_is_deterministic():
    a, b = _run(0, noise_std=0.5), _run(0, noise_std=0.5)
    np.testing.assert_array_equal(a.population.payoff_table, b.population.payoff_table)


def test_ray_results_bit_identical_to_serial():
    """Worker count must change wall-clock only — never results. Noise makes
    the payoff table depend on the per-cell rng, so this pins the seeding."""
    pytest.importorskip("ray")
    serial = _run(0, noise_std=0.5)
    parallel = _run(2, noise_std=0.5)
    np.testing.assert_array_equal(
        serial.population.payoff_table, parallel.population.payoff_table
    )
    for mix_s, mix_p in zip(serial.meta_strategies, parallel.meta_strategies):
        np.testing.assert_array_equal(mix_s, mix_p)
