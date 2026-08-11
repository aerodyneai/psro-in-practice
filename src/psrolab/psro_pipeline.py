"""Parallel PSRO: the loop from psro.py with Ray-parallel workers (Ch. 10).

Per CLAUDE.md rule 1 this is a SEPARATE loop file: parallel dispatch cannot be
expressed as an Oracle/MetaSolver/Evaluator plugged into `run_psro` without
complicating it. Exactly two steps differ from psro.py's loop (the numbered
comments below mirror its structure):

  step 1 (evaluate):        unevaluated payoff cells are simulated in parallel
                            Ray tasks instead of a serial python loop.
  step 3 (best responses):  the two players' BR trainings run as parallel Ray
                            tasks instead of back-to-back.

Everything else — the meta-solve, the population bookkeeping, the callback
protocol, the result type — is identical, and `n_workers=0` runs the same
schedule without Ray at all.

Determinism: every payoff cell and every BR call gets its own child seed from
a numpy SeedSequence spawn tree keyed by (iteration, cell/player). Results are
therefore bit-identical for ANY worker count — parallelism changes wall-clock
only. (They differ from serial `run_psro`, which threads one rng through
everything; the ch10 experiment pins the any-worker-count invariance.)

This is synchronous parallel PSRO: within an iteration, work parallelizes, but
iterations still synchronize — with 2 players the BR phase caps at 2x and
Amdahl bites quickly (the ch10 figure shows exactly where). Removing that
barrier is Pipeline PSRO proper (McAleer et al. 2020), which additionally
trains a pipeline of future policies against moving lower levels; Ch. 10's
prose covers it, the book's experiments do not need it.

Requires ray (`pip install -e ".[rl,distributed]"`) unless n_workers=0.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence

import numpy as np

from psrolab.games.base import Policy, Population, SimGame
from psrolab.meta_solvers.base import MetaSolver
from psrolab.oracles.base import Oracle
from psrolab.psro import PSROResult


def run_psro_parallel(
    game_factory: Callable[[], SimGame],
    oracle_factory: Callable[[], Oracle],
    meta_solver: MetaSolver,
    n_iterations: int,
    n_workers: int,
    episodes_per_profile: int = 100,
    seed: int = 0,
    callbacks: list[Callable[[int, Population, list[np.ndarray]], dict]] | None = None,
    start_iteration: int = 0,
    initial_population: Population | None = None,
) -> PSROResult:
    """Run parallel PSRO. See module docstring for the delta vs `run_psro`.

    Args:
        game_factory / oracle_factory: constructors, not instances — workers
            build their own copies (pyspiel games don't cross process
            boundaries; oracles may hold device state).
        n_workers: Ray worker count; 0 = run the identical schedule serially
            (no ray import), useful for tests and CPU-only CI.
        start_iteration / initial_population: resume support (Ch. 14). Pass a
            checkpointed Population (policies + payoff table) and the
            iteration it was saved at; because every cell and BR seed is keyed
            by (iteration, ...) rather than drawn from a running stream, the
            resumed run's remaining work replays EXACTLY what the
            uninterrupted run would have done — bit-identical final tables.
            `n_iterations` still counts total iterations from 0.
    """
    if n_workers > 0:
        import ray

        if not ray.is_initialized():
            ray.init(num_cpus=n_workers, include_dashboard=False,
                     ignore_reinit_error=True, log_to_driver=False)

    game = game_factory()
    root_seq = np.random.SeedSequence(seed)
    if initial_population is None:
        population = Population(policies=[[p] for p in game.initial_policies()])
    else:
        population = initial_population
    history: list[dict] = []
    meta_strategies: list[np.ndarray] = [np.ones(1) for _ in range(game.n_players)]

    for it in range(start_iteration, n_iterations):
        # 1. Evaluate empirical game — parallel over unevaluated cells.
        _fill_parallel(game_factory, game, population, episodes_per_profile,
                       root_seq, it, n_workers)

        # 2. Solve the meta-game (serial: milliseconds, nothing to gain).
        meta_strategies = meta_solver.solve(population.as_matrix_game())

        # 3-4. Best responses — parallel over players — then append.
        new_policies = _best_responses_parallel(
            game_factory, oracle_factory, population, meta_strategies,
            root_seq, it, n_workers,
        )
        for p, br in enumerate(new_policies):
            population.policies[p].append(br)

        entry: dict = {"iteration": it, "population_sizes": population.sizes()}
        for cb in callbacks or []:
            entry.update(cb(it, population, meta_strategies))
        history.append(entry)

    _fill_parallel(game_factory, game, population, episodes_per_profile,
                   root_seq, n_iterations, n_workers)
    meta_strategies = meta_solver.solve(population.as_matrix_game())
    return PSROResult(population=population, meta_strategies=meta_strategies,
                      history=history)


def _cell_seed(root: np.random.SeedSequence, iteration: int, profile: tuple) -> int:
    """Deterministic per-cell seed, independent of evaluation order/worker."""
    return int(
        np.random.SeedSequence(
            entropy=root.entropy, spawn_key=(1, iteration, *profile)
        ).generate_state(1)[0]
    )


def _br_seed(root: np.random.SeedSequence, iteration: int, player: int) -> int:
    return int(
        np.random.SeedSequence(
            entropy=root.entropy, spawn_key=(2, iteration, player)
        ).generate_state(1)[0]
    )


def _pending_cells(population: Population, n_players: int) -> list[tuple]:
    sizes = population.sizes()
    new_table = np.full([n_players] + sizes, np.nan)
    if population.payoff_table is not None:
        old = population.payoff_table
        slices = tuple([slice(None)] + [slice(0, s) for s in old.shape[1:]])
        new_table[slices] = old
    population.payoff_table = new_table
    return [
        profile
        for profile in itertools.product(*[range(s) for s in sizes])
        if np.isnan(new_table[(slice(None), *profile)]).any()
    ]


def _evaluate_cell(
    game: SimGame, policies: Sequence[Policy], n_episodes: int, cell_seed: int
) -> np.ndarray:
    return game.sample_returns(
        policies, n_episodes, np.random.default_rng(cell_seed)
    )


def _fill_parallel(
    game_factory, game, population: Population, n_episodes: int,
    root_seq: np.random.SeedSequence, iteration: int, n_workers: int,
) -> None:
    cells = _pending_cells(population, game.n_players)
    if not cells:
        return
    if n_workers == 0:
        for profile in cells:
            policies = [population.policies[p][profile[p]] for p in range(game.n_players)]
            population.payoff_table[(slice(None), *profile)] = _evaluate_cell(
                game, policies, n_episodes, _cell_seed(root_seq, iteration, profile)
            )
        return

    import ray

    @ray.remote
    def evaluate(profile, policies, cell_seed):
        return profile, _evaluate_cell(game_factory(), policies, n_episodes, cell_seed)

    futures = [
        evaluate.remote(
            profile,
            [population.policies[p][profile[p]] for p in range(game.n_players)],
            _cell_seed(root_seq, iteration, profile),
        )
        for profile in cells
    ]
    for profile, returns in ray.get(futures):
        population.payoff_table[(slice(None), *profile)] = returns


def _best_responses_parallel(
    game_factory, oracle_factory, population: Population,
    meta_strategies: list[np.ndarray], root_seq: np.random.SeedSequence,
    iteration: int, n_workers: int,
) -> list[Policy]:
    n_players = len(population.policies)
    seeds = [_br_seed(root_seq, iteration, p) for p in range(n_players)]
    if n_workers == 0:
        return [
            oracle_factory().best_response(
                game_factory(), p, population, meta_strategies,
                np.random.default_rng(seeds[p]),
            )
            for p in range(n_players)
        ]

    import ray

    @ray.remote
    def train(player, policies, meta, br_seed):
        pop = Population(policies=policies)
        return oracle_factory().best_response(
            game_factory(), player, pop, meta, np.random.default_rng(br_seed)
        )

    futures = [
        train.remote(p, population.policies, meta_strategies, seeds[p])
        for p in range(n_players)
    ]
    return list(ray.get(futures))
