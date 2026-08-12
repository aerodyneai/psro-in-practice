"""Ch. 14 — PSRO engineering: caching, checkpointing, cost anatomy.

Three demonstrations on the pursuit-evasion capstone with the tabular-Q oracle
(base dependencies only — this chapter is about the loop's plumbing, not the
oracle):

  A. Payoff-table cache hit rate. At iteration k only the O(k) border cells
     (profiles touching a new policy) need simulation out of O(k^2) total —
     the evaluator's caching is what makes PSRO affordable. Measured, not
     asserted.
  B. Checkpoint/resume. A run is stopped after `--kill-at` iterations, its
     Population (policies + payoff table) pickled, reloaded, and resumed.
     Because psro_pipeline keys every cell/BR seed by (iteration, ...), the
     resumed run replays the uninterrupted run's remaining work exactly:
     the final payoff tables are verified BIT-IDENTICAL.
  C. Cost breakdown. Wall-clock per iteration split into evaluation /
     meta-solve / best-response, showing the crossover as the table grows.

Outputs:
    results/cache_hit_rate.csv     figures/cache_hit_rate.{pdf,png}
    results/cost_breakdown.csv     figures/cost_breakdown.{pdf,png}
    results/resume_check.csv       (and a checkpoint.pkl working file)

Determinism: pure numpy; bit-reproducible given --seed.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from psrolab.eval import ProfileEvaluator
from psrolab.games.base import Population, SimGame
from psrolab.games.pursuit_evasion import PursuitEvasionSim
from psrolab.meta_solvers import ZeroSumProjectionNash
from psrolab.oracles.tabular_q import TabularQOracle
from psrolab.psro_pipeline import run_psro_parallel

HERE = Path(__file__).resolve().parent


class InstrumentedEvaluator(ProfileEvaluator):
    """ProfileEvaluator that records per-fill cell counts and wall-clock."""

    def __init__(self, n_episodes_per_profile: int) -> None:
        super().__init__(n_episodes_per_profile)
        self.records: list[dict] = []

    def fill(self, game: SimGame, population: Population, rng) -> None:
        sizes = population.sizes()
        total = int(np.prod(sizes))
        if population.payoff_table is None:
            cached = 0
        else:
            old = population.payoff_table[0]
            cached = int(np.minimum(np.array(old.shape), sizes).prod())
        t0 = time.perf_counter()
        super().fill(game, population, rng)
        self.records.append(
            {
                "total_cells": total,
                "new_cells": total - cached,
                "hit_rate": cached / total if total else 0.0,
                "eval_s": time.perf_counter() - t0,
            }
        )


class TimedSolver(ZeroSumProjectionNash):
    def __init__(self) -> None:
        super().__init__()
        self.times: list[float] = []

    def solve(self, game):
        t0 = time.perf_counter()
        result = super().solve(game)
        self.times.append(time.perf_counter() - t0)
        return result


class TimedOracle(TabularQOracle):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.times: list[float] = []

    def best_response(self, game, player, population, meta_strategies, rng):
        t0 = time.perf_counter()
        result = super().best_response(game, player, population, meta_strategies, rng)
        self.times.append(time.perf_counter() - t0)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--kill-at", type=int, default=6)
    parser.add_argument("--oracle-episodes", type=int, default=3000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--smoke", action="store_true", help="tiny config for CI (<60s)")
    args = parser.parse_args()
    if args.smoke:
        args.iterations, args.kill_at = 4, 2
        args.oracle_episodes, args.eval_episodes = 300, 20
    plt.switch_backend("Agg")
    # Three full-run-equivalents (instrumented + uninterrupted + kill/resume
    # pair); ~3ms per tabular training episode at max_steps=30 (measured).
    est_min = 3 * args.iterations * 2 * args.oracle_episodes * 3e-3 / 60
    print(f"Estimated runtime: ~{max(est_min, 0.3):.0f} min", flush=True)

    suffix = "_smoke" if args.smoke else ""
    results_dir = HERE / f"results{suffix}"
    figures_dir = HERE / f"figures{suffix}"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    evaluator, solver, oracle = _instrumented_run(args)
    _write_cache_csv(results_dir, evaluator)
    _write_breakdown_csv(results_dir, evaluator, solver, oracle)
    _fig_cache(evaluator, figures_dir)
    _fig_breakdown(evaluator, solver, oracle, figures_dir)

    identical = _resume_demo(args, results_dir)
    print(f"checkpoint/resume final tables identical: {identical}")
    assert identical, "resume must replay the uninterrupted run exactly"
    print(f"Wrote CSVs to {results_dir} and figures to {figures_dir}")


def _components(args):
    return (
        InstrumentedEvaluator(args.eval_episodes),
        TimedSolver(),
        TimedOracle(n_episodes=args.oracle_episodes),
    )


def _instrumented_run(args):
    """One plain `run_psro` run through the instrumented components."""
    from psrolab import run_psro

    evaluator, solver, oracle = _components(args)
    run_psro(
        game=PursuitEvasionSim(seed=args.seed),
        oracle=oracle,
        meta_solver=solver,
        evaluator=evaluator,
        n_iterations=args.iterations,
        seed=args.seed,
    )
    return evaluator, solver, oracle


def _write_cache_csv(results_dir: Path, evaluator: InstrumentedEvaluator) -> None:
    with open(results_dir / "cache_hit_rate.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fill_index", "total_cells", "new_cells", "hit_rate"])
        for i, rec in enumerate(evaluator.records):
            writer.writerow([i, rec["total_cells"], rec["new_cells"],
                             f"{rec['hit_rate']:.4f}"])


def _write_breakdown_csv(results_dir: Path, evaluator, solver, oracle) -> None:
    with open(results_dir / "cost_breakdown.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "eval_s", "solve_s", "br_both_players_s"])
        n = len(solver.times) - 1  # final solve is outside the loop
        for it in range(n):
            br = oracle.times[2 * it] + oracle.times[2 * it + 1]
            writer.writerow([it, f"{evaluator.records[it]['eval_s']:.4f}",
                             f"{solver.times[it]:.4f}", f"{br:.4f}"])


def _fig_cache(evaluator: InstrumentedEvaluator, figures_dir: Path) -> None:
    records = evaluator.records
    idx = np.arange(len(records))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(idx, [r["total_cells"] for r in records], color="0.85",
           label="total cells (O(k²))")
    ax.bar(idx, [r["new_cells"] for r in records], color="tab:orange",
           label="simulated this fill (O(k))")
    ax.set_xlabel("fill call (≈ PSRO iteration)")
    ax.set_ylabel("payoff-table cells")
    ax2 = ax.twinx()
    ax2.plot(idx, [r["hit_rate"] for r in records], color="tab:blue", marker=".",
             label="cache hit rate")
    ax2.set_ylabel("cache hit rate")
    ax2.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax2.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title("The payoff-table cache does most of the work")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"cache_hit_rate.{ext}", dpi=200)
    plt.close(fig)


def _fig_breakdown(evaluator, solver, oracle, figures_dir: Path) -> None:
    n = len(solver.times) - 1
    idx = np.arange(n)
    eval_s = np.array([evaluator.records[i]["eval_s"] for i in range(n)])
    solve_s = np.array(solver.times[:n])
    br_s = np.array([oracle.times[2 * i] + oracle.times[2 * i + 1] for i in range(n)])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(idx, br_s, label="best responses (2 players)", color="tab:blue")
    ax.bar(idx, eval_s, bottom=br_s, label="payoff evaluation", color="tab:orange")
    ax.bar(idx, solve_s, bottom=br_s + eval_s, label="meta-solve", color="tab:green")
    ax.set_xlabel("PSRO iteration")
    ax.set_ylabel("seconds")
    ax.set_title("Where each iteration's time goes")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"cost_breakdown.{ext}", dpi=200)
    plt.close(fig)


def _resume_demo(args, results_dir: Path) -> bool:
    """Kill at --kill-at, checkpoint to disk, reload, resume; compare tables."""
    common = {
        "game_factory": PursuitEvasionSim,
        "oracle_factory": lambda: TabularQOracle(n_episodes=args.oracle_episodes),
        "meta_solver": ZeroSumProjectionNash(),
        "n_workers": 0,
        "episodes_per_profile": args.eval_episodes,
        "seed": args.seed,
    }
    uninterrupted = run_psro_parallel(n_iterations=args.iterations, **common)

    partial = run_psro_parallel(n_iterations=args.kill_at, **common)
    checkpoint_path = results_dir / "checkpoint.pkl"
    with open(checkpoint_path, "wb") as f:
        pickle.dump(
            {"policies": partial.population.policies,
             "payoff_table": partial.population.payoff_table,
             "iteration": args.kill_at},
            f,
        )
    with open(checkpoint_path, "rb") as f:
        saved = pickle.load(f)
    resumed = run_psro_parallel(
        n_iterations=args.iterations,
        start_iteration=saved["iteration"],
        initial_population=Population(policies=saved["policies"],
                                      payoff_table=saved["payoff_table"]),
        **common,
    )
    identical = bool(
        np.array_equal(uninterrupted.population.payoff_table,
                       resumed.population.payoff_table)
    )
    with open(results_dir / "resume_check.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["killed_at_iteration", "total_iterations",
                         "final_tables_bit_identical"])
        writer.writerow([args.kill_at, args.iterations, int(identical)])
    return identical


if __name__ == "__main__":
    main()
