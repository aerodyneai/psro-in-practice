"""Ch. 8 — Meta-solver zoo: five solvers, same game, same oracle.

PSRO on Kuhn poker with the tabular-Q oracle held fixed while the meta-solver
varies: uniform, Nash (LP + zero-sum projection), alpha-Rank, regret matching
(CCE marginals), and projected replicator dynamics. One figure, five curves of
honest full-game exploitability. Second output: the runtime-per-solve table on
random zero-sum empirical games of growing size — practitioners need to know
what each solution concept costs as the population grows.

Outputs:
    results/solver_curves.csv           per-iteration traces per solver/seed
    results/solver_runtimes.csv         solve-time vs empirical-game size
    figures/solver_curves.{pdf,png}
    figures/solver_runtimes.{pdf,png}

Requires `pip install -e ".[rl]"` (open_spiel; torch not needed — the oracle
is tabular). Alpha-Rank's runtime rows stop at size 40: its state space is
size^2 profiles and the dense eigendecomposition beyond that is exactly the
scaling wall the table exists to demonstrate.

Determinism: pure numpy end to end, bit-reproducible given --seed.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from psrolab import run_psro
from psrolab.eval import ProfileEvaluator
from psrolab.eval.openspiel_exploitability import full_exploitability
from psrolab.games import MatrixGame
from psrolab.games.openspiel_wrap import OpenSpielGame
from psrolab.meta_solvers import (
    AlphaRankSolver,
    ProjectedReplicatorSolver,
    RegretMatchingSolver,
    UniformSolver,
    ZeroSumProjectionNash,
)
from psrolab.oracles.tabular_q import TabularQOracle
from psrolab.utils.plotstyle import apply_style

HERE = Path(__file__).resolve().parent

SOLVERS = {
    "uniform": UniformSolver,
    "nash_lp": ZeroSumProjectionNash,
    "alpha_rank": AlphaRankSolver,
    "regret_matching": RegretMatchingSolver,
    "projected_replicator": ProjectedReplicatorSolver,
}
ALPHA_RANK_MAX_SIZE = 40


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--oracle-episodes", type=int, default=20000)
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument("--runtime-sizes", type=str, default="2,5,10,20,40,80")
    parser.add_argument("--smoke", action="store_true", help="tiny config for CI (<60s)")
    parser.add_argument("--figdir", type=str, default=None,
                        help="override figures directory (ignored when --smoke is set, "
                             "so smoke never writes to tracked paths)")
    parser.add_argument("--plot-only", action="store_true",
                        help="skip PSRO/training; regenerate figures from the committed CSVs")
    args = parser.parse_args()
    if args.smoke:
        args.iterations, args.oracle_episodes, args.eval_episodes = 2, 1000, 100
        args.n_seeds, args.runtime_sizes = 1, "2,5,10"
    sizes = [int(s) for s in args.runtime_sizes.split(",")]
    plt.switch_backend("Agg")
    apply_style()

    # ~150s per 15-iteration Kuhn run (measured in ch06); timing study ~1 min.
    est_min = len(SOLVERS) * args.n_seeds * args.iterations * 10 / 60 + 1
    print(f"Estimated runtime: ~{est_min:.0f} min", flush=True)

    suffix = "_smoke" if args.smoke else ""
    results_dir = HERE / f"results{suffix}"
    if args.smoke or not args.figdir:
        figures_dir = HERE / f"figures{suffix}"
    else:
        figures_dir = Path(args.figdir)
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        _plot_from_csv(results_dir, figures_dir)
        print(f"Wrote figures to {figures_dir}")
        return

    curve_rows = _psro_curves(args)
    _write(results_dir / "solver_curves.csv", curve_rows)
    runtime_rows = _runtime_table(sizes, args.seed)
    _write(results_dir / "solver_runtimes.csv", runtime_rows)

    _fig_curves(curve_rows, figures_dir)
    _fig_runtimes(runtime_rows, figures_dir)
    _print_runtime_table(runtime_rows)
    print(f"Wrote CSVs to {results_dir} and figures to {figures_dir}")


def _psro_curves(args) -> list[dict]:
    rows = []
    for solver_name, solver_cls in SOLVERS.items():
        for s in range(args.n_seeds):
            game = OpenSpielGame("kuhn_poker", seed=args.seed + s)
            t0 = time.perf_counter()
            result = run_psro(
                game=game,
                oracle=TabularQOracle(n_episodes=args.oracle_episodes),
                meta_solver=solver_cls(),
                evaluator=ProfileEvaluator(n_episodes_per_profile=args.eval_episodes),
                n_iterations=args.iterations,
                seed=args.seed + s,
                callbacks=[
                    lambda it, pop, meta, g=game: {
                        "full_exploitability": full_exploitability(g, pop, meta)
                    }
                ],
            )
            runtime_s = time.perf_counter() - t0
            for entry in result.history:
                rows.append(
                    {
                        "meta_solver": solver_name,
                        "seed": args.seed + s,
                        "iteration": entry["iteration"],
                        "full_exploitability": f"{entry['full_exploitability']:.6f}",
                        "runtime_s": f"{runtime_s:.1f}",
                    }
                )
            print(f"{solver_name:>22} seed {args.seed + s}: final full_exploitability = "
                  f"{result.history[-1]['full_exploitability']:.4f} ({runtime_s:.0f}s)",
                  flush=True)
    return rows


def _runtime_table(sizes: list[int], seed: int) -> list[dict]:
    """Median wall-clock of solver.solve() on random zero-sum empirical games."""
    rows = []
    for size in sizes:
        a = np.random.default_rng(1000 + seed).standard_normal((size, size))
        game = MatrixGame(payoffs=np.stack([a, -a]))
        for solver_name, solver_cls in SOLVERS.items():
            if solver_name == "alpha_rank" and size > ALPHA_RANK_MAX_SIZE:
                continue  # size^2 profiles: dense eig becomes the wall here
            solver = solver_cls()
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                solver.solve(game)
                times.append(time.perf_counter() - t0)
            rows.append(
                {
                    "meta_solver": solver_name,
                    "empirical_game_size": size,
                    "solve_time_s": f"{np.median(times):.3e}",
                }
            )
    return rows


def _write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_from_csv(results_dir: Path, figures_dir: Path) -> None:
    with open(results_dir / "solver_curves.csv") as f:
        curve_rows = list(csv.DictReader(f))
    for r in curve_rows:
        r["seed"] = int(r["seed"])
        r["iteration"] = int(r["iteration"])
    with open(results_dir / "solver_runtimes.csv") as f:
        runtime_rows = list(csv.DictReader(f))
    _fig_curves(curve_rows, figures_dir)
    _fig_runtimes(runtime_rows, figures_dir)


def _fig_curves(rows: list[dict], figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for solver_name in SOLVERS:
        sub = [r for r in rows if r["meta_solver"] == solver_name]
        seeds = sorted({r["seed"] for r in sub})
        curves = np.array(
            [[float(r["full_exploitability"]) for r in sub if r["seed"] == s]
             for s in seeds]
        )
        it = np.arange(1, curves.shape[1] + 1)
        ax.semilogy(it, curves.mean(axis=0), label=solver_name.replace("_", " "))
        if len(seeds) > 1:
            ax.fill_between(it, curves.min(axis=0), curves.max(axis=0),
                            alpha=0.15, lw=0)
    ax.set_xlabel("PSRO iteration")
    ax.set_ylabel("full_exploitability (OpenSpiel exact)")
    ax.set_title("Kuhn poker, tabular-Q oracle: the meta-solver zoo")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"solver_curves.{ext}", dpi=200)
    plt.close(fig)


def _fig_runtimes(rows: list[dict], figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for solver_name in SOLVERS:
        sub = [r for r in rows if r["meta_solver"] == solver_name]
        ax.loglog([int(r["empirical_game_size"]) for r in sub],
                  [float(r["solve_time_s"]) for r in sub],
                  marker="o", label=solver_name.replace("_", " "))
    sizes = sorted({int(r["empirical_game_size"]) for r in rows})
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("empirical game size (strategies per player)")
    ax.set_ylabel("solve time (s, median of 3)")
    ax.set_title("Meta-solver cost as the population grows")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"solver_runtimes.{ext}", dpi=200)
    plt.close(fig)


def _print_runtime_table(rows: list[dict]) -> None:
    sizes = sorted({int(r["empirical_game_size"]) for r in rows})
    print(f"\n{'solver':>22} " + " ".join(f"{s:>9}" for s in sizes))
    for solver_name in SOLVERS:
        cells = []
        for size in sizes:
            match = [r for r in rows if r["meta_solver"] == solver_name
                     and int(r["empirical_game_size"]) == size]
            cells.append(f"{float(match[0]['solve_time_s']):>9.2e}" if match
                         else f"{'—':>9}")
        print(f"{solver_name:>22} " + " ".join(cells))


if __name__ == "__main__":
    main()
