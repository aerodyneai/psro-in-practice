"""Ch. 10 — Parallel PSRO on Leduc: the wall-clock speedup curve.

Runs the identical PSRO schedule (same seeds, bit-identical results — see
psro_pipeline.py) with 1, 2, 4, and 8 Ray workers, PPO oracle on Leduc, and
plots wall-clock against worker count. The punchline is Amdahl's law: with two
players the best-response phase parallelizes only 2-way, so the curve bends
hard after 2 workers — extra workers only accelerate payoff-table evaluation.
Going further requires pipelining across iterations (Pipeline PSRO, McAleer
et al. 2020), which Ch. 10's prose develops from this exact figure.

Each Ray worker pins torch to a single thread so the speedup measures
parallelism, not thread oversubscription.

Outputs:
    results/speedup.csv                 per-run wall-clock + verification
    figures/speedup.{pdf,png}

Requires `pip install -e ".[rl,distributed]"`.
Determinism: results are bit-identical across worker counts by construction
(verified at runtime and recorded in the CSV); wall-clock obviously varies.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt

from psrolab.eval.openspiel_exploitability import full_exploitability
from psrolab.games.openspiel_wrap import OpenSpielGame
from psrolab.meta_solvers import ZeroSumProjectionNash
from psrolab.psro_pipeline import run_psro_parallel
from psrolab.utils.plotstyle import apply_style

HERE = Path(__file__).resolve().parent


def make_game() -> OpenSpielGame:
    return OpenSpielGame("leduc_poker")


class _OracleFactory:
    """Top-level callable so Ray can pickle it into workers."""

    def __init__(self, episodes: int, hidden: int) -> None:
        self.episodes = episodes
        self.hidden = hidden

    def __call__(self):
        import torch

        torch.set_num_threads(1)
        from psrolab.oracles.ppo import PPOOracle

        return PPOOracle(
            total_episodes=self.episodes, hidden=self.hidden, device="cpu"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=str, default="1,2,4,8")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--oracle-episodes", type=int, default=20000)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--smoke", action="store_true", help="tiny config for CI (<60s)")
    parser.add_argument("--figdir", type=str, default=None,
                        help="override figures directory (ignored when --smoke is set, "
                             "so smoke never writes to tracked paths)")
    parser.add_argument("--plot-only", action="store_true",
                        help="skip PSRO/training; regenerate figures from the committed CSVs")
    args = parser.parse_args()
    if args.smoke:
        args.workers, args.iterations = "1,2", 2
        args.oracle_episodes, args.eval_episodes = 500, 50
    worker_counts = [int(w) for w in args.workers.split(",")]
    plt.switch_backend("Agg")
    apply_style()

    serial_min = (2 * args.iterations * args.oracle_episodes * 1.6e-3
                  + args.iterations**2 * args.eval_episodes * 1e-3) / 60
    print(f"Estimated runtime: ~{sum(max(serial_min / min(w, 2), serial_min / 3) for w in worker_counts):.0f} min "
          f"(workers {worker_counts})", flush=True)

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

    rows = []
    reference_table = None
    for n_workers in worker_counts:
        t0 = time.perf_counter()
        result = run_psro_parallel(
            game_factory=make_game,
            oracle_factory=_OracleFactory(args.oracle_episodes, args.hidden),
            meta_solver=ZeroSumProjectionNash(),
            n_iterations=args.iterations,
            n_workers=n_workers,
            episodes_per_profile=args.eval_episodes,
            seed=args.seed,
        )
        wall_clock_s = time.perf_counter() - t0
        game = make_game()
        exploit = full_exploitability(game, result.population, result.meta_strategies)
        if reference_table is None:
            reference_table = result.population.payoff_table
            identical = True
        else:
            identical = bool(
                (reference_table == result.population.payoff_table).all()
            )
        rows.append(
            {
                "n_workers": n_workers,
                "wall_clock_s": f"{wall_clock_s:.1f}",
                "final_full_exploitability": f"{exploit:.6f}",
                "results_identical_to_first_run": int(identical),
            }
        )
        print(f"{n_workers} workers: {wall_clock_s:.0f}s, final full_exploitability "
              f"{exploit:.4f}, identical={identical}", flush=True)
        _shutdown_ray()

    with open(results_dir / "speedup.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _fig(rows, figures_dir)
    print(f"Wrote {results_dir / 'speedup.csv'} and figures to {figures_dir}")


def _shutdown_ray() -> None:
    """Fresh cluster per worker count so num_cpus is re-applied."""
    import ray

    if ray.is_initialized():
        ray.shutdown()


def _plot_from_csv(results_dir: Path, figures_dir: Path) -> None:
    with open(results_dir / "speedup.csv") as f:
        rows = list(csv.DictReader(f))
    _fig(rows, figures_dir)


def _fig(rows: list[dict], figures_dir: Path) -> None:
    workers = [int(r["n_workers"]) for r in rows]
    times = [float(r["wall_clock_s"]) for r in rows]
    speedup = [times[0] / t for t in times]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(workers, speedup, marker="o", label="measured")
    ax.plot(workers, workers, color="0.6", ls="--", lw=1, label="ideal (linear)")
    ax.plot(workers, [min(w, 2) for w in workers], color="tab:red", ls=":", lw=1,
            label="BR phase cap (2 players)")
    ax.set_xticks(workers)
    ax.set_xlabel("Ray workers")
    ax.set_ylabel("speedup vs 1 worker")
    ax.set_title("Parallel PSRO on Leduc: Amdahl bites at 2 workers")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"speedup.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
