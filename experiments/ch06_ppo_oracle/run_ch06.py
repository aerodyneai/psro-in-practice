"""Ch. 6 — Oracle x meta-solver grid on Kuhn poker.

Runs PSRO with every combination of {uniform, Nash} meta-solver and
{tabular-Q, PPO} oracle on Kuhn poker, tracking honest full-game
exploitability each iteration via OpenSpiel's exact calculator. The book
figure shows (a) Nash meta-solver beating uniform for the same oracle, and
(b) tabular Q slightly beating PPO on a game this small — neural nets buy
nothing on 12 information states, which is exactly why the book introduces
them on a game where they're not yet needed.

Outputs:
    results/kuhn_grid.csv                 per-iteration traces per combo/seed
    figures/kuhn_grid.{pdf,png}

Requires `pip install -e ".[rl]"` (torch + open_spiel). CPU is fine for Kuhn.

Determinism: numpy parts are bit-reproducible given --seed. The PPO oracle
seeds torch from the run seed; runs are reproducible on CPU, and CUDA (if
auto-detected) may introduce minor kernel nondeterminism — figures in the
book use CPU runs of this script.
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
from psrolab.games.openspiel_wrap import OpenSpielGame
from psrolab.meta_solvers import UniformSolver, ZeroSumProjectionNash
from psrolab.oracles.ppo import PPOOracle
from psrolab.oracles.tabular_q import TabularQOracle

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--oracle-episodes", type=int, default=20000,
                        help="training episodes per best-response call (both oracles)")
    parser.add_argument("--eval-episodes", type=int, default=2000,
                        help="episodes per payoff-table cell")
    parser.add_argument("--device", type=str, default="auto", help="PPO device")
    parser.add_argument("--smoke", action="store_true", help="tiny config for CI (<60s)")
    args = parser.parse_args()
    if args.smoke:
        args.iterations, args.oracle_episodes, args.eval_episodes = 3, 1500, 200
        args.n_seeds = 1
    plt.switch_backend("Agg")

    # PPO dominates the cost; calibrated on the reference server (CPU).
    est_min = args.n_seeds * (
        2 * args.iterations * args.oracle_episodes * 2.5e-3 / 60  # PPO runs
        + args.iterations * 0.15                                  # everything else
    )
    print(f"Estimated runtime: ~{max(est_min, 0.2):.0f} min", flush=True)

    suffix = "_smoke" if args.smoke else ""
    results_dir = HERE / f"results{suffix}"
    figures_dir = HERE / f"figures{suffix}"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    combos = {
        ("uniform", "tabular_q"): lambda: (UniformSolver(), _tabular(args)),
        ("nash", "tabular_q"): lambda: (ZeroSumProjectionNash(), _tabular(args)),
        ("uniform", "ppo"): lambda: (UniformSolver(), _ppo(args)),
        ("nash", "ppo"): lambda: (ZeroSumProjectionNash(), _ppo(args)),
    }
    rows = []
    for (solver_name, oracle_name), build in combos.items():
        for s in range(args.n_seeds):
            game = OpenSpielGame("kuhn_poker", seed=args.seed + s)
            meta_solver, oracle = build()
            t0 = time.perf_counter()
            result = run_psro(
                game=game,
                oracle=oracle,
                meta_solver=meta_solver,
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
                        "oracle": oracle_name,
                        "seed": args.seed + s,
                        "iteration": entry["iteration"],
                        "full_exploitability": f"{entry['full_exploitability']:.6f}",
                        "runtime_s": f"{runtime_s:.1f}",
                    }
                )
            final = result.history[-1]["full_exploitability"]
            print(f"{solver_name:>8} + {oracle_name:<10} seed {args.seed + s}: "
                  f"final full_exploitability = {final:.4f}  ({runtime_s:.0f}s)",
                  flush=True)

    csv_path = results_dir / "kuhn_grid.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _fig_grid(rows, figures_dir)
    print(f"Wrote {csv_path} and figures to {figures_dir}")


def _tabular(args) -> TabularQOracle:
    return TabularQOracle(n_episodes=args.oracle_episodes)


def _ppo(args) -> PPOOracle:
    return PPOOracle(total_episodes=args.oracle_episodes, device=args.device)


def _fig_grid(rows: list[dict], figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    styles = {
        ("uniform", "tabular_q"): ("tab:blue", ":"),
        ("nash", "tabular_q"): ("tab:blue", "-"),
        ("uniform", "ppo"): ("tab:orange", ":"),
        ("nash", "ppo"): ("tab:orange", "-"),
    }
    for (solver_name, oracle_name), (color, ls) in styles.items():
        sub = [r for r in rows if r["meta_solver"] == solver_name
               and r["oracle"] == oracle_name]
        seeds = sorted({r["seed"] for r in sub})
        curves = np.array(
            [[float(r["full_exploitability"]) for r in sub if r["seed"] == s]
             for s in seeds]
        )
        mean = curves.mean(axis=0)
        it = np.arange(1, len(mean) + 1)
        ax.semilogy(it, mean, color=color, ls=ls,
                    label=f"{solver_name} + {oracle_name.replace('_', ' ')}")
        if len(seeds) > 1:
            ax.fill_between(it, curves.min(axis=0), curves.max(axis=0),
                            color=color, alpha=0.15, lw=0)
    ax.set_xlabel("PSRO iteration")
    ax.set_ylabel("full_exploitability (OpenSpiel exact)")
    ax.set_title("Kuhn poker: Nash meta-solver ahead for both oracles;\n"
                 "exact-ish tabular BRs beat PPO's approximate ones")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"kuhn_grid.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
