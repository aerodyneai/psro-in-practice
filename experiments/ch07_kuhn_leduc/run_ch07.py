"""Ch. 7 — PSRO (Nash + PPO) on Leduc poker.

The step up from Kuhn: 936 information states, six-card deck, two betting
rounds. Tabular methods stop being convenient and the PPO oracle earns its
keep. One configuration — Nash meta-solver + PPO oracle — tracked with the
exact full-game exploitability calculator every iteration.

Outputs:
    results/leduc_nash_ppo.csv           per-iteration trace per seed
    figures/leduc_exploitability.{pdf,png}

Requires `pip install -e ".[rl]"`. The default config is a multi-hour run on
the reference server; use --smoke for a <60s CI pass. GPU (auto-detected)
accelerates the PPO update phase; episode collection is python-loop-bound
either way.

Determinism: numpy parts bit-reproducible given --seed; torch seeded per BR
call. CPU runs are exactly reproducible; CUDA runs may vary slightly across
kernels (noted per CLAUDE.md rule 2 — the book's Leduc figure records the
device used).
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt

from psrolab import run_psro
from psrolab.eval import ProfileEvaluator
from psrolab.eval.openspiel_exploitability import full_exploitability
from psrolab.games.openspiel_wrap import OpenSpielGame
from psrolab.meta_solvers import ZeroSumProjectionNash
from psrolab.oracles.ppo import PPOOracle

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--oracle-episodes", type=int, default=60000,
                        help="PPO training episodes per best-response call")
    parser.add_argument("--eval-episodes", type=int, default=1000,
                        help="episodes per payoff-table cell")
    parser.add_argument("--hidden", type=int, default=128, help="PPO MLP width")
    parser.add_argument("--device", type=str, default="auto", help="PPO device")
    parser.add_argument("--smoke", action="store_true", help="tiny config for CI (<60s)")
    args = parser.parse_args()
    if args.smoke:
        args.iterations, args.oracle_episodes, args.eval_episodes = 2, 1000, 100
        args.hidden, args.n_seeds = 32, 1
    plt.switch_backend("Agg")

    # Calibrated on the reference server (CPU): ~1.6ms per PPO training
    # episode incl. updates; evaluation episodes ~1ms. Default config: ~70 min.
    per_seed_min = (2 * args.iterations * args.oracle_episodes * 1.6e-3
                    + args.iterations**2 * args.eval_episodes * 1e-3) / 60
    print(f"Estimated runtime: ~{max(args.n_seeds * per_seed_min, 0.3):.0f} min "
          f"({args.iterations} iterations, {args.n_seeds} seed(s))", flush=True)

    results_dir = HERE / "results"
    figures_dir = HERE / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    rows = []
    for s in range(args.n_seeds):
        game = OpenSpielGame("leduc_poker", seed=args.seed + s)
        oracle = PPOOracle(
            total_episodes=args.oracle_episodes, hidden=args.hidden, device=args.device
        )
        t0 = time.perf_counter()
        result = run_psro(
            game=game,
            oracle=oracle,
            meta_solver=ZeroSumProjectionNash(),
            evaluator=ProfileEvaluator(n_episodes_per_profile=args.eval_episodes),
            n_iterations=args.iterations,
            seed=args.seed + s,
            callbacks=[
                lambda it, pop, meta, g=game, t=t0: {
                    "full_exploitability": full_exploitability(g, pop, meta),
                    "elapsed_s": time.perf_counter() - t,
                }
            ],
        )
        for entry in result.history:
            rows.append(
                {
                    "seed": args.seed + s,
                    "iteration": entry["iteration"],
                    "full_exploitability": f"{entry['full_exploitability']:.6f}",
                    "elapsed_s": f"{entry['elapsed_s']:.1f}",
                }
            )
            print(f"seed {args.seed + s} iter {entry['iteration']:>2}: "
                  f"full_exploitability = {entry['full_exploitability']:.4f} "
                  f"({entry['elapsed_s']:.0f}s)", flush=True)

    csv_path = results_dir / "leduc_nash_ppo.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _fig(rows, figures_dir)
    print(f"Wrote {csv_path} and figures to {figures_dir}")


def _fig(rows: list[dict], figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    seeds = sorted({r["seed"] for r in rows})
    for s in seeds:
        sub = [r for r in rows if r["seed"] == s]
        it = [int(r["iteration"]) + 1 for r in sub]
        ax.semilogy(it, [float(r["full_exploitability"]) for r in sub],
                    marker=".", label=f"seed {s}")
    ax.set_xlabel("PSRO iteration")
    ax.set_ylabel("full_exploitability (OpenSpiel exact)")
    ax.set_title("Leduc poker: PSRO with Nash meta-solver + PPO oracle")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"leduc_exploitability.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
