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
from psrolab.eval.dashboard import make_dashboard_callback, render_dashboard
from psrolab.eval.openspiel_exploitability import full_exploitability
from psrolab.games.openspiel_wrap import OpenSpielGame
from psrolab.meta_solvers import ZeroSumProjectionNash
from psrolab.oracles.ppo import PPOOracle
from psrolab.utils.plotstyle import apply_style

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
    parser.add_argument("--with-dashboard", action="store_true",
                        help="attach eval/dashboard.py callback and emit Fig 7.3 "
                             "(payoff heatmap + 3 metric panels over the reference run)")
    parser.add_argument("--plateau-range", type=str, default="16,19",
                        help="iteration span (inclusive) to shade in Fig 7.3")
    parser.add_argument("--smoke", action="store_true", help="tiny config for CI (<60s)")
    parser.add_argument("--figdir", type=str, default=None,
                        help="override figures directory (ignored when --smoke is set, "
                             "so smoke never writes to tracked paths)")
    parser.add_argument("--plot-only", action="store_true",
                        help="skip PSRO/training; regenerate figures from the committed CSVs")
    args = parser.parse_args()
    if args.smoke:
        args.iterations, args.oracle_episodes, args.eval_episodes = 2, 1000, 100
        args.hidden, args.n_seeds = 32, 1
    plt.switch_backend("Agg")
    apply_style()

    # Calibrated on the reference server (CPU): ~1.6ms per PPO training
    # episode incl. updates; evaluation episodes ~1ms. Default config: ~70 min.
    per_seed_min = (2 * args.iterations * args.oracle_episodes * 1.6e-3
                    + args.iterations**2 * args.eval_episodes * 1e-3) / 60
    print(f"Estimated runtime: ~{max(args.n_seeds * per_seed_min, 0.3):.0f} min "
          f"({args.iterations} iterations, {args.n_seeds} seed(s))", flush=True)

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
    dashboard_history: list[dict] = []
    for s in range(args.n_seeds):
        game = OpenSpielGame("leduc_poker", seed=args.seed + s)
        oracle = PPOOracle(
            total_episodes=args.oracle_episodes, hidden=args.hidden, device=args.device
        )
        t0 = time.perf_counter()
        callbacks = [
            lambda it, pop, meta, g=game, t=t0: {
                "full_exploitability": full_exploitability(g, pop, meta),
                "elapsed_s": time.perf_counter() - t,
            }
        ]
        if args.with_dashboard and s == 0:
            # Seed-0 gets the dashboard for Fig 7.3; extra seeds are pure
            # exploitability curves (Fig 7.1).
            callbacks.insert(0, make_dashboard_callback(include_payoff_history=True))
        result = run_psro(
            game=game,
            oracle=oracle,
            meta_solver=ZeroSumProjectionNash(),
            evaluator=ProfileEvaluator(n_episodes_per_profile=args.eval_episodes),
            n_iterations=args.iterations,
            seed=args.seed + s,
            callbacks=callbacks,
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
        if args.with_dashboard and s == 0:
            dashboard_history = result.history

    csv_path = results_dir / "leduc_nash_ppo.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _fig(rows, figures_dir)

    if args.with_dashboard and dashboard_history:
        _write_dashboard(dashboard_history, results_dir)
        plateau = tuple(int(x) for x in args.plateau_range.split(","))
        render_dashboard(dashboard_history, figures_dir,
                         stem="leduc_dashboard", plateau_range=plateau)
        print(f"Wrote dashboard history + Fig 7.3 leduc_dashboard.{{pdf,png}}",
              flush=True)
    print(f"Wrote {csv_path} and figures to {figures_dir}")


def _write_dashboard(history: list[dict], results_dir: Path) -> None:
    import numpy as np
    cols: list[str] = []
    for e in history:
        for k in e:
            if k == "payoff_table_p0":
                continue
            if k not in cols:
                cols.append(k)
    with open(results_dir / "leduc_dashboard_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for e in history:
            w.writerow({c: e.get(c, "") for c in cols})
    for e in history:
        if "payoff_table_p0" in e and e["payoff_table_p0"] is not None:
            np.save(results_dir / f"leduc_payoff_p0_iter{e['iteration']:02d}.npy",
                    e["payoff_table_p0"])


def _plot_from_csv(results_dir: Path, figures_dir: Path) -> None:
    with open(results_dir / "leduc_nash_ppo.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["seed"] = int(r["seed"])
    _fig(rows, figures_dir)


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
