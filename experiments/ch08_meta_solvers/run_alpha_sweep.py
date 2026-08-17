"""Ch. 8 α-sweep — how does alpha-Rank's ranking temperature affect PSRO?

Runs PSRO on Kuhn poker with alpha-Rank meta-solver at
α ∈ {1, 5, 50, 500}, tabular-Q oracle held fixed, 3 seeds each. For each
run report:

  - final full_exploitability (honest metric)
  - stationary-distribution entropy (bits) at the last solve — high α
    concentrates on a single sink; low α is nearly uniform.

The book uses this to motivate α as a **temperature knob**: too low and
you get UniformSolver, too high and you get greedy self-play with the
Markov-Conley sink.

Outputs:
    results/alpha_sweep.csv         one row per (alpha, seed, iteration)
    figures/alpha_sweep.{pdf,png}   two panels: e_G vs iter (per alpha),
                                     final entropy vs alpha
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
from psrolab.meta_solvers import AlphaRankSolver
from psrolab.oracles.tabular_q import TabularQOracle
from psrolab.utils.plotstyle import apply_style

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--oracle-episodes", type=int, default=20000)
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument("--alphas", type=str, default="1,5,50,500")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--figdir", type=str, default=None)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.iterations, args.oracle_episodes = 3, 1500
        args.eval_episodes, args.n_seeds = 200, 1
        args.alphas = "1,50"
    alphas = [float(a) for a in args.alphas.split(",")]
    plt.switch_backend("Agg")
    apply_style()

    suffix = "_smoke" if args.smoke else ""
    results_dir = HERE / f"results{suffix}"
    if args.smoke or not args.figdir:
        figures_dir = HERE / f"figures{suffix}"
    else:
        figures_dir = Path(args.figdir)
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        with open(results_dir / "alpha_sweep.csv") as f:
            rows = list(csv.DictReader(f))
        _fig(rows, figures_dir)
        print(f"Wrote figures to {figures_dir}")
        return

    rows: list[dict] = []
    for alpha in alphas:
        for s in range(args.n_seeds):
            rows += _run(alpha, args.seed + s, args)
            _write(results_dir / "alpha_sweep.csv", rows)
    _fig(rows, figures_dir)
    print(f"Wrote {results_dir}/alpha_sweep.csv and figures to {figures_dir}")


def _run(alpha: float, seed: int, args) -> list[dict]:
    game = OpenSpielGame("kuhn_poker", seed=seed)
    solver = AlphaRankSolver(alpha=alpha)
    t0 = time.perf_counter()
    result = run_psro(
        game=game,
        oracle=TabularQOracle(n_episodes=args.oracle_episodes),
        meta_solver=solver,
        evaluator=ProfileEvaluator(n_episodes_per_profile=args.eval_episodes),
        n_iterations=args.iterations, seed=seed,
        callbacks=[
            lambda it, pop, meta, g=game: {
                "full_exploitability": full_exploitability(g, pop, meta),
                "entropy_p0": _entropy(meta[0]),
                "entropy_p1": _entropy(meta[1]),
            }
        ],
    )
    wall_s = time.perf_counter() - t0
    print(f"alpha={alpha:>5g} seed {seed}: final e_G={result.history[-1]['full_exploitability']:.4f} "
          f"entropy=({result.history[-1]['entropy_p0']:.3f}, "
          f"{result.history[-1]['entropy_p1']:.3f}) ({wall_s:.0f}s)",
          flush=True)
    return [
        {
            "alpha": alpha, "seed": seed,
            "iteration": e["iteration"],
            "full_exploitability": f"{e['full_exploitability']:.6f}",
            "entropy_p0_bits": f"{e['entropy_p0']:.6f}",
            "entropy_p1_bits": f"{e['entropy_p1']:.6f}",
        }
        for e in result.history
    ]


def _entropy(p: np.ndarray) -> float:
    q = p[p > 0]
    return float(-np.sum(q * np.log2(q))) if q.size else 0.0


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def _fig(rows: list[dict], figures_dir: Path) -> None:
    alphas = sorted({float(r["alpha"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: e_G vs iteration, one line per alpha.
    ax = axes[0]
    for alpha in alphas:
        sub = [r for r in rows if float(r["alpha"]) == alpha]
        seeds = sorted({int(r["seed"]) for r in sub})
        curves = np.array([
            [float(r["full_exploitability"]) for r in sub
             if int(r["seed"]) == s]
            for s in seeds
        ])
        it = np.arange(1, curves.shape[1] + 1)
        ax.semilogy(it, curves.mean(axis=0), label=f"α = {alpha:g}")
        if len(seeds) > 1:
            ax.fill_between(it, curves.min(axis=0), curves.max(axis=0),
                            alpha=0.15, lw=0)
    ax.set_xlabel("PSRO iteration"); ax.set_ylabel("full_exploitability")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("α-sweep: exploitability")

    # Panel B: final entropy vs alpha.
    ax = axes[1]
    means0, stds0 = [], []
    means1, stds1 = [], []
    for alpha in alphas:
        sub = [r for r in rows if float(r["alpha"]) == alpha]
        seeds = sorted({int(r["seed"]) for r in sub})
        # take each seed's LAST iteration
        finals0, finals1 = [], []
        for s in seeds:
            per = [r for r in sub if int(r["seed"]) == s]
            per.sort(key=lambda r: int(r["iteration"]))
            finals0.append(float(per[-1]["entropy_p0_bits"]))
            finals1.append(float(per[-1]["entropy_p1_bits"]))
        means0.append(np.mean(finals0)); stds0.append(np.std(finals0))
        means1.append(np.mean(finals1)); stds1.append(np.std(finals1))
    ax.errorbar(alphas, means0, yerr=stds0, marker="o",
                color="tab:blue", label="p0 stationary entropy")
    ax.errorbar(alphas, means1, yerr=stds1, marker="s",
                color="tab:orange", label="p1 stationary entropy")
    ax.set_xscale("log")
    ax.set_xlabel("α (ranking temperature)")
    ax.set_ylabel("stationary distribution entropy (bits)")
    ax.set_title("α ↓ → uniform; α ↑ → sink concentration")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"alpha_sweep.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
