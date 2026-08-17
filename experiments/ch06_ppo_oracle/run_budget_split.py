"""Ch. 6 Fig 6.3 — budget-split study.

Fix the *total* episode budget at ``--total-episodes`` per player-seat;
split into (iterations × per-BR episodes) at each element of ``--splits``;
run PSRO with PPO + Nash on Kuhn; record final e_G at each split.

Evaluation-budget episodes are held constant per PSRO cell and excluded
from the split accounting (that's Fig 6.3's whole design: the split is
about training BRs, not about payoff-table quality).

Outputs:
    results/budget_split.csv         final_e_G x split x seed
    figures/budget_split.{pdf,png}   Fig 6.3

Compute: at 300k total x 5 seeds x 5 splits = 25 runs of ~2-6 min each,
~3-4 h wall-clock on the reference server. Also produces a companion
tabular-Q panel at the same splits for comparison.
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
from psrolab.meta_solvers import ZeroSumProjectionNash
from psrolab.oracles.ppo import PPOOracle
from psrolab.oracles.tabular_q import TabularQOracle
from psrolab.utils.plotstyle import apply_style

HERE = Path(__file__).resolve().parent


def _parse_splits(raw: str) -> list[tuple[int, int]]:
    """Parse 'iters:episodes,iters:episodes,...' into a list."""
    return [tuple(int(x) for x in s.split(":")) for s in raw.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--total-episodes", type=int, default=300000,
                        help="fixed total training episodes per player-seat "
                             "(split accounting excludes eval episodes)")
    parser.add_argument("--splits", type=str,
                        default="60:5000,30:10000,15:20000,6:50000,3:100000",
                        help="'iterations:per_br_episodes' pairs; iters*episodes "
                             "must equal --total-episodes")
    parser.add_argument("--eval-episodes", type=int, default=2000,
                        help="episodes per payoff-table cell (held constant, "
                             "excluded from budget accounting)")
    parser.add_argument("--oracles", type=str, default="ppo,tabular_q",
                        help="comma-separated oracles to run in parallel arms")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--figdir", type=str, default=None)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.n_seeds, args.total_episodes = 1, 3000
        args.splits = "6:500,3:1000"
        args.eval_episodes = 100
    plt.switch_backend("Agg")
    apply_style()

    splits = _parse_splits(args.splits)
    for iters, ep in splits:
        assert iters * ep == args.total_episodes, (
            f"split {iters}x{ep}={iters * ep} != total {args.total_episodes}"
        )
    oracles = args.oracles.split(",")

    suffix = "_smoke" if args.smoke else ""
    results_dir = HERE / f"results{suffix}"
    if args.smoke or not args.figdir:
        figures_dir = HERE / f"figures{suffix}"
    else:
        figures_dir = Path(args.figdir)
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        with open(results_dir / "budget_split.csv") as f:
            rows = list(csv.DictReader(f))
        _fig(rows, figures_dir, args.total_episodes)
        print(f"Wrote figures to {figures_dir}")
        return

    rows: list[dict] = []
    for oracle_name in oracles:
        for iters, ep in splits:
            for s in range(args.n_seeds):
                rows.append(_run(oracle_name, iters, ep, args.seed + s, args))
                _write(results_dir / "budget_split.csv", rows)
    _fig(rows, figures_dir, args.total_episodes)
    print(f"Wrote {results_dir}/budget_split.csv and figures to {figures_dir}")


def _run(oracle_name: str, iterations: int, oracle_episodes: int, seed: int,
         args) -> dict:
    game = OpenSpielGame("kuhn_poker", seed=seed)
    if oracle_name == "ppo":
        oracle = PPOOracle(total_episodes=oracle_episodes, device="cpu")
    elif oracle_name == "tabular_q":
        oracle = TabularQOracle(n_episodes=oracle_episodes)
    else:
        raise ValueError(f"unknown oracle {oracle_name}")
    t0 = time.perf_counter()
    result = run_psro(
        game=game, oracle=oracle,
        meta_solver=ZeroSumProjectionNash(),
        evaluator=ProfileEvaluator(n_episodes_per_profile=args.eval_episodes),
        n_iterations=iterations, seed=seed,
        callbacks=[
            lambda it, pop, meta, g=game: {
                "full_exploitability": full_exploitability(g, pop, meta)
            }
        ],
    )
    wall_s = time.perf_counter() - t0
    trace = [(e["iteration"], e["full_exploitability"]) for e in result.history]
    final = trace[-1][1]
    print(f"{oracle_name:>9} split {iterations:>3}x{oracle_episodes:>6} "
          f"seed {seed}: final e_G = {final:.4f} ({wall_s:.0f}s)",
          flush=True)
    return {
        "oracle": oracle_name,
        "iterations": iterations,
        "oracle_episodes": oracle_episodes,
        "total_episodes": iterations * oracle_episodes,
        "seed": seed,
        "final_full_exploitability": f"{final:.6f}",
        "trace": ";".join(f"{it}:{v:.6f}" for it, v in trace),
        "wall_clock_s": f"{wall_s:.1f}",
    }


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def _fig(rows: list[dict], figures_dir: Path, total_episodes: int) -> None:
    oracles: list[str] = []
    for r in rows:
        if r["oracle"] not in oracles:
            oracles.append(r["oracle"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel 1: final e_G vs split.
    ax = axes[0]
    for oracle_name, color in zip(oracles, ("tab:blue", "tab:orange")):
        splits: list[tuple[int, int]] = []
        for r in rows:
            if r["oracle"] != oracle_name:
                continue
            s = (int(r["iterations"]), int(r["oracle_episodes"]))
            if s not in splits:
                splits.append(s)
        splits.sort()
        xs = [s[1] for s in splits]        # per-BR episodes on X
        means, stds = [], []
        for s in splits:
            vals = [float(r["final_full_exploitability"]) for r in rows
                    if r["oracle"] == oracle_name
                    and int(r["iterations"]) == s[0]
                    and int(r["oracle_episodes"]) == s[1]]
            means.append(np.mean(vals)); stds.append(np.std(vals))
            for v in vals:
                ax.scatter([s[1]], [v], color=color, s=10, alpha=0.5)
        ax.errorbar(xs, means, yerr=stds, marker="o", color=color,
                    label=oracle_name, capsize=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("episodes per BR call")
    ax.set_ylabel("final full_exploitability")
    ax.set_title(f"budget-split (total = {total_episodes} episodes/player)")
    ax.legend(frameon=False)

    # Panel 2: e_G trajectories vs cumulative episodes per player.
    ax = axes[1]
    for oracle_name, color in zip(oracles, ("tab:blue", "tab:orange")):
        splits_seen = set()
        for r in rows:
            if r["oracle"] != oracle_name:
                continue
            s = (int(r["iterations"]), int(r["oracle_episodes"]))
            if s in splits_seen:
                continue
            splits_seen.add(s)
            trace = r["trace"]
            if not trace:
                continue
            xs, ys = [], []
            for pair in trace.split(";"):
                it, v = pair.split(":")
                cum = (int(it) + 1) * s[1]
                xs.append(cum); ys.append(max(float(v), 1e-8))
            ax.plot(xs, ys, color=color, alpha=0.45, lw=0.8)
    for oracle_name, color in zip(oracles, ("tab:blue", "tab:orange")):
        ax.plot([], [], color=color, label=oracle_name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("cumulative training episodes per player")
    ax.set_ylabel("full_exploitability")
    ax.set_title("trajectory: what the budget bought")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"budget_split.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
