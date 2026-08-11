"""Ch. 12 — The shootout: every variant, equal budget, honest table.

The book's flagship experiment. Seven population-learning variants — differing
ONLY in meta-solver and/or oracle — run under an identical budget (same PSRO
iterations, same training episodes per best response, same evaluation
episodes) on two games:

    leduc:   Leduc poker, exact full-game exploitability (OpenSpiel).
    cyclic:  9-strategy generalized RPS as a turn-based matrix game, exact
             exploitability computed directly on the matrix.

Variants: psro_nash, psro_uniform (== fictitious self-play), psro_rm (CCE
marginals), psro_alpharank, psro_diverse (Nash + diversity-bonus PPO),
last_k (uniform over last 3), self_play (latest policy only). A variant may
only be declared better if better under this equal budget (CLAUDE.md rule 6);
budgets and wall-clock are recorded per run in the CSV.

Outputs:
    results/shootout.csv           one row per (game, variant, seed)
    results/shootout_summary.csv   mean ± std final exploitability
    results/shootout_table.tex     the book's LaTeX table (booktabs)
    figures/shootout.{pdf,png}     bar chart with per-seed scatter

Requires `pip install -e ".[rl]"`. Full config is an overnight-class run
(~6-8 h on the reference server, CPU); --smoke covers the code path in <60s.
Determinism: numpy + torch seeded per run; CPU throughout.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from psrolab import run_psro
from psrolab.baselines import LastKSolver, SelfPlaySolver
from psrolab.eval import ProfileEvaluator, restricted_exploitability
from psrolab.eval.openspiel_exploitability import full_exploitability
from psrolab.games import MatrixGame, Population
from psrolab.games.openspiel_wrap import OpenSpielGame
from psrolab.meta_solvers import (
    AlphaRankSolver,
    RegretMatchingSolver,
    UniformSolver,
    ZeroSumProjectionNash,
)
from psrolab.oracles.diverse_ppo import DiversePPOOracle
from psrolab.oracles.ppo import PPOOracle

HERE = Path(__file__).resolve().parent

VARIANTS = ["psro_nash", "psro_uniform", "psro_rm", "psro_alpharank",
            "psro_diverse", "last_k", "self_play"]


def cyclic_matrix(n: int = 9) -> np.ndarray:
    a = np.zeros((n, n))
    for offset in range(1, n // 2 + 1):
        for i in range(n):
            a[(i + offset) % n, i] = 1.0
            a[i, (i + offset) % n] = -1.0
    return a


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--variants", type=str, default=",".join(VARIANTS))
    parser.add_argument("--games", type=str, default="cyclic,leduc")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--oracle-episodes", type=int, default=15000)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--diversity-coef", type=float, default=1.0)
    parser.add_argument("--last-k", type=int, default=3)
    parser.add_argument("--smoke", action="store_true", help="tiny config for CI (<60s)")
    args = parser.parse_args()
    if args.smoke:
        args.n_seeds, args.iterations = 1, 2
        args.oracle_episodes, args.eval_episodes = 500, 50
        args.variants, args.games = "psro_nash,self_play", "cyclic"
    variants = args.variants.split(",")
    games = args.games.split(",")
    plt.switch_backend("Agg")

    per_run_min = {"leduc": args.iterations * args.oracle_episodes * 2 * 1.6e-3 / 60 + 2,
                   "cyclic": args.iterations * args.oracle_episodes * 2 * 0.4e-3 / 60 + 1}
    est_h = sum(per_run_min.get(g, 5) for g in games) * len(variants) * args.n_seeds / 60
    print(f"Estimated runtime: ~{est_h:.1f} h "
          f"({len(variants)} variants x {args.n_seeds} seeds x {games})", flush=True)

    results_dir = HERE / "results"
    figures_dir = HERE / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    rows = []
    for game_name in games:
        for variant in variants:
            for s in range(args.n_seeds):
                rows.append(_run(game_name, variant, args.seed + s, args))
                _append_csv(results_dir / "shootout.csv", rows)  # crash-safe
    summary = _summarize(rows)
    _write_summary(results_dir, summary)
    _write_latex(results_dir / "shootout_table.tex", summary, games, args)
    _fig(rows, games, figures_dir)
    _print_summary(summary, games)
    print(f"Wrote CSVs + LaTeX to {results_dir} and figures to {figures_dir}")


def _build_variant(variant: str, args, game_name: str):
    hidden = 128 if game_name == "leduc" else 64
    ppo = {"total_episodes": args.oracle_episodes, "hidden": hidden, "device": "cpu"}
    solvers = {
        "psro_nash": (ZeroSumProjectionNash(), PPOOracle(**ppo)),
        "psro_uniform": (UniformSolver(), PPOOracle(**ppo)),
        "psro_rm": (RegretMatchingSolver(), PPOOracle(**ppo)),
        "psro_alpharank": (AlphaRankSolver(), PPOOracle(**ppo)),
        "psro_diverse": (
            ZeroSumProjectionNash(),
            DiversePPOOracle(diversity_coef=args.diversity_coef, **ppo),
        ),
        "last_k": (LastKSolver(k=args.last_k), PPOOracle(**ppo)),
        "self_play": (SelfPlaySolver(), PPOOracle(**ppo)),
    }
    return solvers[variant]


def _run(game_name: str, variant: str, seed: int, args) -> dict:
    meta_solver, oracle = _build_variant(variant, args, game_name)
    if game_name == "leduc":
        game = OpenSpielGame("leduc_poker", seed=seed)
        matrix = None
    else:
        import pyspiel

        a = cyclic_matrix()
        matrix = MatrixGame(payoffs=np.stack([a, -a]))
        game = OpenSpielGame(
            pyspiel.convert_to_turn_based(
                pyspiel.create_matrix_game(a.tolist(), (-a).tolist())
            ),
            seed=seed,
        )
    t0 = time.perf_counter()
    result = run_psro(
        game=game,
        oracle=oracle,
        meta_solver=meta_solver,
        evaluator=ProfileEvaluator(n_episodes_per_profile=args.eval_episodes),
        n_iterations=args.iterations,
        seed=seed,
    )
    wall_clock_s = time.perf_counter() - t0
    if matrix is None:
        exploit = full_exploitability(game, result.population, result.meta_strategies)
    else:
        exploit = _matrix_exploitability(game, matrix, result.population,
                                         result.meta_strategies)
    print(f"{game_name:>6} {variant:>15} seed {seed}: final full_exploitability = "
          f"{exploit:.4f} ({wall_clock_s:.0f}s)", flush=True)
    return {
        "game": game_name,
        "variant": variant,
        "seed": seed,
        "final_full_exploitability": f"{exploit:.6f}",
        "budget_iterations": args.iterations,
        "budget_train_episodes": 2 * args.iterations * args.oracle_episodes,
        "budget_eval_episodes_per_cell": args.eval_episodes,
        "wall_clock_s": f"{wall_clock_s:.1f}",
    }


def _matrix_exploitability(
    game: OpenSpielGame, matrix: MatrixGame, population: Population,
    meta: list[np.ndarray],
) -> float:
    n = matrix.n_strategies[0]
    state = game.raw_game.new_initial_state()
    observations, legals = [], []
    for player in (0, 1):
        observations.append(game.observation(state, player))
        legals.append(list(state.legal_actions()))
        if player == 0:
            state = state.child(state.legal_actions()[0])
    mixtures = []
    for player in (0, 1):
        mix = np.zeros(n)
        for policy, prob in zip(population.policies[player], meta[player]):
            if prob <= 1e-12:
                continue
            mix += prob * policy.action_probabilities(
                observations[player], legals[player], n
            )
        mixtures.append(mix / mix.sum())
    return restricted_exploitability(matrix, mixtures)


def _append_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict]) -> dict:
    summary: dict = {}
    for row in rows:
        key = (row["game"], row["variant"])
        summary.setdefault(key, []).append(float(row["final_full_exploitability"]))
    return {
        key: (float(np.mean(vals)), float(np.std(vals)), len(vals))
        for key, vals in summary.items()
    }


def _write_summary(results_dir: Path, summary: dict) -> None:
    with open(results_dir / "shootout_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["game", "variant", "mean_full_exploitability",
                         "std_full_exploitability", "n_seeds"])
        for (game_name, variant), (mean, std, n) in sorted(summary.items()):
            writer.writerow([game_name, variant, f"{mean:.6f}", f"{std:.6f}", n])


def _write_latex(path: Path, summary: dict, games: list[str], args) -> None:
    lines = [
        "% Auto-generated by experiments/ch12_shootout/run_ch12.py — do not edit.",
        (f"% Budget: {args.iterations} iterations x 2 players x "
         f"{args.oracle_episodes} episodes; {args.n_seeds} seeds."),
        r"\begin{tabular}{l" + "c" * len(games) + "}",
        r"\toprule",
        "variant & " + " & ".join(games) + r" \\",
        r"\midrule",
    ]
    best = {g: min(mean for (gn, _), (mean, _, _) in summary.items() if gn == g)
            for g in games}
    for variant in VARIANTS:
        cells = []
        for game_name in games:
            entry = summary.get((game_name, variant))
            if entry is None:
                cells.append("---")
                continue
            mean, std, _ = entry
            text = f"{mean:.3f} $\\pm$ {std:.3f}"
            cells.append(rf"\textbf{{{text}}}" if mean == best[game_name] else text)
        if any(c != "---" for c in cells):
            lines.append(variant.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))


def _fig(rows: list[dict], games: list[str], figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(games), figsize=(5.5 * len(games), 4.2),
                             squeeze=False)
    for ax, game_name in zip(axes[0], games):
        sub = [r for r in rows if r["game"] == game_name]
        variants = [v for v in VARIANTS if any(r["variant"] == v for r in sub)]
        means, stds = [], []
        for i, variant in enumerate(variants):
            vals = [float(r["final_full_exploitability"]) for r in sub
                    if r["variant"] == variant]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            ax.scatter([i] * len(vals), vals, color="k", s=10, zorder=3, alpha=0.6)
        ax.bar(range(len(variants)), means, yerr=stds, capsize=3,
               color="tab:blue", alpha=0.75)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([v.replace("psro_", "") for v in variants],
                           rotation=30, ha="right", fontsize=8)
        ax.set_title(game_name)
        ax.set_ylabel("final full_exploitability")
    fig.suptitle("The shootout: equal budget, honest metric")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"shootout.{ext}", dpi=200)
    plt.close(fig)


def _print_summary(summary: dict, games: list[str]) -> None:
    print(f"\n{'variant':>16} " + " ".join(f"{g:>18}" for g in games))
    for variant in VARIANTS:
        cells = []
        for game_name in games:
            entry = summary.get((game_name, variant))
            cells.append(f"{entry[0]:.3f} ± {entry[1]:.3f}" if entry else "—")
        print(f"{variant:>16} " + " ".join(f"{c:>18}" for c in cells))


if __name__ == "__main__":
    main()
