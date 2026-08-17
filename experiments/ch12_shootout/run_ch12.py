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
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from psrolab.utils.plotstyle import apply_style

HERE = Path(__file__).resolve().parent

VARIANTS = ["psro_nash", "psro_uniform", "psro_rm", "psro_alpharank",
            "psro_diverse", "last_k", "self_play"]


def _budget_tag(iterations: int, oracle_episodes: int) -> str:
    """Short human-readable tag for a (iterations, episodes) budget point."""
    def compact(n: int) -> str:
        return f"{n // 1000}k" if n >= 1000 and n % 1000 == 0 else str(n)
    return f"{iterations}x{compact(oracle_episodes)}"


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
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel worker processes across (game, variant, seed) "
                             "tuples; 1 = serial. Each worker gets torch.set_num_threads(1).")
    parser.add_argument("--budget-tag", type=str, default=None,
                        help="human-readable tag for this budget point (e.g., '10x15k'). "
                             "Auto-derived from --iterations and --oracle-episodes when "
                             "omitted. Rows with different tags coexist in shootout.csv.")
    parser.add_argument("--smoke", action="store_true", help="tiny config for CI (<60s)")
    parser.add_argument("--figdir", type=str, default=None,
                        help="override figures directory (ignored when --smoke is set, "
                             "so smoke never writes to tracked paths)")
    parser.add_argument("--plot-only", action="store_true",
                        help="skip PSRO/training; regenerate figures from the committed CSVs")
    args = parser.parse_args()
    if args.smoke:
        args.n_seeds, args.iterations = 1, 2
        args.oracle_episodes, args.eval_episodes = 500, 50
        args.variants, args.games = "psro_nash,self_play", "cyclic"
    variants = args.variants.split(",")
    games = args.games.split(",")
    plt.switch_backend("Agg")
    apply_style()

    per_run_min = {"leduc": args.iterations * args.oracle_episodes * 2 * 1.6e-3 / 60 + 2,
                   "cyclic": args.iterations * args.oracle_episodes * 2 * 0.4e-3 / 60 + 1}
    est_h = sum(per_run_min.get(g, 5) for g in games) * len(variants) * args.n_seeds / 60
    print(f"Estimated runtime: ~{est_h:.1f} h "
          f"({len(variants)} variants x {args.n_seeds} seeds x {games})", flush=True)

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

    budget_tag = args.budget_tag or _budget_tag(args.iterations, args.oracle_episodes)
    args.budget_tag = budget_tag  # stamped onto every row emitted by this call

    csv_path = results_dir / "shootout.csv"
    existing_rows = _load_existing(csv_path)
    done = {(r["game"], r["variant"], int(r["seed"]), r.get("budget_tag", ""))
            for r in existing_rows}
    tasks = [(g, v, args.seed + s)
             for g in games for v in variants for s in range(args.n_seeds)
             if (g, v, args.seed + s, budget_tag) not in done]
    if not tasks:
        print(f"All {len(games) * len(variants) * args.n_seeds} runs for "
              f"budget {budget_tag} already in {csv_path}; nothing to do.")
        rows = existing_rows
    else:
        print(f"Dispatching {len(tasks)} runs on {args.workers} worker(s) "
              f"(budget tag: {budget_tag}). Progress written to {csv_path}.",
              flush=True)
        new_rows = _dispatch(tasks, args, csv_path, existing_rows)
        rows = existing_rows + new_rows

    summary = _summarize(rows)
    _write_summary(results_dir, summary)
    _write_latex_all(results_dir, summary)
    _fig(rows, games, figures_dir)
    _print_summary(summary, games)
    print(f"Wrote CSVs + LaTeX to {results_dir} and figures to {figures_dir}")


def _dispatch(tasks, args, csv_path, existing_rows):
    """Fan out (game, variant, seed) runs across worker processes.

    Rows returned by workers are appended incrementally to csv_path (main
    process only, no lock needed) so a killed run leaves a valid CSV of
    everything done so far — the next invocation resumes.
    """
    new_rows: list[dict] = []
    if args.workers <= 1:
        for game_name, variant, seed in tasks:
            row = _run(game_name, variant, seed, args)
            new_rows.append(row)
            _write_csv(csv_path, existing_rows + new_rows)
        return new_rows

    args_dict = vars(args)
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_worker_init
    ) as pool:
        futures = {
            pool.submit(_run_from_dict, game, variant, seed, args_dict):
                (game, variant, seed)
            for game, variant, seed in tasks
        }
        done_count = 0
        for future in as_completed(futures):
            row = future.result()
            new_rows.append(row)
            done_count += 1
            _write_csv(csv_path, existing_rows + new_rows)
            print(f"[{done_count}/{len(tasks)}] wrote row for "
                  f"{row['game']}/{row['variant']}/seed{row['seed']}",
                  flush=True)
    return new_rows


def _worker_init() -> None:
    """Pin each worker to a single BLAS / torch thread to avoid CPU
    oversubscription when many workers run PPO simultaneously."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _run_from_dict(game_name: str, variant: str, seed: int, args_dict: dict) -> dict:
    """Wrapper that reconstructs the argparse.Namespace inside the worker."""
    args = argparse.Namespace(**args_dict)
    return _run(game_name, variant, seed, args)


def _load_existing(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.setdefault("budget_tag", _budget_tag(
            int(r.get("budget_iterations", 0)),
            int(r.get("budget_train_episodes", 0)) // (2 * max(1, int(r.get("budget_iterations", 1)))),
        ))
    return rows


def _write_csv(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = ["game", "variant", "seed", "budget_tag",
               "final_full_exploitability", "budget_iterations",
               "budget_train_episodes", "budget_eval_episodes_per_cell",
               "wall_clock_s"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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
        "budget_tag": getattr(args, "budget_tag", None)
                       or _budget_tag(args.iterations, args.oracle_episodes),
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


def _summarize(rows: list[dict]) -> dict:
    """Group by (game, variant, budget_tag) → (mean, std, n_seeds)."""
    summary: dict = {}
    for row in rows:
        key = (row["game"], row["variant"], row.get("budget_tag", ""))
        summary.setdefault(key, []).append(float(row["final_full_exploitability"]))
    return {
        key: (float(np.mean(vals)), float(np.std(vals)), len(vals))
        for key, vals in summary.items()
    }


def _write_summary(results_dir: Path, summary: dict) -> None:
    with open(results_dir / "shootout_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["game", "variant", "budget_tag",
                         "mean_full_exploitability",
                         "std_full_exploitability", "n_seeds"])
        for (game_name, variant, tag), (mean, std, n) in sorted(summary.items()):
            writer.writerow([game_name, variant, tag, f"{mean:.6f}",
                             f"{std:.6f}", n])


def _write_latex_all(results_dir: Path, summary: dict) -> None:
    """Emit one LaTeX table per budget_tag present in the data.

    The canonical filename `shootout_table.tex` always tracks the smallest
    budget (the book's original headline); larger budgets emit
    `shootout_table_{budget_tag}.tex` alongside so §12.6's comparison
    across budget points has both files to draw on.
    """
    tags_seen: list[str] = []
    for (_, _, tag) in summary:
        if tag not in tags_seen:
            tags_seen.append(tag)
    if not tags_seen:
        return
    # smallest budget first (by iterations parsed from the tag prefix)
    def _tag_iters(tag: str) -> int:
        try:
            return int(tag.split("x", 1)[0])
        except ValueError:
            return 0
    tags_seen.sort(key=_tag_iters)
    for i, tag in enumerate(tags_seen):
        sub = {k: v for k, v in summary.items() if k[2] == tag}
        path = (results_dir / "shootout_table.tex" if i == 0
                else results_dir / f"shootout_table_{tag}.tex")
        _write_latex(path, sub, tag)


def _write_latex(path: Path, summary: dict, budget_tag: str) -> None:
    games: list[str] = []
    for (g, _, _) in summary:
        if g not in games:
            games.append(g)
    lines = [
        "% Auto-generated by experiments/ch12_shootout/run_ch12.py — do not edit.",
        f"% Budget tag: {budget_tag}.",
        r"\begin{tabular}{l" + "c" * len(games) + "}",
        r"\toprule",
        "variant & " + " & ".join(games) + r" \\",
        r"\midrule",
    ]
    best = {g: min(mean for (gn, _, _), (mean, _, _) in summary.items()
                   if gn == g)
            for g in games}
    for variant in VARIANTS:
        cells = []
        for game_name in games:
            entry = summary.get((game_name, variant, budget_tag))
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


def _plot_from_csv(results_dir: Path, figures_dir: Path) -> None:
    rows = _load_existing(results_dir / "shootout.csv")
    for r in rows:
        r["seed"] = int(r["seed"])
    games = []
    for r in rows:
        if r["game"] not in games:
            games.append(r["game"])
    _fig(rows, games, figures_dir)


def _fig(rows: list[dict], games: list[str], figures_dir: Path) -> None:
    tags: list[str] = []
    for r in rows:
        t = r.get("budget_tag", "")
        if t not in tags:
            tags.append(t)
    fig, axes = plt.subplots(1, len(games), figsize=(5.5 * len(games), 4.2),
                             squeeze=False)
    palette = plt.get_cmap("tab10").colors
    bar_w = 0.8 / max(1, len(tags))
    for ax, game_name in zip(axes[0], games):
        sub = [r for r in rows if r["game"] == game_name]
        variants = [v for v in VARIANTS if any(r["variant"] == v for r in sub)]
        x = np.arange(len(variants))
        for ti, tag in enumerate(tags):
            means, stds = [], []
            for i, variant in enumerate(variants):
                vals = [float(r["final_full_exploitability"]) for r in sub
                        if r["variant"] == variant
                        and r.get("budget_tag", "") == tag]
                if not vals:
                    means.append(np.nan); stds.append(0.0); continue
                means.append(np.mean(vals)); stds.append(np.std(vals))
                offset = (ti - (len(tags) - 1) / 2) * bar_w
                ax.scatter(np.full(len(vals), i + offset), vals,
                           color="k", s=8, zorder=3, alpha=0.55)
            offset = (ti - (len(tags) - 1) / 2) * bar_w
            ax.bar(x + offset, means, width=bar_w * 0.95, yerr=stds,
                   capsize=2, color=palette[ti % 10], alpha=0.8,
                   label=tag if len(tags) > 1 else None)
        ax.set_xticks(x)
        ax.set_xticklabels([v.replace("psro_", "") for v in variants],
                           rotation=30, ha="right", fontsize=8)
        ax.set_title(game_name)
        ax.set_ylabel("final full_exploitability")
        if len(tags) > 1:
            ax.legend(title="budget", fontsize=8)
    fig.suptitle("The shootout: equal budget, honest metric")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"shootout.{ext}", dpi=200)
    plt.close(fig)


def _print_summary(summary: dict, games: list[str]) -> None:
    tags: list[str] = []
    for (_, _, tag) in summary:
        if tag not in tags:
            tags.append(tag)
    for tag in tags:
        print(f"\n[budget {tag}]  {'variant':>16} "
              + " ".join(f"{g:>18}" for g in games))
        for variant in VARIANTS:
            cells = []
            for game_name in games:
                entry = summary.get((game_name, variant, tag))
                cells.append(f"{entry[0]:.3f} ± {entry[1]:.3f}" if entry else "—")
            print(f"{'':>10}{variant:>16} " + " ".join(f"{c:>18}" for c in cells))


if __name__ == "__main__":
    main()
