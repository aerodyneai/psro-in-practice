"""Ch. 12 warm-start ablation (Fig 12.1).

Compares from-scratch PPO best responses (``warm_start=False``) against
warm-started ones (``warm_start=True``, loads the previous BR's state
dict for the same player) on Leduc + cyclic under an identical episode
budget. Two panels:

  * **final full-exploitability** — outcome.
  * **population span** — mean pairwise TV distance between population
    members at sampled observations (mechanism the manuscript's §12.4
    warns about: warm starts risk near-duplicate populations).

Reuses ``psrolab.oracles.diverse_ppo`` helpers for the TV distance
computation so the span metric here matches the diversity bonus's
formulation.

Outputs:
    results/warm_start.csv           per-run summary rows
    figures/warm_start_ablation.{pdf,png}

Requires ``.[rl]``. Compute cost ~= two ch12 shootout arms at whatever
budget you pass, run 5 seeds x 2 games x 2 conditions = 20 runs.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from psrolab import run_psro
from psrolab.eval import ProfileEvaluator, restricted_exploitability
from psrolab.eval.openspiel_exploitability import full_exploitability
from psrolab.games import MatrixGame, Population
from psrolab.games.openspiel_wrap import OpenSpielGame
from psrolab.meta_solvers import ZeroSumProjectionNash
from psrolab.oracles.diverse_ppo import _action_probs, _total_variation
from psrolab.oracles.ppo import PPOOracle
from psrolab.utils.parallel import fan_out
from psrolab.utils.plotstyle import apply_style

HERE = Path(__file__).resolve().parent


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
    parser.add_argument("--games", type=str, default="cyclic,leduc")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--oracle-episodes", type=int, default=15000)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--span-samples", type=int, default=50,
                        help="observations to sample per player for the span metric")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel worker processes across "
                             "(game, warm_start, seed) tuples; 1 = serial")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--figdir", type=str, default=None)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.n_seeds, args.iterations = 1, 2
        args.oracle_episodes, args.eval_episodes = 500, 50
        args.games = "cyclic"
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
        with open(results_dir / "warm_start.csv") as f:
            rows = list(csv.DictReader(f))
        _fig(rows, figures_dir)
        print(f"Wrote figures to {figures_dir}")
        return

    games = args.games.split(",")
    csv_path = results_dir / "warm_start.csv"
    existing = _load_existing(csv_path)
    done = {(r["game"], int(r["warm_start"]), int(r["seed"]))
            for r in existing}
    tasks = [(game_name, int(warm), args.seed + s)
             for game_name in games
             for warm in (False, True)
             for s in range(args.n_seeds)
             if (game_name, int(warm), args.seed + s) not in done]

    if not tasks:
        print(f"All runs already in {csv_path}; nothing to do.")
        rows = existing
    else:
        print(f"Dispatching {len(tasks)} runs on {args.workers} worker(s). "
              f"Progress written to {csv_path}.", flush=True)
        new_rows: list[dict] = []
        args_dict = vars(args)
        def _on_result(row, done_count, total):
            new_rows.append(row)
            _write(csv_path, existing + new_rows)
            tag = "warm" if int(row["warm_start"]) else "scratch"
            print(f"[{done_count}/{total}] {row['game']}/{tag}/seed{row['seed']}: "
                  f"e_G={row['final_full_exploitability']}, "
                  f"span={row['population_span_tv']}", flush=True)
        fan_out(tasks, _run_worker, args_dict, args.workers, _on_result)
        rows = existing + new_rows

    _fig(rows, figures_dir)
    print(f"Wrote {csv_path} and figures to {figures_dir}")


def _load_existing(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _run_worker(game_name: str, warm_start: int, seed: int,
                args_dict: dict) -> dict:
    return _run(game_name, bool(warm_start), seed,
                argparse.Namespace(**args_dict))


def _run(game_name: str, warm_start: bool, seed: int, args) -> dict:
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
    hidden = 128 if game_name == "leduc" else 64
    oracle = PPOOracle(
        total_episodes=args.oracle_episodes, hidden=hidden, device="cpu",
        warm_start=warm_start,
    )
    t0 = time.perf_counter()
    result = run_psro(
        game=game, oracle=oracle,
        meta_solver=ZeroSumProjectionNash(),
        evaluator=ProfileEvaluator(n_episodes_per_profile=args.eval_episodes),
        n_iterations=args.iterations, seed=seed,
    )
    runtime_s = time.perf_counter() - t0
    if matrix is None:
        expl = full_exploitability(game, result.population, result.meta_strategies)
    else:
        expl = _matrix_exploit(game, matrix, result.population, result.meta_strategies)
    span = _population_span(game, result.population, n_samples=args.span_samples,
                            rng=np.random.default_rng(seed + 999))
    tag = "warm" if warm_start else "scratch"
    print(f"{game_name:>6} {tag:>7} seed {seed}: e_G={expl:.4f}, "
          f"span={span:.3f}, wall={runtime_s:.0f}s", flush=True)
    return {
        "game": game_name, "warm_start": int(warm_start), "seed": seed,
        "final_full_exploitability": f"{expl:.6f}",
        "population_span_tv": f"{span:.6f}",
        "budget_iterations": args.iterations,
        "budget_train_episodes": 2 * args.iterations * args.oracle_episodes,
        "wall_clock_s": f"{runtime_s:.1f}",
    }


def _matrix_exploit(game, matrix, population, meta):
    n = matrix.n_strategies[0]
    state = game.raw_game.new_initial_state()
    obs, legals = [], []
    for player in (0, 1):
        obs.append(game.observation(state, player))
        legals.append(list(state.legal_actions()))
        if player == 0:
            state = state.child(state.legal_actions()[0])
    mixtures = []
    for player in (0, 1):
        mix = np.zeros(n)
        for policy, prob in zip(population.policies[player], meta[player]):
            if prob <= 1e-12:
                continue
            mix += prob * policy.action_probabilities(obs[player], legals[player], n)
        mixtures.append(mix / mix.sum())
    return restricted_exploitability(matrix, mixtures)


def _population_span(game: OpenSpielGame, population: Population,
                     n_samples: int, rng: np.random.Generator) -> float:
    """Mean pairwise total-variation distance between population policies
    at a set of sampled observations (per player, averaged over players)."""
    raw = game.raw_game
    n_actions = raw.num_distinct_actions()
    total = 0.0; count = 0
    for player in range(2):
        pop = population.policies[player]
        if len(pop) < 2:
            continue
        obs_batch, legal_batch = _sample_observations(
            raw, game, player, n_samples, rng,
        )
        for obs, legal in zip(obs_batch, legal_batch):
            probs = [_action_probs(p, obs, legal, n_actions) for p in pop]
            pair_tvs = []
            for i in range(len(probs)):
                for j in range(i + 1, len(probs)):
                    pair_tvs.append(_total_variation(probs[i], probs[j]))
            if pair_tvs:
                total += float(np.mean(pair_tvs)); count += 1
    return total / count if count else 0.0


def _sample_observations(raw, game, player: int, n_samples: int,
                         rng: np.random.Generator):
    obs_out, legal_out = [], []
    for _ in range(n_samples):
        state = raw.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                actions_probs = state.chance_outcomes()
                choice_idx = int(rng.choice(
                    len(actions_probs), p=[p for _, p in actions_probs],
                ))
                state.apply_action(actions_probs[choice_idx][0])
                continue
            if state.current_player() == player:
                obs_out.append(game.observation(state, player))
                legal_out.append(list(state.legal_actions()))
                # random continuation
                state.apply_action(int(rng.choice(state.legal_actions())))
                break
            else:
                state.apply_action(int(rng.choice(state.legal_actions())))
    return obs_out, legal_out


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def _fig(rows: list[dict], figures_dir: Path) -> None:
    games: list[str] = []
    for r in rows:
        if r["game"] not in games:
            games.append(r["game"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, ylabel in zip(
        axes,
        ["final_full_exploitability", "population_span_tv"],
        ["final full_exploitability", "mean pairwise TV span"],
    ):
        x = np.arange(len(games))
        for k, tag, color in ((0, "scratch", "tab:gray"), (1, "warm", "tab:red")):
            means, stds = [], []
            for g in games:
                vals = [float(r[metric]) for r in rows
                        if r["game"] == g and int(r["warm_start"]) == k]
                means.append(np.mean(vals) if vals else np.nan)
                stds.append(np.std(vals) if vals else 0.0)
                if vals:
                    offset = -0.15 if k == 0 else 0.15
                    ax.scatter(np.full(len(vals), games.index(g) + offset),
                               vals, color=color, s=12, alpha=0.7, zorder=3)
            offset = -0.2 if k == 0 else 0.2
            ax.bar(x + offset, means, width=0.35, yerr=stds, capsize=2,
                   color=color, alpha=0.75, label=tag)
        ax.set_xticks(x); ax.set_xticklabels(games)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
    fig.suptitle("Warm-started PPO best responses vs from-scratch (Ch. 12 §12.4)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"warm_start_ablation.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
