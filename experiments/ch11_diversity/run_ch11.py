"""Ch. 11 — When does behavioral diversity pay?

PSRO with the DiversePPOOracle, diversity bonus on (lambda=1) vs off
(lambda=0), on two hand-built matrix games wrapped as turn-based OpenSpiel
games:

  transitive:  A[i,j] = (i - j) / n   — a pure skill ladder; strategy n-1
               dominates and there is nothing to hedge against.
  cyclic:      generalized rock-paper-scissors on n strategies, where i beats
               the next floor(n/2) strategies above it (mod n) — maximally
               non-transitive, uniform Nash.

The book's claim, tested honestly: diversity bonuses only earn their compute
on the cyclic game, where covering the strategy cycle matters; on the
transitive game they can only distract the oracle from climbing the ladder.

Exploitability here is exact and named `full_exploitability`: the underlying
matrix is known, and each neural policy reduces to a mixture over its rows via
action_probabilities at the (single) decision state.

Outputs:
    results/diversity_curves.csv
    figures/diversity_curves.{pdf,png}

Requires `pip install -e ".[rl]"`. CPU-friendly (~half an hour full config).
Determinism: numpy + torch seeded; book figures use CPU runs.
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
from psrolab.games import MatrixGame, Population
from psrolab.games.openspiel_wrap import OpenSpielGame
from psrolab.meta_solvers import ZeroSumProjectionNash
from psrolab.oracles.diverse_ppo import DiversePPOOracle

HERE = Path(__file__).resolve().parent


def transitive_matrix(n: int) -> np.ndarray:
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return (i - j) / n


def cyclic_matrix(n: int) -> np.ndarray:
    """Generalized RPS: i beats the floor(n/2) strategies cyclically above it."""
    assert n % 2 == 1, "odd n keeps the cycle balanced (uniform Nash)"
    a = np.zeros((n, n))
    for offset in range(1, n // 2 + 1):
        for i in range(n):
            a[(i + offset) % n, i] = 1.0
            a[i, (i + offset) % n] = -1.0
    return a


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-strategies", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--oracle-episodes", type=int, default=15000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--diversity-coef", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--smoke", action="store_true", help="tiny config for CI (<60s)")
    args = parser.parse_args()
    if args.smoke:
        args.iterations, args.oracle_episodes, args.eval_episodes = 2, 800, 50
        args.n_seeds, args.n_strategies = 1, 5
    plt.switch_backend("Agg")

    est_min = 2 * 2 * args.n_seeds * args.iterations * args.oracle_episodes * 1e-3 / 60
    print(f"Estimated runtime: ~{max(est_min, 0.3):.0f} min", flush=True)

    suffix = "_smoke" if args.smoke else ""
    results_dir = HERE / f"results{suffix}"
    figures_dir = HERE / f"figures{suffix}"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    matrices = {
        "transitive": transitive_matrix(args.n_strategies),
        "cyclic": cyclic_matrix(args.n_strategies),
    }
    rows = []
    for game_name, a in matrices.items():
        for coef in (0.0, args.diversity_coef):
            for s in range(args.n_seeds):
                rows += _run(game_name, a, coef, args.seed + s, args)
    _write(results_dir / "diversity_curves.csv", rows)
    _fig(rows, figures_dir, args.diversity_coef)
    print(f"Wrote {results_dir / 'diversity_curves.csv'} and figures to {figures_dir}")


def _run(game_name: str, a: np.ndarray, coef: float, seed: int, args) -> list[dict]:
    import pyspiel

    matrix = MatrixGame(payoffs=np.stack([a, -a]))
    raw = pyspiel.convert_to_turn_based(
        pyspiel.create_matrix_game(a.tolist(), (-a).tolist())
    )
    game = OpenSpielGame(raw, seed=seed)
    t0 = time.perf_counter()
    result = run_psro(
        game=game,
        oracle=DiversePPOOracle(
            diversity_coef=coef, total_episodes=args.oracle_episodes,
            device=args.device,
        ),
        meta_solver=ZeroSumProjectionNash(),
        evaluator=ProfileEvaluator(n_episodes_per_profile=args.eval_episodes),
        n_iterations=args.iterations,
        seed=seed,
        callbacks=[
            lambda it, pop, meta, g=game, m=matrix: {
                "full_exploitability": _matrix_exploitability(g, m, pop, meta)
            }
        ],
    )
    runtime_s = time.perf_counter() - t0
    print(f"{game_name:>10} lambda={coef:>3} seed {seed}: final full_exploitability = "
          f"{result.history[-1]['full_exploitability']:.4f} ({runtime_s:.0f}s)",
          flush=True)
    return [
        {
            "game": game_name,
            "diversity_coef": coef,
            "seed": seed,
            "iteration": e["iteration"],
            "full_exploitability": f"{e['full_exploitability']:.6f}",
            "runtime_s": f"{runtime_s:.1f}",
        }
        for e in result.history
    ]


def _matrix_exploitability(
    game: OpenSpielGame, matrix: MatrixGame, population: Population,
    meta: list[np.ndarray],
) -> float:
    """Exact exploitability: reduce each neural policy to a row mixture."""
    n = matrix.n_strategies[0]
    state = game.raw_game.new_initial_state()
    observations = []
    legals = []
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
            probs = policy.action_probabilities(
                observations[player], legals[player], n
            ) if hasattr(policy, "action_probabilities") else None
            if probs is None:
                probs = np.zeros(n)
                probs[policy.act(observations[player], legals[player])] = 1.0
            mix += prob * probs
        mixtures.append(mix / mix.sum())
    return restricted_exploitability(matrix, mixtures)


def _write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fig(rows: list[dict], figures_dir: Path, coef: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, game_name in zip(axes, ["transitive", "cyclic"]):
        for arm, color in [(0.0, "tab:gray"), (coef, "tab:green")]:
            sub = [r for r in rows if r["game"] == game_name
                   and float(r["diversity_coef"]) == arm]
            seeds = sorted({r["seed"] for r in sub})
            curves = np.array(
                [[float(r["full_exploitability"]) for r in sub if r["seed"] == s]
                 for s in seeds]
            )
            it = np.arange(1, curves.shape[1] + 1)
            label = "diversity off" if arm == 0.0 else f"diversity on (λ={arm})"
            ax.semilogy(it, curves.mean(axis=0), color=color, label=label)
            if len(seeds) > 1:
                ax.fill_between(it, curves.min(axis=0), curves.max(axis=0),
                                color=color, alpha=0.15, lw=0)
        ax.set_title(f"{game_name} game")
        ax.set_xlabel("PSRO iteration")
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("full_exploitability (exact, matrix)")
    fig.suptitle("Diversity bonus on vs off: transitive and cyclic games")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"diversity_curves.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
