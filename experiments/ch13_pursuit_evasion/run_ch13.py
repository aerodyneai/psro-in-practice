"""Ch. 13 — Full PSRO (Nash + PPO) on the pursuit-evasion capstone.

The whole book in one run: a custom simultaneous-move game, the unchanged PSRO
loop, the Ch. 6 PPO oracle, the Nash meta-solver, and honest measurement on a
game with NO exact exploitability calculator. Three figures:

  1. trajectories.{pdf,png}   top-down episode traces at iterations 1 / 5 / 15
                              — pursuit behavior visibly sharpens.
  2. meta_support.{pdf,png}   Nash meta-strategy weights over the population
                              per iteration — support grows and shifts.
  3. exploitability_proxy.{pdf,png}
                              population-tournament proxy: how exploitable is
                              iteration k's meta-strategy against the FINAL
                              population, plus a fresh PPO best-response probe
                              against the final meta (approximate lower bound
                              on true exploitability).

There is no `full_exploitability` column here and the CSV says so: columns are
`restricted_exploitability_vs_final_population` (exact w.r.t. the population,
blind outside it) and `br_probe_gain` (sampled, approximate). Ch. 13's prose
is precisely about living with that.

Outputs:
    results/psro_metrics.csv, results/br_probe.csv, figures/ (above)

Requires `pip install -e ".[rl]"` (torch; no OpenSpiel needed). Default config
~1.5 h on the reference server (CPU); --smoke <60s.
Determinism: numpy + torch seeded; CPU.
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
from psrolab.games import MatrixGame
from psrolab.games.pursuit_evasion import PursuitEvasionSim
from psrolab.meta_solvers import ZeroSumProjectionNash
from psrolab.oracles.base import MixtureOpponent
from psrolab.oracles.ppo import PPOOracle

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--oracle-episodes", type=int, default=20000)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--probe-episodes", type=int, default=20000,
                        help="training episodes for the final BR probe")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--snapshot-iterations", type=str, default="1,5,15",
                        help="iterations rendered in the trajectory figure")
    parser.add_argument("--torch-threads", type=int, default=0,
                        help="torch CPU threads; 0 = torch default (measured "
                             "2.4x faster than pinning to 1 here — see NOTES.md)")
    parser.add_argument("--smoke", action="store_true", help="tiny config for CI (<60s)")
    args = parser.parse_args()
    if args.smoke:
        args.iterations, args.oracle_episodes, args.eval_episodes = 2, 300, 20
        args.probe_episodes, args.hidden, args.snapshot_iterations = 300, 32, "1,2"
    snapshots = [int(s) for s in args.snapshot_iterations.split(",")]
    plt.switch_backend("Agg")

    if args.torch_threads > 0:
        import torch

        torch.set_num_threads(args.torch_threads)

    # ~9ms per training episode at torch's default threading (measured; the
    # single-thread-pinned run cost 20ms/episode — see NOTES.md).
    est_min = ((2 * args.iterations * args.oracle_episodes + 2 * args.probe_episodes)
               * 9e-3 + args.iterations**2 * args.eval_episodes * 4e-3) / 60
    print(f"Estimated runtime: ~{max(est_min, 0.3):.0f} min", flush=True)

    suffix = "_smoke" if args.smoke else ""
    results_dir = HERE / f"results{suffix}"
    figures_dir = HERE / f"figures{suffix}"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    game = PursuitEvasionSim(seed=args.seed)
    metas: list[list[np.ndarray]] = []

    def record_meta(it, pop, meta):
        metas.append([m.copy() for m in meta])
        supports = (int((meta[0] > 1e-3).sum()), int((meta[1] > 1e-3).sum()))
        print(f"iteration {it:>2}: population {pop.sizes()}, Nash supports "
              f"{supports}, elapsed {time.perf_counter() - t0:.0f}s", flush=True)
        return {"support_p0": supports[0], "support_p1": supports[1]}

    t0 = time.perf_counter()
    result = run_psro(
        game=game,
        oracle=PPOOracle(total_episodes=args.oracle_episodes, hidden=args.hidden,
                         device=args.device),
        meta_solver=ZeroSumProjectionNash(),
        evaluator=ProfileEvaluator(n_episodes_per_profile=args.eval_episodes),
        n_iterations=args.iterations,
        seed=args.seed,
        callbacks=[record_meta],
    )
    print(f"PSRO done in {time.perf_counter() - t0:.0f}s; "
          f"population {result.population.sizes()}", flush=True)

    tournament = _population_tournament(result, metas)
    _write_metrics(results_dir / "psro_metrics.csv", result, tournament)
    probe = _br_probe(game, result, args)
    _write_probe(results_dir / "br_probe.csv", probe)

    _fig_trajectories(game, result, metas, snapshots, figures_dir, args.seed)
    _fig_meta_support(metas, figures_dir)
    _fig_proxy(tournament, probe, figures_dir)
    print(f"Wrote CSVs to {results_dir} and figures to {figures_dir}")


def _pad_meta(meta: list[np.ndarray], sizes: list[int]) -> list[np.ndarray]:
    return [np.pad(m, (0, s - len(m))) for m, s in zip(meta, sizes)]


def _population_tournament(result, metas) -> list[float]:
    """Exploitability of each iteration's meta vs the FINAL population.

    Exact w.r.t. the final empirical game; says nothing about strategies
    outside the population (hence the honest column name)."""
    final_game: MatrixGame = result.population.as_matrix_game()
    sizes = final_game.n_strategies
    return [
        restricted_exploitability(final_game, _pad_meta(meta, sizes))
        for meta in metas
    ]


def _br_probe(game, result, args) -> dict:
    """Train a fresh PPO BR against the final meta; its mean return against the
    meta is a sampled lower bound on how exploitable the final answer is."""
    rng = np.random.default_rng(args.seed + 999)
    gains = {}
    for player in (0, 1):
        oracle = PPOOracle(total_episodes=args.probe_episodes, hidden=args.hidden,
                           device=args.device)
        probe_policy = oracle.best_response(
            game, player, result.population, result.meta_strategies, rng
        )
        opponents = MixtureOpponent(result.population, result.meta_strategies,
                                    exclude_player=player)
        total = 0.0
        n_eval = 2000 if args.probe_episodes > 1000 else 100
        for _ in range(n_eval):
            sampled = opponents.sample(rng)
            joint = [probe_policy if p == player else sampled[p] for p in (0, 1)]
            total += game.play_episode(joint, rng)[player]
        gains[player] = total / n_eval
    return {"pursuer_probe_gain": gains[0], "evader_probe_gain": gains[1],
            "mean_probe_gain": 0.5 * (gains[0] + gains[1]),
            "probe_episodes": args.probe_episodes}


def _write_metrics(path: Path, result, tournament: list[float]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "population_size",
                         "restricted_exploitability_vs_final_population",
                         "support_p0", "support_p1"])
        for entry, tour in zip(result.history, tournament):
            writer.writerow([entry["iteration"], entry["population_sizes"][0],
                             f"{tour:.6f}", entry["support_p0"], entry["support_p1"]])


def _write_probe(path: Path, probe: dict) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(probe))
        writer.writeheader()
        writer.writerow(probe)


def _sample_joint_policies(result, meta, rng):
    idx = [int(rng.choice(len(meta[p]), p=meta[p])) for p in (0, 1)]
    return [result.population.policies[p][idx[p]] for p in (0, 1)]


def _fig_trajectories(game, result, metas, snapshots, figures_dir, seed) -> None:
    fig, axes = plt.subplots(1, len(snapshots), figsize=(4.2 * len(snapshots), 4.4))
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(seed + 77)
    for ax, it in zip(axes, snapshots):
        meta = metas[min(it - 1, len(metas) - 1)]
        for _ in range(6):
            joint = _sample_joint_policies(result, meta, rng)
            _, trajectory = game.play_episode(joint, rng, record=True)
            captured = len(trajectory) - 1 < game.max_steps
            ax.plot(trajectory[:, 0, 0], trajectory[:, 0, 1], color="tab:red",
                    lw=1.2, alpha=0.7)
            ax.plot(trajectory[:, 1, 0], trajectory[:, 1, 1], color="tab:blue",
                    lw=1.2, alpha=0.7)
            marker = "x" if captured else "o"
            ax.scatter(*trajectory[-1, 1], color="tab:blue", marker=marker, s=40,
                       zorder=3)
            ax.scatter(*trajectory[0, 0], color="tab:red", marker="s", s=25, zorder=3)
        ax.set_title(f"iteration {it}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].plot([], [], color="tab:red", label="pursuer")
    axes[0].plot([], [], color="tab:blue", label="evader (x = caught)")
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("Pursuit-evasion under the iteration-k meta-strategy (6 episodes each)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"trajectories.{ext}", dpi=200)
    plt.close(fig)


def _fig_meta_support(metas, figures_dir) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    n_iter = len(metas)
    n_pols = len(metas[-1][0])
    for player, ax in enumerate(axes):
        weights = np.zeros((n_iter, n_pols))
        for k, meta in enumerate(metas):
            weights[k, : len(meta[player])] = meta[player]
        image = ax.imshow(weights.T, aspect="auto", origin="lower", cmap="viridis",
                          extent=(1, n_iter, -0.5, n_pols - 0.5))
        ax.set_xlabel("PSRO iteration")
        ax.set_ylabel("policy index")
        ax.set_title(["pursuer", "evader"][player])
        fig.colorbar(image, ax=ax, label="Nash weight", shrink=0.85)
    fig.suptitle("Meta-strategy support over time")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"meta_support.{ext}", dpi=200)
    plt.close(fig)


def _fig_proxy(tournament, probe, figures_dir) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    iterations = np.arange(1, len(tournament) + 1)
    ax.plot(iterations, tournament, marker="o", ms=4,
            label="restricted exploitability vs final population")
    ax.axhline(probe["mean_probe_gain"], color="tab:red", ls="--", lw=1,
               label=f"final BR-probe gain ({probe['mean_probe_gain']:.3f})")
    ax.set_xlabel("PSRO iteration")
    ax.set_ylabel("exploitability proxy")
    ax.set_title("Population tournament + best-response probe")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"exploitability_proxy.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
