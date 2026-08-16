"""Ch. 4 — PSRO as double oracle on random zero-sum matrix games.

Runs `run_psro` with the exact matrix oracle + Nash LP meta-solver (== the
double-oracle algorithm) on random zero-sum games of growing size. Three
studies, three book figures:

  A. Population size at convergence vs game size — how many strategies does DO
     actually need before its restricted Nash is a full-game Nash?
  B. Exploitability vs iteration — `full_exploitability` (true game) falls to
     zero while `restricted_exploitability` (over the population) is pinned at
     ~0 from iteration 1. The book's honest-measurement lesson in one plot.
  C. Payoff noise (`MatrixGameSim(noise_std=...)`) — noisy payoff estimates put
     a floor under full exploitability, motivating episode-count choices for
     the RL chapters.

Outputs:
    results/convergence_vs_size.csv        one row per (size, seed)
    results/exploitability_vs_iteration.csv  per-iteration traces (seed 0)
    results/noise_effect.csv               per-iteration traces per noise level
    figures/do_population_vs_size.{pdf,png}
    figures/do_exploitability.{pdf,png}
    figures/do_noise_floor.{pdf,png}

Determinism: pure numpy; bit-reproducible given --seed. `run_psro` has no early
stopping (the loop is sacred), so convergence runs use doubling restarts: run
with an iteration cap, and if the full-game exploitability never hit the
tolerance, double the cap and rerun from scratch (same seed → same trajectory,
bounded geometric overhead).
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
from psrolab.games import ExactMatrixOracle, MatrixGame, MatrixGameSim, Population
from psrolab.meta_solvers import ZeroSumProjectionNash
from psrolab.utils.plotstyle import apply_style

HERE = Path(__file__).resolve().parent
CONVERGENCE_TOL = 1e-6
LOG_FLOOR = 1e-12  # for plotting exact zeros on log axes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="base seed; run s uses seed+s")
    parser.add_argument("--sizes", type=str, default="10,25,50,100,250,500")
    parser.add_argument("--n-seeds", type=int, default=5, help="random games per size")
    parser.add_argument("--noise-levels", type=str, default="0,0.5,2.0")
    parser.add_argument("--noise-size", type=int, default=50, help="game size for study C")
    parser.add_argument("--noise-iterations", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=100,
                        help="episodes per profile in the noise study (C)")
    parser.add_argument("--smoke", action="store_true", help="tiny config for CI (<60s)")
    parser.add_argument("--figdir", type=str, default=None,
                        help="override figures directory (ignored when --smoke is set, "
                             "so smoke never writes to tracked paths)")
    parser.add_argument("--plot-only", action="store_true",
                        help="skip PSRO/training; regenerate figures from the committed CSVs")
    args = parser.parse_args()
    if args.smoke:
        args.sizes, args.n_seeds = "10,20", 2
        args.noise_levels, args.noise_size, args.noise_iterations = "0,1.0", 10, 8
    sizes = [int(s) for s in args.sizes.split(",")]
    noise_levels = [float(s) for s in args.noise_levels.split(",")]

    # Cost per run is dominated by the Nash LP over growing populations,
    # roughly cubic in the iteration cap; the constant is calibrated on the
    # reference laptop (full default config ≈ 25 min). A missed convergence cap
    # doubles that size's cost (see _run_until_converged).
    est_s = sum(args.n_seeds * (0.75 * n + 6) ** 3 * 4.7e-6 for n in sizes) + 10
    print(f"Estimated runtime: ~{max(est_s / 60, 0.1):.0f} min "
          f"(sizes={sizes}, {args.n_seeds} seeds each; up to 2x if caps are missed)",
          flush=True)
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
        _plot_from_csv(results_dir, figures_dir)
        print(f"Wrote figures to {figures_dir}")
        return

    conv_rows, trace_rows = _study_convergence(sizes, args)
    _write_rows(results_dir / "convergence_vs_size.csv", conv_rows)
    _write_rows(results_dir / "exploitability_vs_iteration.csv", trace_rows)
    noise_rows = _study_noise(noise_levels, args)
    _write_rows(results_dir / "noise_effect.csv", noise_rows)

    _fig_population_vs_size(conv_rows, figures_dir)
    _fig_exploitability(trace_rows, figures_dir)
    _fig_noise_floor(noise_rows, figures_dir)
    print(f"Wrote CSVs to {results_dir} and figures to {figures_dir}")


def make_random_zero_sum(n: int, seed: int) -> MatrixGame:
    """Random zero-sum game with iid standard-normal payoffs to player 0."""
    a = np.random.default_rng(seed).standard_normal((n, n))
    return MatrixGame(payoffs=np.stack([a, -a]))


def expand_to_full(
    population: Population, meta_strategies: list[np.ndarray], n_strategies: int
) -> list[np.ndarray]:
    """Lift meta-strategies over the population to mixtures over the full game.

    Population entries are PureStrategy policies; probability mass lands on
    their underlying strategy index (duplicates accumulate).
    """
    mixtures = []
    for player in range(2):
        mix = np.zeros(n_strategies)
        for policy, prob in zip(population.policies[player], meta_strategies[player]):
            mix[policy.index] += prob
        mixtures.append(mix)
    return mixtures


def exploitability_callback(game: MatrixGame):
    """Per-iteration metrics: full (true game) and restricted (population) kind."""

    def callback(it: int, population: Population, meta: list[np.ndarray]) -> dict:
        full_mix = expand_to_full(population, meta, game.n_strategies[0])
        return {
            "full_exploitability": restricted_exploitability(game, full_mix),
            "restricted_exploitability": restricted_exploitability(
                population.as_matrix_game(), meta
            ),
        }

    return callback


def run_double_oracle(
    game: MatrixGame, seed: int, n_iterations: int, noise_std: float = 0.0, episodes: int = 1
):
    """One `run_psro` call wired up as double oracle on `game`."""
    return run_psro(
        game=MatrixGameSim(game, noise_std=noise_std),
        oracle=ExactMatrixOracle(game),
        meta_solver=ZeroSumProjectionNash(),
        evaluator=ProfileEvaluator(n_episodes_per_profile=episodes),
        n_iterations=n_iterations,
        seed=seed,
        callbacks=[exploitability_callback(game)],
    )


def _study_convergence(sizes: list[int], args) -> tuple[list[dict], list[dict]]:
    conv_rows, trace_rows = [], []
    for n in sizes:
        for s in range(args.n_seeds):
            game = make_random_zero_sum(n, seed=1000 * n + args.seed + s)
            t0 = time.perf_counter()
            history, cap = _run_until_converged(game, seed=args.seed + s)
            runtime_s = time.perf_counter() - t0
            hit = next(
                (e for e in history if e["full_exploitability"] < CONVERGENCE_TOL), None
            )
            converged_iter = hit["iteration"] if hit else len(history) - 1
            if hit is None:
                print(f"  WARNING: size {n} seed {s} not converged at cap {cap}")
            conv_rows.append(
                {
                    "game_size": n,
                    "seed": args.seed + s,
                    "converged": int(hit is not None),
                    "converged_iteration": converged_iter,
                    "population_size": converged_iter + 1,
                    "full_exploitability": f"{history[converged_iter]['full_exploitability']:.3e}",
                    "runtime_s": f"{runtime_s:.3f}",
                }
            )
            if s == 0:
                for e in history[: converged_iter + 1]:
                    trace_rows.append(
                        {
                            "game_size": n,
                            "iteration": e["iteration"],
                            "population_size": e["iteration"] + 1,
                            "restricted_exploitability": f"{e['restricted_exploitability']:.3e}",
                            "full_exploitability": f"{e['full_exploitability']:.3e}",
                        }
                    )
        done = [r for r in conv_rows if r["game_size"] == n]
        mean_pop = np.mean([r["population_size"] for r in done])
        print(f"size {n:>4}: mean population at convergence = {mean_pop:.1f}", flush=True)
    return conv_rows, trace_rows


def _run_until_converged(game: MatrixGame, seed: int) -> tuple[list[dict], int]:
    """Doubling-restart driver (run_psro has no early stop — see module docstring)."""
    n = game.n_strategies[0]
    # Nash support of iid-Gaussian zero-sum games is empirically ~2n/3 (see
    # results/convergence_vs_size.csv), so 0.75n + 6 converges first try with
    # ~15% headroom; a miss costs one doubled rerun. See NOTES.md.
    cap = max(8, int(0.75 * n) + 6)
    max_cap = 2 * n + 4
    while True:
        result = run_double_oracle(game, seed=seed, n_iterations=cap)
        if any(e["full_exploitability"] < CONVERGENCE_TOL for e in result.history):
            return result.history, cap
        if cap >= max_cap:
            return result.history, cap
        cap = min(2 * cap, max_cap)


def _study_noise(noise_levels: list[float], args) -> list[dict]:
    rows = []
    n = args.noise_size
    game = make_random_zero_sum(n, seed=1000 * n + args.seed)
    for noise_std in noise_levels:
        result = run_double_oracle(
            game, seed=args.seed, n_iterations=args.noise_iterations,
            noise_std=noise_std, episodes=args.episodes,
        )
        for e in result.history:
            rows.append(
                {
                    "noise_std": noise_std,
                    "episodes_per_profile": args.episodes,
                    "iteration": e["iteration"],
                    "restricted_exploitability": f"{e['restricted_exploitability']:.3e}",
                    "full_exploitability": f"{e['full_exploitability']:.3e}",
                }
            )
        print(f"noise_std {noise_std}: final full_exploitability = "
              f"{result.history[-1]['full_exploitability']:.4f}")
    return rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fig_population_vs_size(conv_rows: list[dict], figures_dir: Path) -> None:
    sizes = sorted({r["game_size"] for r in conv_rows})
    means, stds = [], []
    for n in sizes:
        pops = [r["population_size"] for r in conv_rows if r["game_size"] == n]
        means.append(np.mean(pops))
        stds.append(np.std(pops))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=3, label="DO population")
    ax.plot(sizes, sizes, color="0.5", ls="--", lw=1, label="game size (worst case)")
    ax.set_xlabel("game size (strategies per player)")
    ax.set_ylabel("population size at convergence")
    ax.set_title("Double oracle converges well before enumerating the game")
    ax.legend(frameon=False)
    _save(fig, figures_dir / "do_population_vs_size")


def _fig_exploitability(trace_rows: list[dict], figures_dir: Path) -> None:
    sizes = sorted({r["game_size"] for r in trace_rows})
    fig, ax = plt.subplots(figsize=(6, 4))
    for n in sizes:
        rows = [r for r in trace_rows if r["game_size"] == n]
        it = [r["iteration"] + 1 for r in rows]
        full = [max(float(r["full_exploitability"]), LOG_FLOOR) for r in rows]
        ax.semilogy(it, full, label=f"n={n} (full)")
    rows = [r for r in trace_rows if r["game_size"] == sizes[-1]]
    restricted = [max(float(r["restricted_exploitability"]), LOG_FLOOR) for r in rows]
    ax.semilogy([r["iteration"] + 1 for r in rows], restricted, color="k", ls=":",
                label="restricted (any n): ~0 always")
    ax.set_xlabel("PSRO iteration")
    ax.set_ylabel("exploitability")
    ax.set_title("Full-game exploitability falls; restricted says 'done' immediately")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, figures_dir / "do_exploitability")


def _fig_noise_floor(noise_rows: list[dict], figures_dir: Path) -> None:
    levels = sorted({r["noise_std"] for r in noise_rows})
    fig, ax = plt.subplots(figsize=(6, 4))
    for noise_std in levels:
        rows = [r for r in noise_rows if r["noise_std"] == noise_std]
        it = [r["iteration"] + 1 for r in rows]
        full = [max(float(r["full_exploitability"]), LOG_FLOOR) for r in rows]
        eff = noise_std / np.sqrt(float(rows[0]["episodes_per_profile"]))
        ax.semilogy(it, full, marker=".", ms=4,
                    label=f"noise_std={noise_std} (payoff sem {eff:.2f})")
    ax.set_xlabel("PSRO iteration")
    ax.set_ylabel("full_exploitability")
    ax.set_title("Payoff-estimation noise puts a floor under exploitability")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, figures_dir / "do_noise_floor")


def _plot_from_csv(results_dir: Path, figures_dir: Path) -> None:
    def _read(path: Path) -> list[dict]:
        with open(path) as f:
            return list(csv.DictReader(f))

    conv_rows = _read(results_dir / "convergence_vs_size.csv")
    for r in conv_rows:
        r["game_size"] = int(r["game_size"])
        r["population_size"] = int(r["population_size"])
    trace_rows = _read(results_dir / "exploitability_vs_iteration.csv")
    for r in trace_rows:
        r["game_size"] = int(r["game_size"])
        r["iteration"] = int(r["iteration"])
    noise_rows = _read(results_dir / "noise_effect.csv")
    for r in noise_rows:
        r["noise_std"] = float(r["noise_std"])
        r["iteration"] = int(r["iteration"])
    _fig_population_vs_size(conv_rows, figures_dir)
    _fig_exploitability(trace_rows, figures_dir)
    _fig_noise_floor(noise_rows, figures_dir)


def _save(fig, stem: Path) -> None:
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
