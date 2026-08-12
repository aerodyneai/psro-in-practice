"""Ch. 3 — Fictitious play: oscillating play, converging averages.

Runs fictitious play on rock-paper-scissors and matching pennies. Three book
figures: (1) the time-averaged strategy converging to the uniform Nash,
(2) the average-strategy trajectory spiraling through the simplex while the
instantaneous best responses cycle forever, (3) exploitability of the averages
vs iteration on a log-log scale.

Outputs:
    results/fp_rps.csv, results/fp_matching_pennies.csv   per-iteration traces
    figures/fp_convergence.{pdf,png}
    figures/fp_trajectory.{pdf,png}
    figures/fp_exploitability.{pdf,png}

Exploitability columns are named `full_exploitability`: FP here operates on the
full matrix game directly (no population restriction), so the number is exact.

Fully deterministic (pure numpy, fixed tie-breaking); --seed is accepted for
interface uniformity but unused.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from psrolab.baselines import FictitiousPlay
from psrolab.eval import restricted_exploitability
from psrolab.games import MatrixGame

HERE = Path(__file__).resolve().parent
SIMPLEX_VERTICES = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
RPS_LABELS = ["Rock", "Paper", "Scissors"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="unused; FP is deterministic")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--smoke", action="store_true", help="tiny config for CI (<60s)")
    args = parser.parse_args()
    if args.smoke:
        args.iterations = 200
    plt.switch_backend("Agg")

    suffix = "_smoke" if args.smoke else ""
    results_dir = HERE / f"results{suffix}"
    figures_dir = HERE / f"figures{suffix}"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    rps = MatrixGame(
        payoffs=np.stack([(a := np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], float)), -a])
    )
    pennies = MatrixGame(payoffs=np.stack([(a := np.array([[1, -1], [-1, 1]], float)), -a]))

    runs = {}
    for name, game, labels in [
        ("rps", rps, RPS_LABELS),
        ("matching_pennies", pennies, ["Heads", "Tails"]),
    ]:
        result = FictitiousPlay(game).run(args.iterations)
        exploit = np.array(
            [
                restricted_exploitability(game, [result.averages[0][t], result.averages[1][t]])
                for t in range(args.iterations)
            ]
        )
        runs[name] = (result, exploit)
        _write_csv(results_dir / f"fp_{name}.csv", result, exploit, labels)
        print(
            f"FP on {name}: {args.iterations} iterations, "
            f"final full_exploitability = {exploit[-1]:.4f}"
        )

    _fig_convergence(runs["rps"][0], figures_dir)
    _fig_trajectory(runs["rps"][0], figures_dir)
    _fig_exploitability(runs, figures_dir)
    print(f"Wrote CSVs to {results_dir} and figures to {figures_dir}")


def _write_csv(path: Path, result, exploit: np.ndarray, labels: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["iteration"]
        for p in range(2):
            header += [f"p{p}_avg_{lab.lower()}" for lab in labels]
        header += ["p0_best_response", "p1_best_response", "full_exploitability"]
        writer.writerow(header)
        for t in range(len(exploit)):
            row = [t + 1]
            for p in range(2):
                row += [f"{v:.6f}" for v in result.averages[p][t]]
            row += [result.best_responses[t, 0], result.best_responses[t, 1]]
            row += [f"{exploit[t]:.6e}"]
            writer.writerow(row)


def _fig_convergence(result, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    t = np.arange(1, len(result.averages[0]) + 1)
    for i, label in enumerate(RPS_LABELS):
        ax.plot(t, result.averages[0][:, i], label=label)
    ax.axhline(1 / 3, color="0.4", ls="--", lw=1, label="Nash (1/3)")
    ax.set_xscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("time-averaged probability (player 0)")
    ax.set_title("FP on RPS: averages converge to the uniform Nash")
    ax.legend(frameon=False)
    _save(fig, figures_dir / "fp_convergence")


def _fig_trajectory(result, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    points = result.averages[0] @ SIMPLEX_VERTICES
    triangle = np.vstack([SIMPLEX_VERTICES, SIMPLEX_VERTICES[0]])
    ax.plot(triangle[:, 0], triangle[:, 1], color="0.3", lw=1)
    sc = ax.scatter(
        points[:, 0], points[:, 1], c=np.arange(len(points)), cmap="viridis", s=4, zorder=2
    )
    nash = np.full(3, 1 / 3) @ SIMPLEX_VERTICES
    ax.scatter(*nash, marker="*", color="tab:red", s=180, zorder=3, label="Nash")
    ax.scatter(*points[0], marker="s", color="k", s=40, zorder=3, label="start")
    for vertex, label, off, align in zip(
        SIMPLEX_VERTICES, RPS_LABELS, [(0, -0.05), (0, -0.05), (0, 0.03)],
        ["center", "center", "center"],
    ):
        ax.text(vertex[0] + off[0], vertex[1] + off[1], label, ha=align, fontsize=9)
    fig.colorbar(sc, ax=ax, label="iteration", shrink=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, 1.0)
    ax.axis("off")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("FP on RPS: average strategy spirals into Nash\n(instantaneous play cycles forever)")
    _save(fig, figures_dir / "fp_trajectory")


def _fig_exploitability(runs: dict, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (_, exploit) in runs.items():
        t = np.arange(1, len(exploit) + 1)
        ax.loglog(t, np.maximum(exploit, 1e-12), label=name.replace("_", " "))
    t = np.arange(1, len(next(iter(runs.values()))[1]) + 1)
    ax.loglog(t, 0.8 / np.sqrt(t), color="0.5", ls=":", lw=1,
              label=r"$\propto 1/\sqrt{t}$ (guide)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("full_exploitability of time-averaged profile")
    ax.set_title(r"FP exploitability decays roughly like $1/\sqrt{t}$")
    ax.legend(frameon=False)
    _save(fig, figures_dir / "fp_exploitability")


def _save(fig, stem: Path) -> None:
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
