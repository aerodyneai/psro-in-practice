"""Ch. 2 — Nash equilibria from scratch on three small matrix games.

Solves rock-paper-scissors, matching pennies, and a weighted (non-transitive)
RPS variant by (a) support enumeration and (b) the maximin LP, cross-checking
that both find a zero-exploitability profile with the same game value.

Outputs:
    results/nash_solutions.csv       one row per (game, method)
    figures/nash_simplex.{pdf,png}   strategy simplex per game with Nash marked

Book figure: strategy simplex with Nash equilibria (Fig. 2.x).

Fully deterministic: both solvers are exact; --seed is accepted for interface
uniformity but unused. Runs in well under a minute — no --smoke mode needed.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from psrolab.eval import restricted_exploitability
from psrolab.games import MatrixGame
from psrolab.meta_solvers import NashSolverLP, SupportEnumerationSolver

HERE = Path(__file__).resolve().parent

# Weighted RPS: winning with rock scores double. Antisymmetric, so still
# zero-sum with value 0, but the Nash shifts to (1/4, 1/2, 1/4) — paper is
# played most because it beats the high-stakes rock. First non-uniform Nash
# readers meet, and the book's running example of a non-transitive game.
GAMES: dict[str, tuple[np.ndarray, list[str]]] = {
    "rock_paper_scissors": (
        np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float),
        ["Rock", "Paper", "Scissors"],
    ),
    "weighted_rps": (
        np.array([[0, -1, 2], [1, 0, -1], [-2, 1, 0]], dtype=float),
        ["Rock", "Paper", "Scissors"],
    ),
    "matching_pennies": (
        np.array([[1, -1], [-1, 1]], dtype=float),
        ["Heads", "Tails"],
    ),
}

SIMPLEX_VERTICES = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="unused; solvers are exact")
    parser.parse_args()
    plt.switch_backend("Agg")

    results_dir = HERE / "results"
    figures_dir = HERE / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    solvers = {"support_enumeration": SupportEnumerationSolver(), "lp": NashSolverLP()}
    rows = []
    solutions: dict[str, dict[str, list[np.ndarray]]] = {}
    for game_name, (a, _) in GAMES.items():
        game = MatrixGame(payoffs=np.stack([a, -a]))
        solutions[game_name] = {}
        for method, solver in solvers.items():
            t0 = time.perf_counter()
            mixtures = solver.solve(game)
            runtime_ms = 1e3 * (time.perf_counter() - t0)
            solutions[game_name][method] = mixtures
            rows.append(
                {
                    "game": game_name,
                    "method": method,
                    "x": ";".join(f"{p:.6f}" for p in mixtures[0]),
                    "y": ";".join(f"{p:.6f}" for p in mixtures[1]),
                    "value": f"{game.expected_payoffs(mixtures)[0]:.6f}",
                    # The game IS the full game here — no population restriction.
                    "full_exploitability": f"{restricted_exploitability(game, mixtures):.2e}",
                    "runtime_ms": f"{runtime_ms:.3f}",
                }
            )

    csv_path = results_dir / "nash_solutions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['game']:>22} {row['method']:>20}  x=({row['x']})  "
            f"exploit={row['full_exploitability']}  {row['runtime_ms']}ms"
        )

    _render_figure(solutions, figures_dir)
    print(f"\nWrote {csv_path} and {figures_dir / 'nash_simplex.(pdf|png)'}")


def _render_figure(solutions: dict, figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, game_name in zip(axes, GAMES):
        labels = GAMES[game_name][1]
        if len(labels) == 3:
            _draw_simplex(ax, labels)
            for method, marker, color in [
                ("support_enumeration", "X", "tab:red"),
                ("lp", "o", "tab:blue"),
            ]:
                point = solutions[game_name][method][0] @ SIMPLEX_VERTICES
                ax.scatter(*point, marker=marker, s=110, facecolors="none" if marker == "o"
                           else color, edgecolors=color, linewidths=1.8, zorder=3,
                           label=method.replace("_", " "))
        else:
            _draw_segment(ax, labels)
            for method, marker, color in [
                ("support_enumeration", "X", "tab:red"),
                ("lp", "o", "tab:blue"),
            ]:
                p_heads = solutions[game_name][method][0][0]
                ax.scatter(p_heads, 0.0, marker=marker, s=110,
                           facecolors="none" if marker == "o" else color, edgecolors=color,
                           linewidths=1.8, zorder=3, label=method.replace("_", " "))
        ax.set_title(game_name.replace("_", " "))
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("Nash equilibria: support enumeration vs LP (markers coincide)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"nash_simplex.{ext}", dpi=200)
    plt.close(fig)


def _draw_simplex(ax: plt.Axes, labels: list[str]) -> None:
    triangle = np.vstack([SIMPLEX_VERTICES, SIMPLEX_VERTICES[0]])
    ax.plot(triangle[:, 0], triangle[:, 1], color="0.3", lw=1)
    offsets = [(-0.03, -0.05), (0.03, -0.05), (0.0, 0.04)]
    aligns = ["right", "left", "center"]
    for vertex, label, off, align in zip(SIMPLEX_VERTICES, labels, offsets, aligns):
        ax.text(vertex[0] + off[0], vertex[1] + off[1], label, ha=align, fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.15, 1.05)


def _draw_segment(ax: plt.Axes, labels: list[str]) -> None:
    ax.plot([0, 1], [0, 0], color="0.3", lw=1)
    ax.text(0, -0.05, f"always {labels[0]}", ha="center", fontsize=9)
    ax.text(1, -0.05, f"always {labels[1]}", ha="center", fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.6, 0.6)


if __name__ == "__main__":
    main()
