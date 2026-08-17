"""Ch. 3 — Shapley's 3x3 counterexample to fictitious-play convergence.

Shapley (1964) constructed a 3×3 general-sum game on which fictitious play's
time-averaged strategies **do not converge** — they trace a limit cycle. This
is the concrete example the chapter cites for "even FP's convergence
guarantee has caveats" (holds for 2-player zero-sum; fails in general).

Payoff matrices (Foster-Young 1998 formulation of the Shapley counterexample):

    Player 0:  A = [[1, 0, 0],  Player 1:  B = [[0, 1, 0],
                    [0, 1, 0],                   [0, 0, 1],
                    [0, 0, 1]]                   [1, 0, 0]]

    Player 0 wants to *match* (row=col); player 1 wants to be *one ahead*
    ((col = row + 1) mod 3). The BR pair chases itself around a 6-cycle,
    the empirical averages never settle, and the exploitability of FP's
    time-average sits at a constant floor.

Outputs:
    results/shapley_fp.csv           per-iteration averages + BR indices
    figures/shapley_limit_cycle.{pdf,png}   trajectory + non-converging e_G

Deterministic, seconds of compute. Uses the general fictitious-play
implementation from ``psrolab.baselines.fp`` (which already supports any
2-player MatrixGame, not just zero-sum).
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
from psrolab.utils.plotstyle import apply_style

HERE = Path(__file__).resolve().parent
SIMPLEX_VERTICES = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])

ROW = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
COL = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="unused; FP is deterministic")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--figdir", type=str, default=None)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.iterations = 300
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
        with open(results_dir / "shapley_fp.csv") as f:
            rows = list(csv.DictReader(f))
        _fig(rows, figures_dir)
        print(f"Wrote figures to {figures_dir}")
        return

    game = MatrixGame(payoffs=np.stack([ROW, COL]))
    fp = FictitiousPlay(game)
    result = fp.run(args.iterations)
    rows = []
    for t in range(args.iterations):
        p0 = result.averages[0][t]; p1 = result.averages[1][t]
        # For general-sum, use max(BR value − current value) across players
        # as the exploitability convention (matches restricted_exploitability).
        expl = restricted_exploitability(game, [p0, p1])
        rows.append({
            "iteration": t,
            "p0_a1": f"{p0[0]:.6f}", "p0_a2": f"{p0[1]:.6f}", "p0_a3": f"{p0[2]:.6f}",
            "p1_a1": f"{p1[0]:.6f}", "p1_a2": f"{p1[1]:.6f}", "p1_a3": f"{p1[2]:.6f}",
            "br_p0": int(result.best_responses[t, 0]),
            "br_p1": int(result.best_responses[t, 1]),
            "full_exploitability": f"{expl:.6f}",
        })
    _write(results_dir / "shapley_fp.csv", rows)

    final_p0 = result.averages[0][-1]; final_p1 = result.averages[1][-1]
    print(f"final averages: p0={final_p0.round(3)}, p1={final_p1.round(3)}")
    print(f"final exploitability = {float(rows[-1]['full_exploitability']):.4f}")
    print(f"BR sequence's cycle period (last 30 iters, p0): "
          f"{result.best_responses[-30:, 0].tolist()}")
    _fig(rows, figures_dir)
    print(f"Wrote {results_dir}/shapley_fp.csv and figures to {figures_dir}")


def _write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def _fig(rows: list[dict], figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: player 0's time-averaged trajectory on the simplex, coloured by t.
    ax = axes[0]
    triangle = np.vstack([SIMPLEX_VERTICES, SIMPLEX_VERTICES[0]])
    ax.plot(triangle[:, 0], triangle[:, 1], color="0.4", lw=1)
    for vertex, label in zip(SIMPLEX_VERTICES, ["a1", "a2", "a3"]):
        ax.text(vertex[0], vertex[1], label, ha="center", va="center", fontsize=9)
    xs, ys, ts = [], [], []
    for r in rows:
        p = np.array([float(r["p0_a1"]), float(r["p0_a2"]), float(r["p0_a3"])])
        pt = p @ SIMPLEX_VERTICES
        xs.append(pt[0]); ys.append(pt[1]); ts.append(int(r["iteration"]))
    ax.scatter(xs, ys, c=ts, cmap="viridis", s=2)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title("player 0's time-averaged trajectory on the simplex\n"
                 "colour = iteration; the average never settles")

    # Panel B: exploitability vs iteration (no polynomial decay for Shapley).
    ax = axes[1]
    it = [int(r["iteration"]) + 1 for r in rows]
    expl = [max(float(r["full_exploitability"]), 1e-8) for r in rows]
    ax.loglog(it, expl, color="tab:red", lw=0.8)
    # For contrast, overlay a 1/sqrt(t) reference (what FP achieves on 2p ZS)
    ref = 1.0 / np.sqrt(np.array(it))
    ax.loglog(it, ref, color="0.6", ls="--", label=r"$1/\sqrt{t}$ (2p ZS reference)")
    ax.set_xlabel("iteration"); ax.set_ylabel("full_exploitability")
    ax.set_title("Shapley's game: no polynomial decay,\n"
                 "empirical averages ride a limit cycle")
    ax.legend(frameon=False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"shapley_limit_cycle.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
