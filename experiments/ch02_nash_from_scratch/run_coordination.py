"""Ch. 2 Pitfall 2.1 — coordination-game panel with multiple Nash equilibria.

Battle of the Sexes: two pure-strategy Nash and one mixed. Support-enumeration
finds all three; the LP solver (zero-sum only) is not applicable. The panel
places all three equilibria on the (p_row, p_col) unit square so a reader can
see why "the Nash equilibrium" is a category error in general-sum games.

Outputs:
    results/coordination_nash.csv     one row per equilibrium
    figures/coordination_nash.{pdf,png}
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from psrolab.games import MatrixGame
from psrolab.meta_solvers.support_enum import enumerate_nash
from psrolab.utils.plotstyle import apply_style

HERE = Path(__file__).resolve().parent


# Battle of the Sexes. row = player 0's payoffs; col = player 1's.
ROW = np.array([[3.0, 0.0], [0.0, 2.0]])
COL = np.array([[2.0, 0.0], [0.0, 3.0]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="unused; solver is exact")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--figdir", type=str, default=None)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
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
        with open(results_dir / "coordination_nash.csv") as f:
            rows = list(csv.DictReader(f))
        _fig(rows, figures_dir)
        print(f"Wrote figures to {figures_dir}")
        return

    game = MatrixGame(payoffs=np.stack([ROW, COL]))
    eqs = enumerate_nash(game)
    rows = []
    for i, eq in enumerate(eqs):
        p0, p1 = eq[0], eq[1]
        row_value = float(p0 @ ROW @ p1)
        col_value = float(p0 @ COL @ p1)
        rows.append({
            "index": i,
            "p0_opera": f"{p0[0]:.6f}",
            "p1_opera": f"{p1[0]:.6f}",
            "p0_ballet": f"{p0[1]:.6f}",
            "p1_ballet": f"{p1[1]:.6f}",
            "p0_value": f"{row_value:.6f}",
            "p1_value": f"{col_value:.6f}",
        })
    _write(results_dir / "coordination_nash.csv", rows)

    print(f"Battle of the Sexes: {len(eqs)} Nash equilibria")
    for r in rows:
        print(f"  eq {r['index']}: p0=({r['p0_opera']}, {r['p0_ballet']}), "
              f"p1=({r['p1_opera']}, {r['p1_ballet']}), "
              f"values=({r['p0_value']}, {r['p1_value']})")
    _fig(rows, figures_dir)
    print(f"Wrote {results_dir}/coordination_nash.csv and figures to {figures_dir}")


def _write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def _fig(rows: list[dict], figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    # Payoff heatmaps as background (row's view + col's view diagonal)
    ax.set_xlabel("player 1 P(Opera)")
    ax.set_ylabel("player 0 P(Opera)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.plot([0, 1], [0, 1], color="0.85", lw=0.8)
    ax.axhline(0.5, color="0.85", lw=0.5)
    ax.axvline(0.5, color="0.85", lw=0.5)
    markers = ["o", "s", "^", "D"]
    colours = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i, r in enumerate(rows):
        x = float(r["p1_opera"]); y = float(r["p0_opera"])
        m = markers[i % len(markers)]; c = colours[i % len(colours)]
        ax.scatter([x], [y], marker=m, color=c, s=140, zorder=3, edgecolors="k",
                   label=f"eq {i}: p0={y:.2f}, p1={x:.2f}, "
                         f"value=({r['p0_value']}, {r['p1_value']})")
    ax.set_title("Battle of the Sexes: three Nash equilibria coexist\n"
                 "'the' Nash is not a category in general-sum games")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35),
              fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"coordination_nash.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
