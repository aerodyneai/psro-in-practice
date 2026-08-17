"""Ch. 1 — Why games are hard: two studies for the free Leanpub sample.

Both studies are laptop-scale (pure numpy, seconds of compute) and produce
one figure each. The chapter uses them to motivate the whole book: naive
single-agent thinking breaks when the "environment" is another learner.

Study A — Independent Q-learners on matching pennies.
    Two tabular Q-learners, each treating the other as its environment.
    Neither converges; the pair orbits the mixed equilibrium (½, ½)
    indefinitely.
    Figure: independent_q_pennies.{pdf,png}

Study B — Joint-policy correlation (JPC) on a coordination game.
    k = 5 pairs of Q-learners, each pair trained independently (different
    seeds) on a k-color matching game. Cross-play matrix shows a strong
    diagonal (partners trained together match cleanly) and dim
    off-diagonal (swapped partners fail to coordinate).
    Figure: jpc_crossplay_matrix.{pdf,png}

Outputs:
    results/independent_q_pennies.csv
    results/jpc_crossplay.csv

Determinism: pure numpy + --seed; bit-reproducible. --smoke shortens the
Study A run and drops Study B to k=3 pairs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from psrolab.utils.plotstyle import apply_style

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- Study A ---
#
# Matching pennies with the row player wanting to match. Two stateless
# Q-learners choose action a ∈ {0=H, 1=T}. Payoffs:
#     row: +1 if same, -1 if different
#     col: -1 if same, +1 if different
# Each agent tracks Q[a] and picks softmax(Q / tau). After each play, both
# agents update Q[chosen] toward the observed reward.
#
# Hyperparameters (learning rate, temperature) are pinned in argparse
# defaults and chosen so the orbit is visually clean; provenance in NOTES.md
# per CLAUDE.md's no-silent-tuning rule.

MP_PAYOFF_ROW = np.array([[1, -1], [-1, 1]])


def _softmax_probs(q: np.ndarray, tau: float) -> np.ndarray:
    logits = q / tau
    logits -= logits.max()
    p = np.exp(logits)
    return p / p.sum()


def study_a(n_steps: int, lr: float, tau: float, seed: int) -> np.ndarray:
    """Return (n_steps, 2) trace of P(a=0) per agent per step."""
    rng = np.random.default_rng(seed)
    q0 = np.zeros(2)
    q1 = np.zeros(2)
    trace = np.zeros((n_steps, 2))
    for t in range(n_steps):
        p0 = _softmax_probs(q0, tau)
        p1 = _softmax_probs(q1, tau)
        trace[t] = (p0[0], p1[0])
        a0 = int(rng.choice(2, p=p0))
        a1 = int(rng.choice(2, p=p1))
        r0 = float(MP_PAYOFF_ROW[a0, a1])
        r1 = -r0
        q0[a0] += lr * (r0 - q0[a0])
        q1[a1] += lr * (r1 - q1[a1])
    return trace


# ---------------------------------------------------------------- Study B ---
#
# k-color matching game. Both players pick a color c ∈ {0, ..., k-1};
# reward is +1 if they match and 0 otherwise. Multiple strict Nash equilibria
# (one per colour); independent Q-learners with different seeds land on
# different colours, and the cross-play matrix's off-diagonal cells reveal
# the coordination failure.
#
# Each "pair" trains a pair of Q-learners together (same seed for both
# players in a pair, distinct seed across pairs); we then evaluate every
# (pair i's agent 0) x (pair j's agent 1) crossing.


def _train_pair(
    k_colors: int, n_steps: int, lr: float, tau: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Train two coordinating Q-learners; return their final action probs."""
    rng = np.random.default_rng(seed)
    q0 = np.zeros(k_colors)
    q1 = np.zeros(k_colors)
    for _ in range(n_steps):
        p0 = _softmax_probs(q0, tau)
        p1 = _softmax_probs(q1, tau)
        a0 = int(rng.choice(k_colors, p=p0))
        a1 = int(rng.choice(k_colors, p=p1))
        r = 1.0 if a0 == a1 else 0.0
        q0[a0] += lr * (r - q0[a0])
        q1[a1] += lr * (r - q1[a1])
    return _softmax_probs(q0, tau), _softmax_probs(q1, tau)


def study_b(
    n_pairs: int, k_colors: int, n_steps: int, lr: float, tau: float,
    seed: int,
) -> np.ndarray:
    """Return (n_pairs, n_pairs) cross-play expected reward matrix."""
    pairs = [_train_pair(k_colors, n_steps, lr, tau, seed + 10 * i)
             for i in range(n_pairs)]
    mat = np.zeros((n_pairs, n_pairs))
    for i in range(n_pairs):
        for j in range(n_pairs):
            p_i0 = pairs[i][0]         # pair i's agent 0
            p_j1 = pairs[j][1]         # pair j's agent 1
            mat[i, j] = float((p_i0 * p_j1).sum())  # E[1{same colour}]
    return mat


# ------------------------------------------------------------------- main ---


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="tiny config for CI (<60s)")
    parser.add_argument("--figdir", type=str, default=None,
                        help="override figures directory (ignored when --smoke is set, "
                             "so smoke never writes to tracked paths)")
    parser.add_argument("--plot-only", action="store_true",
                        help="skip training; regenerate figures from the committed CSVs")
    # Study A defaults — lr and tau picked so agents visibly orbit around
    # (0.5, 0.5) rather than damping to it or saturating to the corners.
    # See NOTES.md for the sweep that produced these values.
    parser.add_argument("--a-steps", type=int, default=30000)
    parser.add_argument("--a-lr", type=float, default=0.15)
    parser.add_argument("--a-tau", type=float, default=1.0)
    # Study B defaults
    parser.add_argument("--b-pairs", type=int, default=5)
    parser.add_argument("--b-colors", type=int, default=5)
    parser.add_argument("--b-steps", type=int, default=20000)
    parser.add_argument("--b-lr", type=float, default=0.05)
    parser.add_argument("--b-tau", type=float, default=0.05)
    args = parser.parse_args()

    if args.smoke:
        args.a_steps = 3000
        args.b_pairs, args.b_steps = 3, 2000

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

    # -- Study A --
    print(f"Study A: {args.a_steps} steps, lr={args.a_lr}, tau={args.a_tau}, seed={args.seed}")
    trace = study_a(args.a_steps, args.a_lr, args.a_tau, args.seed)
    with open(results_dir / "independent_q_pennies.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "p0_heads", "p1_heads"])
        for t in range(trace.shape[0]):
            w.writerow([t, f"{trace[t, 0]:.6f}", f"{trace[t, 1]:.6f}"])
    dev = float(np.mean(np.sqrt((trace[:, 0] - 0.5) ** 2 + (trace[:, 1] - 0.5) ** 2)))
    print(f"  mean radial deviation from (0.5, 0.5): {dev:.4f} "
          f"(a clean orbit has ~0.05-0.25)")

    # -- Study B --
    print(f"Study B: {args.b_pairs} pairs x {args.b_colors} colors x "
          f"{args.b_steps} steps, seed base={args.seed}")
    mat = study_b(args.b_pairs, args.b_colors, args.b_steps,
                  args.b_lr, args.b_tau, args.seed)
    with open(results_dir / "jpc_crossplay.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_i", "pair_j", "expected_match_prob"])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                w.writerow([i, j, f"{mat[i, j]:.6f}"])
    diag = float(np.mean(np.diag(mat)))
    off = float((mat.sum() - np.trace(mat)) / (mat.size - mat.shape[0]))
    gap = diag - off
    print(f"  diagonal mean: {diag:.3f}, off-diagonal mean: {off:.3f}, "
          f"gap = {gap:.3f}")
    print(f"  --> record this number in the manuscript §1.3 in place of "
          f"any placeholder.")

    _plot_a(trace, figures_dir, args)
    _plot_b(mat, figures_dir, args)
    print(f"Wrote CSVs to {results_dir} and figures to {figures_dir}")


# ------------------------------------------------------------- plotting ----


def _plot_from_csv(results_dir: Path, figures_dir: Path) -> None:
    trace_rows = list(csv.DictReader(open(results_dir / "independent_q_pennies.csv")))
    trace = np.array([[float(r["p0_heads"]), float(r["p1_heads"])] for r in trace_rows])
    _plot_a(trace, figures_dir, args=None)
    mat_rows = list(csv.DictReader(open(results_dir / "jpc_crossplay.csv")))
    n = max(int(r["pair_i"]) for r in mat_rows) + 1
    mat = np.zeros((n, n))
    for r in mat_rows:
        mat[int(r["pair_i"]), int(r["pair_j"])] = float(r["expected_match_prob"])
    _plot_b(mat, figures_dir, args=None)


def _plot_a(trace: np.ndarray, figures_dir: Path, args) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    n = trace.shape[0]
    step = np.arange(n)
    ax.plot(step, trace[:, 0], color="tab:blue", lw=0.7, label="agent 0 P(heads)")
    ax.plot(step, trace[:, 1], color="tab:orange", lw=0.7, label="agent 1 P(heads)")
    ax.axhline(0.5, color="0.4", ls="--", lw=0.8)
    ax.set_xlabel("training step")
    ax.set_ylabel("P(heads)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Independent Q-learners orbit the mixed equilibrium\n"
                 "matching pennies; neither converges")
    ax.legend(loc="upper right")

    # Inset: (p0, p1) trajectory, colour-graded by time.
    tail = max(n // 5, 500)
    inset = ax.inset_axes((0.62, 0.10, 0.35, 0.36))
    idx = np.linspace(n - tail, n - 1, min(tail, 2000)).astype(int)
    inset.plot(trace[idx, 0], trace[idx, 1], color="0.3", lw=0.5)
    inset.scatter(trace[idx, 0], trace[idx, 1], c=idx, cmap="viridis", s=1)
    inset.scatter([0.5], [0.5], marker="+", color="k", s=60, lw=1.4)
    inset.set_xlim(0, 1); inset.set_ylim(0, 1)
    inset.set_xticks([0, 0.5, 1]); inset.set_yticks([0, 0.5, 1])
    inset.set_xlabel("agent 0", fontsize=8); inset.set_ylabel("agent 1", fontsize=8)
    inset.set_title(f"last {tail} steps", fontsize=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"independent_q_pennies.{ext}")
    plt.close(fig)


def _plot_b(mat: np.ndarray, figures_dir: Path, args) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0, origin="lower")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            colour = "white" if mat[i, j] < 0.5 else "black"
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color=colour)
    ax.set_xlabel("pair j's agent 1")
    ax.set_ylabel("pair i's agent 0")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_yticks(range(mat.shape[0]))
    diag = np.mean(np.diag(mat))
    off = (mat.sum() - np.trace(mat)) / (mat.size - mat.shape[0])
    ax.set_title("Joint-policy correlation on a 5-colour matching game\n"
                 f"diag={diag:.2f}, off-diag={off:.2f}, gap={diag - off:.2f}")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04, label="E[reward]")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"jpc_crossplay_matrix.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
