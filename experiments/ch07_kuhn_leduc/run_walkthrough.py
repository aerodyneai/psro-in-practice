"""Ch. 7 walkthrough — one Kuhn+tabular+Nash run traced by the dashboard.

Produces Fig 7.1 (payoff heatmaps at t=1,3,5,10 with support sizes),
along with per-iteration values for the four §7.2 marginal-number
call-outs (seed policy's meta-mass at t=2, support sizes/entropy per t,
BR value vs incumbent per t).

Outputs:
    results/walkthrough_history.csv   per-iteration dashboard entries
    figures/kuhn_walkthrough.{pdf,png} Fig 7.1

Fast: <2 minutes on the reference server, no GPU. Wired to the shared
``psrolab.eval.dashboard`` module so the same instruments serve Fig 7.3
(Leduc reference run) and Ch. 13's capstone re-expression.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from psrolab import run_psro
from psrolab.eval import ProfileEvaluator
from psrolab.eval.dashboard import make_dashboard_callback
from psrolab.eval.openspiel_exploitability import full_exploitability
from psrolab.games.openspiel_wrap import OpenSpielGame
from psrolab.meta_solvers import ZeroSumProjectionNash
from psrolab.oracles.tabular_q import TabularQOracle
from psrolab.utils.plotstyle import apply_style

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--oracle-episodes", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--figdir", type=str, default=None)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.iterations, args.oracle_episodes, args.eval_episodes = 3, 1000, 200
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

    game = OpenSpielGame("kuhn_poker", seed=args.seed)
    dash = make_dashboard_callback(include_payoff_history=True)
    result = run_psro(
        game=game,
        oracle=TabularQOracle(n_episodes=args.oracle_episodes),
        meta_solver=ZeroSumProjectionNash(),
        evaluator=ProfileEvaluator(n_episodes_per_profile=args.eval_episodes),
        n_iterations=args.iterations,
        seed=args.seed,
        callbacks=[
            dash,
            lambda it, pop, meta, g=game: {
                "full_exploitability": full_exploitability(g, pop, meta)
            },
        ],
    )
    rows = []
    for e in result.history:
        row = {"iteration": e["iteration"]}
        for k, v in e.items():
            if k == "iteration" or k == "payoff_table_p0":
                continue
            row[k] = v
        rows.append(row)
    _write_history(results_dir / "walkthrough_history.csv", rows)

    # save payoff snapshots as .npy alongside (needed for the heatmap panel)
    for e in result.history:
        if "payoff_table_p0" in e:
            np.save(results_dir / f"payoff_p0_iter{e['iteration']:02d}.npy",
                    e["payoff_table_p0"])

    _plot(result.history, figures_dir)
    _print_call_outs(result.history)
    print(f"Wrote history to {results_dir} and figures to {figures_dir}")


def _write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _plot(history: list[dict], figures_dir: Path) -> None:
    # 2 x 3 grid: top row = 3 heatmaps (iter 1/3/5), bottom row = 3 metric panels.
    payoff_iters = [1, 3, 5]
    payoffs = {e["iteration"]: e["payoff_table_p0"] for e in history
               if "payoff_table_p0" in e}
    fig = plt.figure(figsize=(11.5, 6.5))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)
    for i, it in enumerate(payoff_iters):
        ax = fig.add_subplot(gs[0, i])
        if it in payoffs and payoffs[it] is not None:
            t = payoffs[it]
            im = ax.imshow(t, cmap="RdBu_r",
                           vmin=-max(1e-6, np.abs(t).max()),
                           vmax=max(1e-6, np.abs(t).max()))
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"payoff table @ iter {it}")
        else:
            ax.text(0.5, 0.5, f"no snapshot @ iter {it}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

    ax_supp = fig.add_subplot(gs[1, 0])
    it = [e["iteration"] for e in history]
    for p, ls in ((0, "-"), (1, "--")):
        ax_supp.plot(it, [e.get(f"support_size_p{p}", np.nan) for e in history],
                     ls, color="tab:blue", label=f"support (p{p})")
        ax_supp.plot(it, [e.get(f"entropy_bits_p{p}", np.nan) for e in history],
                     ls, color="tab:orange", label=f"entropy (p{p})")
    ax_supp.set_xlabel("iteration"); ax_supp.set_title("support size and entropy")
    ax_supp.legend(fontsize=7, frameon=False)

    ax_new = fig.add_subplot(gs[1, 1])
    for p, c in ((0, "tab:blue"), (1, "tab:orange")):
        ax_new.plot(it, [e.get(f"newest_member_mass_p{p}", np.nan) for e in history],
                    marker="o", color=c, label=f"p{p}")
    ax_new.axhline(0.0, color="0.6", lw=0.5)
    ax_new.set_xlabel("iteration"); ax_new.set_title("newest member mass")
    ax_new.legend(fontsize=7, frameon=False)

    ax_marg = fig.add_subplot(gs[1, 2])
    for p, c in ((0, "tab:blue"), (1, "tab:orange")):
        ax_marg.plot(it, [e.get(f"newest_value_p{p}", np.nan) for e in history],
                     "-", color=c, label=f"newest (p{p})")
        ax_marg.plot(it, [e.get(f"incumbent_value_p{p}", np.nan) for e in history],
                     ":", color=c, label=f"incumbent (p{p})")
    ax_marg.set_xlabel("iteration"); ax_marg.set_title("BR value vs incumbent")
    ax_marg.legend(fontsize=7, frameon=False, ncol=2)

    fig.suptitle("Kuhn walkthrough — the four §7.5 instruments on one run",
                 fontsize=12)
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"kuhn_walkthrough.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


def _print_call_outs(history: list[dict]) -> None:
    """The four §7.2 call-outs the manuscript is going to slot numbers into."""
    print("\n§7.2 call-outs (report these back to fill the manuscript's bands):")
    for e in history:
        it = e["iteration"]
        s0 = e.get("support_size_p0"); h0 = e.get("entropy_bits_p0")
        s1 = e.get("support_size_p1"); h1 = e.get("entropy_bits_p1")
        m0 = e.get("newest_member_mass_p0"); m1 = e.get("newest_member_mass_p1")
        nv = e.get("newest_value_p0"); iv = e.get("incumbent_value_p0")
        expl = e.get("full_exploitability")
        parts = [f"iter {it:>2}",
                 f"support p0/p1 = {s0}/{s1}",
                 f"entropy p0/p1 = {h0:.2f}/{h1:.2f}"
                 if h0 is not None else "entropy p0/p1 = -",
                 f"newest mass p0/p1 = {m0:.3f}/{m1:.3f}"
                 if isinstance(m0, float) else "newest mass p0/p1 = -",
                 f"BR value p0 = {nv:.3f} vs incumbent {iv:.3f}"
                 if isinstance(nv, float) else "BR value p0 = -",
                 f"e_G = {expl:.4f}" if expl is not None else "e_G = -"]
        print("  " + " | ".join(parts))


def _plot_from_csv(results_dir: Path, figures_dir: Path) -> None:
    with open(results_dir / "walkthrough_history.csv") as f:
        rows = list(csv.DictReader(f))
    history: list[dict] = []
    for r in rows:
        entry: dict = {"iteration": int(r["iteration"])}
        for k, v in r.items():
            if k == "iteration":
                continue
            try:
                entry[k] = float(v)
            except (TypeError, ValueError):
                continue
        # attach payoff snapshots from disk when present
        p_path = results_dir / f"payoff_p0_iter{entry['iteration']:02d}.npy"
        if p_path.exists():
            entry["payoff_table_p0"] = np.load(p_path)
        history.append(entry)
    _plot(history, figures_dir)


if __name__ == "__main__":
    main()
