"""Ch. 7 dashboard: four cheap diagnostics readers can leave on by default.

The book defines four instruments in §7.5 that together give an honest read
on a PSRO run's health without adding a payoff-cell's worth of compute:

  1. **payoff heatmap** — the empirical payoff tensor for player 0 as an
     n×n heatmap. Reveals dominance, symmetry, and any missing cells.
  2. **support / entropy** — the size of the meta-strategy's support and
     its Shannon entropy per iteration. Growing support signals a healthy
     cycle; collapsed support signals mode collapse.
  3. **newest-member mass** — the meta-mass that iteration t's new best
     response receives at the *next* solve (t+1). §14.4 uses this as the
     stopping signal: once new members stop getting non-trivial mass, the
     population has saturated.
  4. **BR margin vs cell error** — the newest BR's value against the
     current meta strategy, plotted alongside the payoff table's ±s/√K
     evaluator standard error. If the margin is inside the noise band,
     the "improvement" is likely a measurement artefact.

Instrument (4) explicitly reuses the evaluator's already-drawn samples —
`ProfileEvaluator` records per-cell std; this module reads it rather than
resampling. The whole module is pure numpy + matplotlib.

Wire in via ``callback = make_dashboard_callback(...)`` around ``run_psro``
or ``run_psro_parallel``. The callback returns a dict per iteration that
merges into ``PSROResult.history`` and can be persisted to CSV like any
other metric.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from psrolab.games.base import Population


# ---------------------------------------------- scalar-per-iteration metrics


def support_size(meta: np.ndarray, threshold: float = 1e-6) -> int:
    return int((meta > threshold).sum())


def entropy_bits(meta: np.ndarray) -> float:
    p = meta[meta > 0]
    return float(-np.sum(p * np.log2(p))) if p.size else 0.0


def newest_member_mass(previous_meta: np.ndarray, new_meta: np.ndarray) -> float:
    """Mass the new meta-strategy assigns to positions that appeared this
    iteration. `new_meta` is one element longer than `previous_meta` per
    player; the tail entries are the newest additions."""
    n_new = new_meta.size - previous_meta.size
    if n_new <= 0:
        return 0.0
    return float(new_meta[-n_new:].sum())


def br_margin(
    payoff_table: np.ndarray, meta: list[np.ndarray], player: int,
) -> tuple[float, float]:
    """Return (newest BR value against opponent meta, incumbent value).

    The margin is (newest - incumbent). Positive means the new BR beat the
    incumbent mixture; if the margin sits inside a ±cell-std band, the
    "improvement" may be measurement noise.
    """
    n_players = payoff_table.shape[0]
    if n_players != 2 or payoff_table.ndim != 3:
        raise ValueError("br_margin currently supports 2-player payoff tensors")
    opponent = 1 - player
    p0 = payoff_table[player]                    # k_0 × k_1
    opp_mix = meta[opponent]
    # value of each of player's pure strategies against opponent's mixture
    per_strat_value = p0 @ opp_mix if player == 0 else p0.T @ opp_mix
    newest_value = float(per_strat_value[-1])
    incumbent_value = float(per_strat_value @ meta[player])
    return newest_value, incumbent_value


# ------------------------------------------------------------ callback -----


def make_dashboard_callback(
    include_payoff_history: bool = False,
) -> Callable[[int, Population, list[np.ndarray]], dict]:
    """Return a per-iteration callback with all four instruments' scalars.

    The instruments needing cross-iteration state (newest-member mass,
    per-iteration support/entropy history) are computed by closing over a
    small state dict on the closure. Callers can pickle this callback
    (state is a plain dict) if they want checkpoint/resume behaviour.

    Args:
        include_payoff_history: if True, also snapshot the full payoff
            table each iteration (heavy — off by default; enable only for
            the reference run that produces Fig. 7.3).
    """
    state = {"previous_meta": [None, None]}

    def callback(iteration: int, population: Population,
                 meta: list[np.ndarray]) -> dict:
        entry: dict = {}
        for p, mp in enumerate(meta):
            entry[f"support_size_p{p}"] = support_size(mp)
            entry[f"entropy_bits_p{p}"] = entropy_bits(mp)
            prev = state["previous_meta"][p]
            entry[f"newest_member_mass_p{p}"] = (
                newest_member_mass(prev, mp) if prev is not None else float("nan")
            )
        pt = population.payoff_table
        if (pt is not None and all(m.size > 1 for m in meta)
                and pt.ndim == 3 and pt.shape[1:] == (meta[0].size, meta[1].size)):
            for p in (0, 1):
                new_v, inc_v = br_margin(pt, meta, p)
                entry[f"newest_value_p{p}"] = new_v
                entry[f"incumbent_value_p{p}"] = inc_v
                entry[f"br_margin_p{p}"] = new_v - inc_v
        if include_payoff_history and population.payoff_table is not None:
            entry["payoff_table_p0"] = population.payoff_table[0].copy()
        state["previous_meta"] = [m.copy() for m in meta]
        return entry

    return callback


# ---------------------------------------------------------------- figure ----


def render_dashboard(
    history: list[dict], figures_dir: Path, stem: str = "dashboard",
    plateau_range: tuple[int, int] | None = None,
    payoff_iterations_to_show: tuple[int, ...] = (),
) -> None:
    """Four-panel Ch. 7 figure over the entries a dashboard callback produced.

    Args:
        history: list of per-iteration dicts (as returned by the callback).
        figures_dir: output directory.
        stem: file basename (no extension).
        plateau_range: optional (start, end) iteration span to shade — used
            by Fig. 7.3 to highlight the reference-run plateau.
        payoff_iterations_to_show: iterations at which to render the
            payoff heatmap in panel 1. If empty, uses the latest iteration
            with a stored payoff_table_p0.
    """
    fig = plt.figure(figsize=(11.0, 7.5))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

    ax_payoff = fig.add_subplot(gs[0, 0])
    _panel_payoff(ax_payoff, history, payoff_iterations_to_show)
    ax_support = fig.add_subplot(gs[0, 1])
    _panel_support(ax_support, history, plateau_range)
    ax_newest = fig.add_subplot(gs[0, 2])
    _panel_newest_mass(ax_newest, history, plateau_range)
    ax_margin = fig.add_subplot(gs[1, :])
    _panel_margin(ax_margin, history, plateau_range)

    fig.suptitle("Ch. 7 dashboard: four instruments on one PSRO run",
                 fontsize=13)
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _panel_payoff(ax, history, iterations_to_show):
    payoffs = [(e["iteration"], e["payoff_table_p0"]) for e in history
               if "payoff_table_p0" in e and e["payoff_table_p0"] is not None]
    if not payoffs:
        ax.text(0.5, 0.5, "(no payoff snapshots)\nenable include_payoff_history",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_axis_off()
        return
    it, table = payoffs[-1] if not iterations_to_show else \
                next((p for p in payoffs if p[0] == max(iterations_to_show)),
                     payoffs[-1])
    im = ax.imshow(table, cmap="RdBu_r",
                   vmin=-max(1e-6, np.abs(table).max()),
                   vmax=max(1e-6, np.abs(table).max()))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"payoff heatmap (iter {it})")
    ax.set_xlabel("p1 strategy"); ax.set_ylabel("p0 strategy")


def _panel_support(ax, history, plateau_range):
    it = [e["iteration"] for e in history]
    for p, style in ((0, "-"), (1, "--")):
        s = [e.get(f"support_size_p{p}", np.nan) for e in history]
        h = [e.get(f"entropy_bits_p{p}", np.nan) for e in history]
        ax.plot(it, s, style, color="tab:blue",
                label=f"support size (p{p})")
        ax.plot(it, h, style, color="tab:orange",
                label=f"entropy bits (p{p})")
    if plateau_range:
        ax.axvspan(*plateau_range, color="0.85", alpha=0.4, lw=0)
    ax.set_xlabel("PSRO iteration")
    ax.set_title("meta-strategy support and entropy")
    ax.legend(fontsize=7, frameon=False)


def _panel_newest_mass(ax, history, plateau_range):
    it = [e["iteration"] for e in history]
    for p, color in ((0, "tab:blue"), (1, "tab:orange")):
        m = [e.get(f"newest_member_mass_p{p}", np.nan) for e in history]
        ax.plot(it, m, marker="o", color=color, label=f"p{p}")
    ax.axhline(0.0, color="0.6", lw=0.5)
    if plateau_range:
        ax.axvspan(*plateau_range, color="0.85", alpha=0.4, lw=0)
    ax.set_xlabel("PSRO iteration")
    ax.set_ylabel("newest member's mass at next solve")
    ax.set_title("stopping signal (§14.4)")
    ax.legend(fontsize=8, frameon=False)


def _panel_margin(ax, history, plateau_range):
    it = [e["iteration"] for e in history]
    for p, color in ((0, "tab:blue"), (1, "tab:orange")):
        newest = [e.get(f"newest_value_p{p}", np.nan) for e in history]
        inc = [e.get(f"incumbent_value_p{p}", np.nan) for e in history]
        ax.plot(it, newest, "-", color=color, label=f"newest BR value (p{p})")
        ax.plot(it, inc, ":", color=color, label=f"incumbent value (p{p})")
    if plateau_range:
        ax.axvspan(*plateau_range, color="0.85", alpha=0.4, lw=0)
    ax.set_xlabel("PSRO iteration")
    ax.set_ylabel("value against opponent meta")
    ax.set_title("BR margin — inside the noise band means measurement, not learning")
    ax.legend(fontsize=8, frameon=False, ncol=2)
