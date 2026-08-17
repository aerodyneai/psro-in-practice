"""Ch. 6 Fig 6.1 — BR quality vs training episodes.

Freezes one intermediate PSRO population + Nash mixture as a **serialized
shared reference target** (same target for every budget point), then trains
each oracle at n_episodes in ``--episode-grid`` for ``--n-seeds`` seeds
each, and reports the value gap  u(BR_exact, σ) − u(ABR, σ)  where BR_exact
comes from OpenSpiel's best-response calculator.

At low budgets the gap is dominated by the tabular oracle's
unseen-state defaults (Kuhn has 12 info states, so this bites quickly);
the plot's low-episode side is meant to show that explicitly.

Outputs:
    results/reference_target.pkl   frozen population + mixture (built once)
    results/br_quality.csv         value gaps
    figures/br_quality.{pdf,png}   Fig 6.1

Compute: 5 grid points x 2 oracles x 5 seeds = 50 runs at PPO's cost.
Reference-target build is ~2 min once.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from psrolab import run_psro
from psrolab.eval import ProfileEvaluator
from psrolab.eval.openspiel_exploitability import full_exploitability
from psrolab.games.openspiel_wrap import OpenSpielGame
from psrolab.meta_solvers import ZeroSumProjectionNash
from psrolab.oracles.ppo import PPOOracle
from psrolab.oracles.tabular_q import TabularQOracle
from psrolab.utils.plotstyle import apply_style

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--reference-iterations", type=int, default=5,
                        help="PSRO iterations to build the reference target")
    parser.add_argument("--reference-episodes", type=int, default=10000,
                        help="episodes per BR in the reference build")
    parser.add_argument("--episode-grid", type=str,
                        default="500,2000,5000,20000,80000")
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument("--target-player", type=int, default=0)
    parser.add_argument("--force-rebuild", action="store_true",
                        help="rebuild reference target even if pickle exists")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--figdir", type=str, default=None)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.n_seeds = 1
        args.reference_iterations, args.reference_episodes = 2, 500
        args.episode_grid = "200,800"
    plt.switch_backend("Agg")
    apply_style()

    grid = [int(x) for x in args.episode_grid.split(",")]

    suffix = "_smoke" if args.smoke else ""
    results_dir = HERE / f"results{suffix}"
    if args.smoke or not args.figdir:
        figures_dir = HERE / f"figures{suffix}"
    else:
        figures_dir = Path(args.figdir)
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        with open(results_dir / "br_quality.csv") as f:
            rows = list(csv.DictReader(f))
        _fig(rows, figures_dir)
        print(f"Wrote figures to {figures_dir}")
        return

    ref_path = results_dir / "reference_target.pkl"
    if ref_path.exists() and not args.force_rebuild:
        with open(ref_path, "rb") as f:
            reference = pickle.load(f)
        print(f"Loaded reference target from {ref_path}")
    else:
        reference = _build_reference(args)
        with open(ref_path, "wb") as f:
            pickle.dump(reference, f)
        print(f"Built and saved reference target to {ref_path}")

    game = OpenSpielGame("kuhn_poker", seed=args.seed)
    br_exact_value = _compute_exact_br_value(
        game, reference, target_player=args.target_player,
    )
    print(f"Reference target: player {args.target_player}'s exact-BR value = "
          f"{br_exact_value:.6f}")

    rows: list[dict] = []
    for oracle_name in ("ppo", "tabular_q"):
        for n_ep in grid:
            for s in range(args.n_seeds):
                rows.append(_run(oracle_name, n_ep, args.seed + s,
                                 game, reference, br_exact_value, args))
                _write(results_dir / "br_quality.csv", rows)
    _fig(rows, figures_dir)
    print(f"Wrote {results_dir}/br_quality.csv and figures to {figures_dir}")


def _build_reference(args) -> dict:
    """Warm-up PSRO run: freeze its population + Nash mixture as the target."""
    game = OpenSpielGame("kuhn_poker", seed=args.seed)
    oracle = PPOOracle(total_episodes=args.reference_episodes, device="cpu")
    result = run_psro(
        game=game, oracle=oracle,
        meta_solver=ZeroSumProjectionNash(),
        evaluator=ProfileEvaluator(n_episodes_per_profile=args.eval_episodes),
        n_iterations=args.reference_iterations, seed=args.seed,
        callbacks=[
            lambda it, pop, meta, g=game: {
                "full_exploitability": full_exploitability(g, pop, meta)
            }
        ],
    )
    return {
        "population": result.population,
        "meta_strategies": result.meta_strategies,
        "final_full_exploitability":
            result.history[-1].get("full_exploitability"),
    }


def _compute_exact_br_value(game: OpenSpielGame, reference: dict,
                            target_player: int) -> float:
    """Value of the exact best response against reference's opponent mixture."""
    opp_policy, joint_policy = _joint_reference_policy(
        game, reference, target_player,
    )
    from open_spiel.python.algorithms import best_response
    br = best_response.BestResponsePolicy(game.raw_game, target_player, opp_policy)
    return float(br.value(game.raw_game.new_initial_state()))


def _joint_reference_policy(game: OpenSpielGame, reference: dict,
                            target_player: int):
    """Build (opponent TabularPolicy alone, joint TabularPolicy) from reference."""
    from open_spiel.python import policy as os_policy
    from psrolab.eval.openspiel_exploitability import _reach_weighted_action_probs
    tabular = os_policy.TabularPolicy(game.raw_game)
    opp = 1 - target_player
    for player in range(2):
        weights = _reach_weighted_action_probs(
            game, reference["population"], reference["meta_strategies"], player,
        )
        for info_state, probs in weights.items():
            total = probs.sum()
            if total <= 0.0:
                continue
            row = tabular.policy_for_key(info_state)
            row[:] = probs / total
    # opponent-only view (target_player uses uniform default; but downstream
    # `BestResponsePolicy(game, target_player, opp_policy)` only queries
    # opp's action_probabilities so we can safely reuse `tabular` for both).
    return tabular, tabular


def _run(oracle_name: str, n_episodes: int, seed: int,
         game: OpenSpielGame, reference: dict, br_exact_value: float,
         args) -> dict:
    if oracle_name == "ppo":
        oracle = PPOOracle(total_episodes=n_episodes, device="cpu")
    elif oracle_name == "tabular_q":
        oracle = TabularQOracle(n_episodes=n_episodes)
    else:
        raise ValueError
    t0 = time.perf_counter()
    br_policy = oracle.best_response(
        game=game, player=args.target_player,
        population=reference["population"],
        meta_strategies=reference["meta_strategies"],
        rng=np.random.default_rng(seed),
    )
    wall_s = time.perf_counter() - t0

    br_value = _policy_value_against_reference(
        game, br_policy, reference, args.target_player,
    )
    gap = br_exact_value - br_value
    print(f"{oracle_name:>9} n_ep={n_episodes:>6} seed {seed}: "
          f"BR value = {br_value:.4f}, gap = {gap:.4f} ({wall_s:.0f}s)",
          flush=True)
    return {
        "oracle": oracle_name, "n_episodes": n_episodes, "seed": seed,
        "target_player": args.target_player,
        "br_value": f"{br_value:.6f}",
        "exact_br_value": f"{br_exact_value:.6f}",
        "value_gap": f"{gap:.6f}",
        "wall_clock_s": f"{wall_s:.1f}",
    }


def _policy_value_against_reference(
    game: OpenSpielGame, br_policy, reference: dict, target_player: int,
) -> float:
    """Value to target_player of (br_policy, reference-opponent) under exact
    game dynamics — via OpenSpiel's expected_game_score."""
    from open_spiel.python import policy as os_policy
    from open_spiel.python.algorithms import expected_game_score
    from psrolab.eval.openspiel_exploitability import (
        _reach_weighted_action_probs,
    )

    tabular = os_policy.TabularPolicy(game.raw_game)
    opp = 1 - target_player
    # opponent side: reach-weighted mixture from the reference population.
    weights = _reach_weighted_action_probs(
        game, reference["population"], reference["meta_strategies"], opp,
    )
    for info_state, probs in weights.items():
        total = probs.sum()
        if total > 0.0:
            row = tabular.policy_for_key(info_state)
            row[:] = probs / total
    # target-player side: fill from br_policy's action_probabilities where
    # available, else one-hot(act).
    n_actions = game.raw_game.num_distinct_actions()
    state = game.raw_game.new_initial_state()
    _fill_br_side(game, br_policy, target_player, tabular, state,
                  visited=set(), n_actions=n_actions)
    values = expected_game_score.policy_value(
        game.raw_game.new_initial_state(), [tabular, tabular],
    )
    return float(values[target_player])


def _fill_br_side(game: OpenSpielGame, br_policy, target_player: int,
                  tabular, state, visited: set, n_actions: int) -> None:
    if state.is_terminal():
        return
    if state.is_chance_node():
        for a, _ in state.chance_outcomes():
            _fill_br_side(game, br_policy, target_player, tabular,
                          state.child(a), visited, n_actions)
        return
    if state.current_player() != target_player:
        for a in state.legal_actions():
            _fill_br_side(game, br_policy, target_player, tabular,
                          state.child(a), visited, n_actions)
        return
    key = state.information_state_string(target_player)
    if key not in visited:
        visited.add(key)
        obs = game.observation(state, target_player)
        legal = state.legal_actions()
        method = getattr(br_policy, "action_probabilities", None)
        if method is not None:
            probs = np.asarray(method(obs, legal, n_actions))
        else:
            probs = np.zeros(n_actions)
            probs[br_policy.act(obs, legal)] = 1.0
        row = tabular.policy_for_key(key)
        s = probs.sum()
        if s > 0:
            row[:] = probs / s
    for a in state.legal_actions():
        _fill_br_side(game, br_policy, target_player, tabular,
                      state.child(a), visited, n_actions)


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def _fig(rows: list[dict], figures_dir: Path) -> None:
    oracles: list[str] = []
    for r in rows:
        if r["oracle"] not in oracles:
            oracles.append(r["oracle"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for oracle_name, color in zip(oracles, ("tab:blue", "tab:orange")):
        grid: list[int] = []
        for r in rows:
            if r["oracle"] != oracle_name:
                continue
            n = int(r["n_episodes"])
            if n not in grid:
                grid.append(n)
        grid.sort()
        means, stds = [], []
        for n in grid:
            vals = [max(float(r["value_gap"]), 1e-6) for r in rows
                    if r["oracle"] == oracle_name and int(r["n_episodes"]) == n]
            means.append(np.mean(vals)); stds.append(np.std(vals))
            for v in vals:
                ax.scatter([n], [v], color=color, s=12, alpha=0.5)
        ax.errorbar(grid, means, yerr=stds, marker="o", color=color,
                    label=oracle_name, capsize=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training episodes per BR call")
    ax.set_ylabel("value gap  u(BR_exact,σ) − u(ABR,σ)")
    ax.set_title("BR quality vs training budget (shared reference target)")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"br_quality.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
