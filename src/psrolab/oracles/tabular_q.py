"""Tabular Q-learning best-response oracle (Ch. 6).

The book's first RL oracle, chosen for pedagogy: readers see the full
oracle-in-the-loop mechanic — train against the frozen opponent mixture, return
a greedy policy — with zero neural machinery. On Kuhn poker this oracle plus
the Nash meta-solver reaches full-game exploitability < 0.05 within 15 PSRO
iterations (pinned by tests/test_kuhn_tabular.py).

Works on any OpenSpielGame (duck-typed: needs `raw_game` and `observation`).
States are keyed by the observation tensor's bytes, so the same policy class
works for any game small enough to enumerate by experience.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from psrolab.games.base import Policy, Population, SimGame
from psrolab.oracles.base import MixtureOpponent, Oracle


@dataclass
class TabularQPolicy:
    """Greedy policy over learned Q-values, keyed by observation bytes.

    Deterministic: unseen states fall back to the first legal action, ties
    break toward the lowest action index.
    """

    q_values: dict[bytes, np.ndarray]

    def act(self, observation: np.ndarray, legal_actions: Sequence[int]) -> int:
        row = self.q_values.get(observation.tobytes())
        if row is None:
            return int(legal_actions[0])
        legal = list(legal_actions)
        return int(legal[int(np.argmax(row[legal]))])

    def action_probabilities(
        self, observation: np.ndarray, legal_actions: Sequence[int], n_actions: int
    ) -> np.ndarray:
        probs = np.zeros(n_actions)
        probs[self.act(observation, legal_actions)] = 1.0
        return probs


class TabularQOracle(Oracle):
    """Q-learning against the meta-strategy mixture opponent.

    Each episode samples one opponent policy from the meta-strategy
    (MixtureOpponent — the same object the PPO oracle trains against in
    Ch. 6), then plays with epsilon-greedy exploration, updating Q on the
    learner's own transitions only. Rewards are the terminal returns; poker
    games have no intermediate reward.

    Args:
        n_episodes: training episodes per best-response call.
        alpha: Q-learning step size.
        epsilon: (start, end) linear exploration schedule over the episodes.
        gamma: discount; 1.0 is correct for short episodic games.
    """

    def __init__(
        self,
        n_episodes: int = 20000,
        alpha: float = 0.1,
        epsilon: tuple[float, float] = (1.0, 0.05),
        gamma: float = 1.0,
    ) -> None:
        self.n_episodes = n_episodes
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

    def best_response(
        self,
        game: SimGame,
        player: int,
        population: Population,
        meta_strategies: Sequence[np.ndarray],
        rng: np.random.Generator,
    ) -> Policy:
        raw = game.raw_game  # type: ignore[attr-defined] — needs an OpenSpielGame
        n_actions = raw.num_distinct_actions()
        opponents = MixtureOpponent(population, meta_strategies, exclude_player=player)
        q_values: dict[bytes, np.ndarray] = {}
        eps_start, eps_end = self.epsilon

        for episode in range(self.n_episodes):
            eps = eps_start + (eps_end - eps_start) * episode / max(self.n_episodes - 1, 1)
            opponent = opponents.sample(rng)
            state = raw.new_initial_state()
            pending: tuple[np.ndarray, int] | None = None  # learner's last (q_row, action)

            while not state.is_terminal():
                if state.is_chance_node():
                    actions, probs = zip(*state.chance_outcomes())
                    state.apply_action(int(rng.choice(actions, p=probs)))
                elif state.current_player() != player:
                    current = state.current_player()
                    obs = game.observation(state, current)  # type: ignore[attr-defined]
                    state.apply_action(opponent[current].act(obs, state.legal_actions()))
                else:
                    obs = game.observation(state, player)  # type: ignore[attr-defined]
                    legal = state.legal_actions()
                    row = q_values.setdefault(obs.tobytes(), np.zeros(n_actions))
                    if pending is not None:
                        prev_row, prev_action = pending
                        target = self.gamma * row[legal].max()
                        prev_row[prev_action] += self.alpha * (target - prev_row[prev_action])
                    if rng.random() < eps:
                        action = int(rng.choice(legal))
                    else:
                        action = int(legal[int(np.argmax(row[legal]))])
                    pending = (row, action)
                    state.apply_action(action)

            if pending is not None:
                prev_row, prev_action = pending
                reward = state.returns()[player]
                prev_row[prev_action] += self.alpha * (reward - prev_row[prev_action])

        return TabularQPolicy(q_values=q_values)
