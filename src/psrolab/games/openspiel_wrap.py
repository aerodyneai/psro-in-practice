"""OpenSpiel games behind the SimGame interface (Ch. 6-7).

Adapts any 2-player, sequential, zero-sum OpenSpiel game (Kuhn, Leduc, ...)
to the black-box `SimGame` contract the PSRO loop expects. Policies receive the
information-state tensor as observation plus the legal-action list, exactly as
in the Policy protocol — RL oracles never touch pyspiel directly except through
the two escape hatches this class deliberately exposes for step-level training:

  `raw_game`               the underlying pyspiel.Game
  `observation(state, p)`  the observation tensor policies will see

NOT imported from psrolab.games.__init__: open_spiel is an optional dependency
(`pip install -e ".[rl]"`), and the base test suite must run without it.
Import as `from psrolab.games.openspiel_wrap import OpenSpielGame`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from psrolab.games.base import Policy, SimGame


class UniformRandomPolicy:
    """Uniform over legal actions. The iteration-0 population for every game.

    Carries its own rng (Policy.act has no rng argument — the protocol is
    frozen); deterministic given the construction seed and call sequence.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray, legal_actions: Sequence[int]) -> int:
        return int(self._rng.choice(legal_actions))

    def action_probabilities(
        self, observation: np.ndarray, legal_actions: Sequence[int], n_actions: int
    ) -> np.ndarray:
        """Exact per-action distribution — used by the exploitability converter."""
        probs = np.zeros(n_actions)
        probs[list(legal_actions)] = 1.0 / len(legal_actions)
        return probs


class OpenSpielGame(SimGame):
    """A 2-player sequential OpenSpiel game as a SimGame.

    Args:
        game_name: pyspiel short name, e.g. "kuhn_poker", "leduc_poker".
        seed: seeds the initial (uniform-random) policies only; episode
            randomness comes from the rng passed to `sample_returns`.
    """

    def __init__(self, game_name: str, seed: int = 0) -> None:
        import pyspiel

        self.raw_game = pyspiel.load_game(game_name)
        game_type = self.raw_game.get_type()
        assert game_type.dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL, (
            "OpenSpielGame supports sequential games only"
        )
        self.n_players = self.raw_game.num_players()
        assert self.n_players == 2, "psrolab is 2-player for now"
        self.game_name = game_name
        self._seed = seed
        self._use_info_state = (
            game_type.provides_information_state_tensor
        )
        self.obs_dim = (
            self.raw_game.information_state_tensor_size()
            if self._use_info_state
            else self.raw_game.observation_tensor_size()
        )

    def observation(self, state, player: int) -> np.ndarray:
        """The observation tensor a Policy sees at `state` for `player`."""
        if self._use_info_state:
            return np.asarray(state.information_state_tensor(player), dtype=np.float64)
        return np.asarray(state.observation_tensor(player), dtype=np.float64)

    def play_episode(
        self, policies: Sequence[Policy], rng: np.random.Generator
    ) -> np.ndarray:
        """Play one episode; return per-player returns, shape (n_players,)."""
        state = self.raw_game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                state.apply_action(int(rng.choice(actions, p=probs)))
            else:
                player = state.current_player()
                obs = self.observation(state, player)
                action = policies[player].act(obs, state.legal_actions())
                state.apply_action(action)
        return np.asarray(state.returns())

    def sample_returns(
        self, policies: Sequence[Policy], n_episodes: int, rng: np.random.Generator
    ) -> np.ndarray:
        total = np.zeros(self.n_players)
        for _ in range(n_episodes):
            total += self.play_episode(policies, rng)
        return total / n_episodes

    def initial_policies(self) -> list[Policy]:
        return [UniformRandomPolicy(seed=self._seed + p) for p in range(self.n_players)]
