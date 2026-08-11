"""2D pursuit-evasion: the book's capstone domain (Ch. 13).

One pursuer, one evader in the unit square. Discrete 8-way + stay actions,
simultaneous moves, zero-sum terminal reward: +1 to the pursuer on capture
(distance < capture_radius), +1 to the evader on surviving `max_steps`.

The default parameters are deliberately balanced for a MIXED equilibrium:
equal speeds and a capture radius smaller than one step mean the pursuer
cannot win on pace — capture requires predicting the evader's dodge, and a
predictable evader (e.g. deterministic fleeing, which corners itself against
the walls) is punished. Scripted probes at these defaults: a greedy chaser
captures a random walker 99%, a deterministic fleer 62%, and a wall-aware
juking evader only ~41% — the gap between those last two is exactly the
mixing incentive PSRO is there to exploit. (The first draft of this game gave
the pursuer a 22% speed edge and a fat capture radius; the equilibrium was
pure pursuer-wins and every PSRO figure was degenerate. Ch. 13 tells that
story — domain design is part of the method.) Pure numpy physics, no external
simulator, deterministic given the rng.

Interface notes:
  * Simultaneous moves are presented TURN-BASED (player 0 commits, then
    player 1, then physics steps). Observations are built from the last
    resolved physics state, so player 1 never sees player 0's pending move —
    information-equivalent to simultaneous play, and it lets the Ch. 6 PPO /
    tabular-Q oracles drive this game unchanged (they speak the same tiny
    state protocol as the OpenSpiel wrapper: is_terminal / is_chance_node /
    chance_outcomes / current_player / legal_actions / apply_action / returns,
    plus game.observation and game.obs_dim).
  * A root chance node picks one of `n_spawns` evader spawn points, so
    policies must generalize across starts and pure strategies stay
    exploitable.

Observation (10-dim, per player): own xy, opponent xy, relative xy, distance,
fraction of time remaining, own and opponent distance to the nearest wall.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from psrolab.games.base import Policy, SimGame

ACTIONS = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]],
    dtype=float,
)
_DIAG = np.sqrt(2.0)
ACTIONS[[2, 4, 6, 8]] /= _DIAG  # unit-length diagonals
N_ACTIONS = len(ACTIONS)


class PursuitEvasionSim(SimGame):
    """SimGame wrapper (player 0 = pursuer, player 1 = evader).

    Args:
        pursuer_speed / evader_speed: per-step displacement.
        capture_radius: capture when distance drops below this.
        max_steps: episode cap; evader wins on reaching it.
        n_spawns: evader spawn points on a circle around the arena center
            (root chance node). Pursuer always starts at (0.1, 0.1).
        seed: seeds the initial uniform-random policies only.
    """

    def __init__(
        self,
        pursuer_speed: float = 0.05,
        evader_speed: float = 0.05,
        capture_radius: float = 0.04,
        max_steps: int = 30,
        n_spawns: int = 6,
        seed: int = 0,
    ) -> None:
        self.n_players = 2
        self.pursuer_speed = pursuer_speed
        self.evader_speed = evader_speed
        self.capture_radius = capture_radius
        self.max_steps = max_steps
        self.n_spawns = n_spawns
        self._seed = seed
        self.obs_dim = 10

    # -- pyspiel-like game surface (what the RL oracles use) ----------------
    @property
    def raw_game(self) -> PursuitEvasionSim:
        return self

    def new_initial_state(self) -> _State:
        return _State(self)

    def num_distinct_actions(self) -> int:
        return N_ACTIONS

    def observation(self, state: _State, player: int) -> np.ndarray:
        return state.observation(player)

    # -- SimGame interface --------------------------------------------------
    def sample_returns(
        self, policies: Sequence[Policy], n_episodes: int, rng: np.random.Generator
    ) -> np.ndarray:
        total = np.zeros(2)
        for _ in range(n_episodes):
            total += self.play_episode(policies, rng)
        return total / n_episodes

    def initial_policies(self) -> list[Policy]:
        return [RandomWalkPolicy(seed=self._seed + p) for p in range(2)]

    def play_episode(
        self,
        policies: Sequence[Policy],
        rng: np.random.Generator,
        record: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Play one episode; optionally also return positions (T+1, 2, 2)."""
        state = self.new_initial_state()
        trajectory = []
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                state.apply_action(int(rng.choice(outcomes, p=probs)))
                if record:
                    trajectory.append(state.positions.copy())
                continue
            player = state.current_player()
            action = policies[player].act(state.observation(player), state.legal_actions())
            state.apply_action(action)
            if record and player == 1:
                trajectory.append(state.positions.copy())
        returns = state.returns()
        if record:
            return returns, np.array(trajectory)
        return returns

    def spawn_position(self, index: int) -> np.ndarray:
        angle = 2.0 * np.pi * index / self.n_spawns
        return np.array([0.5, 0.5]) + 0.35 * np.array([np.cos(angle), np.sin(angle)])


class _State:
    """Turn-based view of the simultaneous game. See module docstring."""

    def __init__(self, game: PursuitEvasionSim) -> None:
        self.game = game
        self.positions = np.array([[0.1, 0.1], [0.5, 0.5]])  # evader set by chance
        self.step_count = 0
        self._spawned = False
        self._pending: int | None = None  # pursuer's committed action
        self._captured = False

    def is_chance_node(self) -> bool:
        return not self._spawned

    def chance_outcomes(self) -> list[tuple[int, float]]:
        n = self.game.n_spawns
        return [(i, 1.0 / n) for i in range(n)]

    def is_terminal(self) -> bool:
        return self._spawned and (
            self._captured or self.step_count >= self.game.max_steps
        )

    def current_player(self) -> int:
        return 0 if self._pending is None else 1

    def legal_actions(self) -> list[int]:
        return list(range(N_ACTIONS))

    def apply_action(self, action: int) -> None:
        if not self._spawned:
            self.positions[1] = self.game.spawn_position(action)
            self._spawned = True
            return
        if self._pending is None:
            self._pending = action
            return
        speeds = (self.game.pursuer_speed, self.game.evader_speed)
        for player, act in ((0, self._pending), (1, action)):
            self.positions[player] = np.clip(
                self.positions[player] + speeds[player] * ACTIONS[act], 0.0, 1.0
            )
        self._pending = None
        self.step_count += 1
        distance = float(np.linalg.norm(self.positions[0] - self.positions[1]))
        if distance < self.game.capture_radius:
            self._captured = True

    def returns(self) -> np.ndarray:
        if not self.is_terminal():
            return np.zeros(2)
        return np.array([1.0, -1.0]) if self._captured else np.array([-1.0, 1.0])

    def observation(self, player: int) -> np.ndarray:
        own, opp = self.positions[player], self.positions[1 - player]
        rel = opp - own
        return np.array(
            [
                *own,
                *opp,
                *rel,
                np.linalg.norm(rel),
                1.0 - self.step_count / self.game.max_steps,
                _wall_distance(own),
                _wall_distance(opp),
            ]
        )


class RandomWalkPolicy:
    """Uniform over the 9 actions; the iteration-0 population."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray, legal_actions: Sequence[int]) -> int:
        return int(self._rng.choice(legal_actions))

    def action_probabilities(
        self, observation: np.ndarray, legal_actions: Sequence[int], n_actions: int
    ) -> np.ndarray:
        probs = np.zeros(n_actions)
        probs[list(legal_actions)] = 1.0 / len(legal_actions)
        return probs


def _wall_distance(pos: np.ndarray) -> float:
    return float(min(pos[0], 1.0 - pos[0], pos[1], 1.0 - pos[1]))
