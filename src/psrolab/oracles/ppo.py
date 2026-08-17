"""Minimal single-file PPO best-response oracle (Ch. 6).

This file is printed in full in the book, so it optimizes for readability over
speed and over feature count: one actor-critic MLP, legal-action masking, GAE,
the clipped surrogate objective, and nothing else. No frameworks — the point
is that a complete neural best-response oracle for PSRO fits in ~300 lines.

Training loop: the opponent is frozen (a MixtureOpponent sampling from the
meta-strategy each episode), so from the learner's seat the game is a plain
MDP — chance nodes and opponent moves are just environment dynamics. PPO
collects only the learner's own transitions; the terminal return is the only
reward (poker games have no intermediate reward).

Determinism: all sampling flows from the numpy Generator handed in by
`run_psro`; torch is seeded from it per best-response call. Runs are
bit-reproducible on CPU. On CUDA, cuDNN/cuBLAS kernels may introduce minor
nondeterminism; the experiment scripts note this (CLAUDE.md rule 2).
Device is auto-detected; CUDA is never required.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from psrolab.games.base import Policy, Population, SimGame
from psrolab.oracles.base import MixtureOpponent, Oracle


class ActorCritic(nn.Module):
    """Shared-trunk MLP with a masked policy head and a value head."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(
        self, obs: torch.Tensor, legal_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (masked action log-probabilities, state value)."""
        h = self.trunk(obs)
        logits = self.policy_head(h).masked_fill(~legal_mask, -torch.inf)
        return torch.log_softmax(logits, dim=-1), self.value_head(h).squeeze(-1)


class PPOPolicy:
    """A trained actor as a frozen population policy.

    Stochastic: act() samples from the masked softmax with an internal rng
    (the Policy protocol has no rng argument), deterministic given the seed.
    `action_probabilities` exposes the exact distribution for the
    reach-weighted exploitability converter (eval/openspiel_exploitability).
    """

    def __init__(self, net: ActorCritic, seed: int) -> None:
        self._net = net.to("cpu").eval()  # inference is tiny; keep it simple
        self._rng = np.random.default_rng(seed)

    def _probs(self, observation: np.ndarray, legal_actions: Sequence[int]) -> np.ndarray:
        n_actions = self._net.policy_head.out_features
        mask = torch.zeros(n_actions, dtype=torch.bool)
        mask[list(legal_actions)] = True
        with torch.no_grad():
            log_probs, _ = self._net(
                torch.as_tensor(observation, dtype=torch.float32), mask
            )
        return log_probs.exp().numpy()

    def act(self, observation: np.ndarray, legal_actions: Sequence[int]) -> int:
        probs = self._probs(observation, legal_actions)
        return int(self._rng.choice(len(probs), p=probs))

    def action_probabilities(
        self, observation: np.ndarray, legal_actions: Sequence[int], n_actions: int
    ) -> np.ndarray:
        return self._probs(observation, legal_actions)


@dataclass
class _Rollout:
    """Flat storage for the learner's transitions across a batch of episodes."""

    obs: list[np.ndarray] = field(default_factory=list)
    legal_masks: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    episode_ends: list[bool] = field(default_factory=list)


class PPOOracle(Oracle):
    """PPO trained against the meta-strategy mixture opponent.

    Args:
        total_episodes: training episodes per best-response call.
        episodes_per_update: rollout batch size between gradient phases.
        lr, clip_eps, entropy_coef, value_coef, epochs, minibatch_size:
            standard PPO knobs; defaults are sane for Kuhn/Leduc.
        gamma, gae_lambda: discounting/GAE. gamma=1.0 suits short episodes.
        hidden: MLP width.
        device: "auto" picks CUDA when available, else CPU.
    """

    def __init__(
        self,
        total_episodes: int = 30000,
        episodes_per_update: int = 1000,
        lr: float = 2.5e-4,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        epochs: int = 4,
        minibatch_size: int = 256,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        hidden: int = 64,
        device: str = "auto",
        warm_start: bool = False,
    ) -> None:
        self.total_episodes = total_episodes
        self.episodes_per_update = episodes_per_update
        self.lr = lr
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.hidden = hidden
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )
        # When warm_start=True, we cache each player's most recent BR net
        # state_dict and re-load it at the start of the *next* call for that
        # player. Ch. 12's §12.4 ablation quantifies how much this saves in
        # training episodes vs how much it costs in population diversity.
        self.warm_start = warm_start
        self._last_state_dicts: dict[int, dict] = {}

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
        obs_dim = game.obs_dim  # type: ignore[attr-defined]
        seed = int(rng.integers(2**31))
        torch.manual_seed(seed)

        net = ActorCritic(obs_dim, n_actions, self.hidden).to(self.device)
        if self.warm_start and player in self._last_state_dicts:
            net.load_state_dict(self._last_state_dicts[player])
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        opponents = MixtureOpponent(population, meta_strategies, exclude_player=player)

        episodes_done = 0
        while episodes_done < self.total_episodes:
            batch = min(self.episodes_per_update, self.total_episodes - episodes_done)
            rollout = self._collect(game, raw, player, opponents, net, batch, rng)
            self._update(net, optimizer, rollout)
            episodes_done += batch

        if self.warm_start:
            self._last_state_dicts[player] = {
                k: v.detach().clone() for k, v in net.state_dict().items()
            }
        return PPOPolicy(net, seed=seed + 1)

    def _collect(
        self, game, raw, player: int, opponents: MixtureOpponent,
        net: ActorCritic, n_episodes: int, rng: np.random.Generator,
    ) -> _Rollout:
        rollout = _Rollout()
        net.eval()
        for _ in range(n_episodes):
            opponent = opponents.sample(rng)
            state = raw.new_initial_state()
            steps_this_episode = 0
            while not state.is_terminal():
                if state.is_chance_node():
                    actions, probs = zip(*state.chance_outcomes())
                    state.apply_action(int(rng.choice(actions, p=probs)))
                elif state.current_player() != player:
                    current = state.current_player()
                    obs = game.observation(state, current)
                    state.apply_action(opponent[current].act(obs, state.legal_actions()))
                else:
                    obs = game.observation(state, player)
                    mask = np.zeros(net.policy_head.out_features, dtype=bool)
                    mask[state.legal_actions()] = True
                    with torch.no_grad():
                        log_probs, value = net(
                            torch.as_tensor(obs, dtype=torch.float32, device=self.device),
                            torch.as_tensor(mask, device=self.device),
                        )
                    probs = log_probs.exp().cpu().numpy()
                    action = int(rng.choice(len(probs), p=probs))
                    rollout.obs.append(obs)
                    rollout.legal_masks.append(mask)
                    rollout.actions.append(action)
                    rollout.log_probs.append(float(log_probs[action]))
                    rollout.values.append(float(value))
                    rollout.rewards.append(0.0)
                    rollout.episode_ends.append(False)
                    steps_this_episode += 1
                    state.apply_action(action)
            if steps_this_episode:
                rollout.rewards[-1] = float(state.returns()[player])
                rollout.episode_ends[-1] = True
        return rollout

    def _update(self, net: ActorCritic, optimizer, rollout: _Rollout) -> None:
        n = len(rollout.actions)
        if n == 0:
            return
        advantages = np.zeros(n)
        returns = np.zeros(n)
        gae, next_value = 0.0, 0.0
        for t in range(n - 1, -1, -1):  # GAE, resetting across episode boundaries
            if rollout.episode_ends[t]:
                gae, next_value = 0.0, 0.0
            delta = rollout.rewards[t] + self.gamma * next_value - rollout.values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
            returns[t] = gae + rollout.values[t]
            next_value = rollout.values[t]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        device = self.device
        obs = torch.as_tensor(np.array(rollout.obs), dtype=torch.float32, device=device)
        masks = torch.as_tensor(np.array(rollout.legal_masks), device=device)
        actions = torch.as_tensor(rollout.actions, device=device)
        old_log_probs = torch.as_tensor(rollout.log_probs, dtype=torch.float32, device=device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=device)

        net.train()
        for _ in range(self.epochs):
            for start in range(0, n, self.minibatch_size):
                mb = slice(start, min(start + self.minibatch_size, n))
                log_probs_all, values = net(obs[mb], masks[mb])
                log_probs = log_probs_all.gather(1, actions[mb].unsqueeze(1)).squeeze(1)
                ratio = (log_probs - old_log_probs[mb]).exp()
                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                policy_loss = -torch.min(ratio * adv[mb], clipped * adv[mb]).mean()
                value_loss = (values - ret[mb]).pow(2).mean()
                entropy = -(log_probs_all.exp() * log_probs_all.nan_to_num(neginf=0.0)
                            ).sum(-1).mean()
                loss = (policy_loss + self.value_coef * value_loss
                        - self.entropy_coef * entropy)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
