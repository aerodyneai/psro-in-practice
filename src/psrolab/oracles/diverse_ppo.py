"""PPO with a behavioral-diversity bonus (Ch. 11).

Implements a state-wise behavioral-diversity reward in the spirit of PSD-PSRO
(Yao et al. 2023, "Policy Space Diversity for Non-Transitive Games", NeurIPS):
the learner is paid extra for visiting states where its action distribution
differs from the *nearest* population policy's. PSD-PSRO defines diversity as
a distance in occupancy-weighted policy space; weighting by the learner's own
state visitation (which on-policy collection does for free) reduces it to the
per-visited-state bonus used here:

    r_t  +=  lambda * min_i TV( pi_new(.|s_t), pi_i(.|s_t) )

where TV is total-variation distance and i ranges over the player's existing
population. lambda = 0 recovers the plain PPO oracle exactly — Ch. 11's
experiment is precisely that ablation, on one transitive and one cyclic game
(diversity should only pay on the cyclic one).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from psrolab.games.base import Policy, Population, SimGame
from psrolab.oracles.base import MixtureOpponent
from psrolab.oracles.ppo import PPOOracle, _Rollout


class DiversePPOOracle(PPOOracle):
    """PPOOracle plus the behavioral-diversity term.

    Args:
        diversity_coef: bonus scale lambda. 0 disables the bonus (== PPOOracle).
        Remaining args: see PPOOracle.
    """

    def __init__(self, diversity_coef: float = 0.5, **ppo_kwargs) -> None:
        super().__init__(**ppo_kwargs)
        self.diversity_coef = diversity_coef

    def best_response(
        self,
        game: SimGame,
        player: int,
        population: Population,
        meta_strategies: Sequence[np.ndarray],
        rng: np.random.Generator,
    ) -> Policy:
        self._own_population = list(population.policies[player])
        try:
            return super().best_response(game, player, population, meta_strategies, rng)
        finally:
            self._own_population = []

    def _collect(
        self, game, raw, player: int, opponents: MixtureOpponent,
        net, n_episodes: int, rng: np.random.Generator,
    ) -> _Rollout:
        rollout = super()._collect(game, raw, player, opponents, net, n_episodes, rng)
        if self.diversity_coef == 0.0 or not self._own_population or not rollout.obs:
            return rollout

        n_actions = net.policy_head.out_features
        obs = torch.as_tensor(
            np.array(rollout.obs), dtype=torch.float32, device=self.device
        )
        masks = torch.as_tensor(np.array(rollout.legal_masks), device=self.device)
        with torch.no_grad():
            log_probs, _ = net(obs, masks)
        new_probs = log_probs.exp().cpu().numpy()

        for t in range(len(rollout.obs)):
            legal = [int(a) for a in np.flatnonzero(rollout.legal_masks[t])]
            distances = [
                _total_variation(
                    new_probs[t],
                    _action_probs(policy, rollout.obs[t], legal, n_actions),
                )
                for policy in self._own_population
            ]
            rollout.rewards[t] += self.diversity_coef * min(distances)
        return rollout


def _action_probs(
    policy: Policy, obs: np.ndarray, legal: list[int], n_actions: int
) -> np.ndarray:
    method = getattr(policy, "action_probabilities", None)
    if method is not None:
        return np.asarray(method(obs, legal, n_actions))
    probs = np.zeros(n_actions)
    probs[policy.act(obs, legal)] = 1.0
    return probs


def _total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())
