"""Actor-critic network for PPO (continuous actions)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.act_dim = act_dim

        def trunk():
            layers, last = [], obs_dim
            for h in hidden:
                layers += [_layer_init(nn.Linear(last, h)), nn.Tanh()]
                last = h
            return nn.Sequential(*layers), last

        self.actor_body, last_a = trunk()
        self.critic_body, last_c = trunk()
        self.mean = _layer_init(nn.Linear(last_a, act_dim), std=0.01)
        self.value = _layer_init(nn.Linear(last_c, 1), std=1.0)
        # state-independent log std (smaller -> more precise actions)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.9))

    def _dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mean(self.actor_body(obs))
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        dist = self._dist(obs)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        value = self.value(self.critic_body(obs)).squeeze(-1)
        return action, logp, value

    @torch.no_grad()
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value(self.critic_body(obs)).squeeze(-1)

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        dist = self._dist(obs)
        logp = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.value(self.critic_body(obs)).squeeze(-1)
        return logp, entropy, value

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean(self.actor_body(obs))
