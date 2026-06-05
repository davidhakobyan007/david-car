"""PPO trainer, vectorized and mixed-precision, designed to keep the GPU busy.

The whole rollout buffer lives on the GPU; the simulator never leaves the
device, so for large ``num_envs`` the GPU stays saturated between updates.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import Config
from .env import MorphQuadEnv, OBS_DIM, ACT_DIM
from .model import ActorCritic


@dataclass
class TrainStats:
    update: int
    global_step: int
    sps: int
    ep_return: float
    ep_len: float
    pass_rate: float
    crash_rate: float
    policy_loss: float
    value_loss: float
    entropy: float


class PPOTrainer:
    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        p = cfg.ppo
        torch.manual_seed(p.seed)

        self.env = MorphQuadEnv(cfg, p.num_envs, device)
        self.model = ActorCritic(OBS_DIM, ACT_DIM, p.hidden_sizes).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=p.learning_rate, eps=1e-5)

        self.use_amp = bool(p.amp) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        n, T = p.num_envs, p.rollout_len
        dev = device
        self.obs_buf = torch.zeros(T, n, OBS_DIM, device=dev)
        self.act_buf = torch.zeros(T, n, ACT_DIM, device=dev)
        self.logp_buf = torch.zeros(T, n, device=dev)
        self.rew_buf = torch.zeros(T, n, device=dev)
        self.val_buf = torch.zeros(T, n, device=dev)
        self.done_buf = torch.zeros(T, n, device=dev)   # for-GAE done (terminal|trunc)

        self.batch_size = n * T
        self.minibatch_size = self.batch_size // p.num_minibatches

        # running episode trackers
        self._ep_ret = torch.zeros(n, device=dev)
        self._ep_len = torch.zeros(n, device=dev)
        self._ret_hist: list[float] = []
        self._len_hist: list[float] = []
        self._pass_hist: list[float] = []
        self._crash_hist: list[float] = []

        self.next_obs = self.env.obs()

    # --------------------------------------------------------------- rollout
    def _collect(self):
        p = self.cfg.ppo
        gamma = p.gamma
        for step in range(p.rollout_len):
            obs = self.next_obs
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                action, logp, value = self.model.act(obs)

            next_obs, rew, terminal, trunc, term_obs, info = self.env.step(action.float())

            # bootstrap truncated episodes: add gamma * V(terminal_obs)
            if trunc.any():
                with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    tv = self.model.get_value(term_obs)
                rew = rew + gamma * tv * trunc.float()

            done = (terminal | trunc).float()

            self.obs_buf[step] = obs
            self.act_buf[step] = action
            self.logp_buf[step] = logp
            self.val_buf[step] = value.float()
            self.rew_buf[step] = rew
            self.done_buf[step] = done

            # episode bookkeeping
            self._ep_ret += rew
            self._ep_len += 1
            fin = (terminal | trunc)
            if fin.any():
                fi = fin
                self._ret_hist.extend(self._ep_ret[fi].tolist())
                self._len_hist.extend(self._ep_len[fi].tolist())
                self._pass_hist.extend(info["passed"][fi].float().tolist())
                self._crash_hist.extend(info["crash"][fi].float().tolist())
                self._ep_ret[fi] = 0.0
                self._ep_len[fi] = 0.0

            self.next_obs = next_obs

        # GAE
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            last_val = self.model.get_value(self.next_obs).float()
        adv = torch.zeros_like(self.rew_buf)
        lastgae = torch.zeros(p.num_envs, device=self.device)
        for t in reversed(range(p.rollout_len)):
            nextnonterminal = 1.0 - self.done_buf[t]
            nextvalue = last_val if t == p.rollout_len - 1 else self.val_buf[t + 1]
            delta = self.rew_buf[t] + gamma * nextvalue * nextnonterminal - self.val_buf[t]
            lastgae = delta + gamma * p.gae_lambda * nextnonterminal * lastgae
            adv[t] = lastgae
        returns = adv + self.val_buf
        return adv, returns

    # ---------------------------------------------------------------- update
    def _update(self, adv, returns):
        p = self.cfg.ppo
        b_obs = self.obs_buf.reshape(-1, OBS_DIM)
        b_act = self.act_buf.reshape(-1, ACT_DIM)
        b_logp = self.logp_buf.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = self.val_buf.reshape(-1)

        idx = torch.randperm(self.batch_size, device=self.device)
        pl = vl = ent = 0.0
        nupd = 0
        for _ in range(p.epochs):
            for start in range(0, self.batch_size, self.minibatch_size):
                mb = idx[start:start + self.minibatch_size]
                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    newlogp, entropy, newval = self.model.evaluate(b_obs[mb], b_act[mb])
                    ratio = torch.exp(newlogp - b_logp[mb])
                    p1 = -mb_adv * ratio
                    p2 = -mb_adv * torch.clamp(ratio, 1 - p.clip_coef, 1 + p.clip_coef)
                    policy_loss = torch.max(p1, p2).mean()

                    # clipped value loss
                    v_unclip = (newval - b_ret[mb]) ** 2
                    v_clip = b_val[mb] + torch.clamp(
                        newval - b_val[mb], -p.clip_coef, p.clip_coef
                    )
                    v_clip = (v_clip - b_ret[mb]) ** 2
                    value_loss = 0.5 * torch.max(v_unclip, v_clip).mean()

                    entropy_loss = entropy.mean()
                    loss = policy_loss - p.ent_coef * entropy_loss + p.vf_coef * value_loss

                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), p.max_grad_norm)
                self.scaler.step(self.opt)
                self.scaler.update()

                pl += policy_loss.item(); vl += value_loss.item()
                ent += entropy_loss.item(); nupd += 1
        return pl / nupd, vl / nupd, ent / nupd

    # ----------------------------------------------------------------- loop
    def train(self, on_stats=None):
        p = self.cfg.ppo
        steps_per_update = p.num_envs * p.rollout_len
        num_updates = max(1, p.total_steps // steps_per_update)
        global_step = 0
        start = time.time()

        for update in range(1, num_updates + 1):
            if p.anneal_lr:
                frac = 1.0 - (update - 1) / num_updates
                for g in self.opt.param_groups:
                    g["lr"] = frac * p.learning_rate

            adv, returns = self._collect()
            pl, vl, ent = self._update(adv, returns)
            global_step += steps_per_update

            sps = int(global_step / (time.time() - start))
            stats = TrainStats(
                update=update, global_step=global_step, sps=sps,
                ep_return=_mean(self._ret_hist), ep_len=_mean(self._len_hist),
                pass_rate=_mean(self._pass_hist), crash_rate=_mean(self._crash_hist),
                policy_loss=pl, value_loss=vl, entropy=ent,
            )
            self._ret_hist.clear(); self._len_hist.clear()
            self._pass_hist.clear(); self._crash_hist.clear()
            if on_stats is not None:
                on_stats(stats)
        return self.model


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0
