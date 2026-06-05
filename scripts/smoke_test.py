"""Tiny end-to-end sanity check (CPU friendly, a few seconds).

Verifies shapes, that the sim is finite/stable, that a couple of PPO updates
run, and that a short rollout + GIF render works. Run:

    pip install -r requirements.txt
    python scripts/smoke_test.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morphquad.config import default_config
from morphquad.env import MorphQuadEnv, OBS_DIM, ACT_DIM
from morphquad.ppo import PPOTrainer
from morphquad import viz


def check_env():
    cfg = default_config()
    dev = torch.device("cpu")
    env = MorphQuadEnv(cfg, num_envs=8, device=dev)
    obs = env.obs()
    assert obs.shape == (8, OBS_DIM), obs.shape
    for _ in range(50):
        a = torch.randn(8, ACT_DIM)
        obs, rew, term, trunc, tobs, info = env.step(a)
        assert obs.shape == (8, OBS_DIM)
        assert rew.shape == (8,)
        assert torch.isfinite(obs).all(), "non-finite obs"
        assert torch.isfinite(rew).all(), "non-finite reward"
    print("[ok] env: shapes + finite over 50 steps")


def check_train():
    cfg = default_config()
    cfg.ppo.num_envs = 64
    cfg.ppo.rollout_len = 16
    cfg.ppo.total_steps = 64 * 16 * 3   # 3 updates
    cfg.ppo.amp = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = PPOTrainer(cfg, dev)
    seen = []
    trainer.train(on_stats=lambda s: seen.append(s))
    assert len(seen) >= 1
    last = seen[-1]
    assert last.value_loss == last.value_loss  # not NaN
    print(f"[ok] train: {len(seen)} updates on {dev}, last ret={last.ep_return:.2f}")
    return trainer


def check_viz():
    cfg = default_config()
    dev = torch.device("cpu")
    cfg, traj = viz.rollout(cfg, dev, policy="scripted", max_steps=120)
    assert traj["pos"].shape[1] == 3
    out = "smoke_flight.gif"
    viz.animate(cfg, traj, out=out, show=False, fps=20)
    assert os.path.exists(out)
    print(f"[ok] viz: rendered {out} ({len(traj['pos'])} frames, "
          f"passed={traj['passed']})")


if __name__ == "__main__":
    print("device cuda available:", torch.cuda.is_available())
    check_env()
    check_train()
    check_viz()
    print("\nALL SMOKE CHECKS PASSED")
