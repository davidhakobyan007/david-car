"""Train the morphing quad to fold through gaps.

Examples
--------
    # auto-detect GPU, sensible defaults
    python -m morphquad.train

    # push more parallel envs to fully load a big GPU
    python -m morphquad.train --envs 16384 --total-steps 80000000

    # quick smoke run (works on CPU in this container)
    python -m morphquad.train --envs 64 --total-steps 20000 --device cpu
"""
from __future__ import annotations

import argparse
import os

import torch

from .config import default_config
from .device import pick_device, tune_for_throughput, describe
from .ppo import PPOTrainer, TrainStats


def parse_args():
    p = argparse.ArgumentParser(description="Train MorphQuad with GPU PPO.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--envs", type=int, default=None, help="parallel sims")
    p.add_argument("--rollout", type=int, default=None)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default="checkpoints", help="checkpoint dir")
    p.add_argument("--save-every", type=int, default=25, help="updates between saves")
    p.add_argument("--log-every", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = default_config()
    if args.envs is not None: cfg.ppo.num_envs = args.envs
    if args.rollout is not None: cfg.ppo.rollout_len = args.rollout
    if args.total_steps is not None: cfg.ppo.total_steps = args.total_steps
    if args.lr is not None: cfg.ppo.learning_rate = args.lr
    if args.seed is not None: cfg.ppo.seed = args.seed
    if args.no_amp: cfg.ppo.amp = False

    device = pick_device(args.device)
    tune_for_throughput(device)
    print("=" * 64)
    print(" MorphQuad training")
    print(" ", describe(device))
    print(f"  envs={cfg.ppo.num_envs}  rollout={cfg.ppo.rollout_len}  "
          f"amp={cfg.ppo.amp and device.type == 'cuda'}")
    print(f"  steps/update={cfg.ppo.num_envs * cfg.ppo.rollout_len:,}  "
          f"total={cfg.ppo.total_steps:,}")
    print("=" * 64)

    os.makedirs(args.out, exist_ok=True)
    trainer = PPOTrainer(cfg, device)

    def on_stats(s: TrainStats):
        if s.update % args.log_every == 0:
            print(f"upd {s.update:>5} | step {s.global_step:>12,} | {s.sps:>7,} sps "
                  f"| ret {s.ep_return:7.2f} | len {s.ep_len:6.1f} "
                  f"| pass {s.pass_rate:5.1%} | crash {s.crash_rate:5.1%} "
                  f"| pi {s.policy_loss:+.3f} v {s.value_loss:.3f} H {s.entropy:.3f}")
        if s.update % args.save_every == 0:
            path = os.path.join(args.out, "morphquad_latest.pt")
            torch.save(
                {"model": trainer.model.state_dict(), "cfg": cfg.to_dict(),
                 "update": s.update, "global_step": s.global_step},
                path,
            )

    try:
        trainer.train(on_stats=on_stats)
    except KeyboardInterrupt:
        print("\n[interrupted] saving final checkpoint...")
    finally:
        path = os.path.join(args.out, "morphquad_final.pt")
        torch.save({"model": trainer.model.state_dict(), "cfg": cfg.to_dict()}, path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
