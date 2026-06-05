# MorphQuad — a self-learning morphing 5-inch drone

A GPU-accelerated reinforcement-learning project where a **5-inch quadcopter
learns to fold its arms in flight (X ↔ H ↔ folded) to thread through tight
gaps.** The arms aren't decoration: folding them physically shrinks the
drone's lateral span, its roll authority, and its inertia in the simulator,
so the agent has to learn *when* to morph in order to fit through a slot that
is narrower than its spread-out body.

Everything — the physics and the learning — runs as batched tensors, so the
whole thing lives on your GPU and a few thousand simulated drones train in
parallel.

```
   spread (X)            folding              folded (fits slot)
   \   /                  \  /                   ||
    \ /         ───►        \/          ───►      ||      ───►  through!
    / \                    /  \                   ||
   /   \                  /    \                  ||
```

## What's inside

| File | Role |
|------|------|
| `morphquad/config.py`   | All tunables: drone geometry, the gap task, PPO. |
| `morphquad/quatutil.py` | Batched quaternion / rigid-body math (pure torch). |
| `morphquad/env.py`      | Vectorized morphing-quad simulator (thousands of envs on GPU). |
| `morphquad/model.py`    | Actor-critic network. |
| `morphquad/ppo.py`      | PPO trainer — mixed precision, GPU-saturating. |
| `morphquad/train.py`    | Training CLI. |
| `morphquad/viz.py`      | 3D rollout viewer: drone + folding arms + slotted wall → GIF. |
| `scripts/smoke_test.py` | Fast CPU sanity check of the whole pipeline. |

## The physics, briefly

Standard rigid-body quadrotor, with one twist: the lateral rotor positions
scale with `cos(phi)`, where `phi` is the arm-fold angle. So folding:

- **shrinks the lateral half-span** → the drone can fit a narrow slot,
- **shrinks the roll torque arm** → it loses roll authority while folded,
- **shrinks `Ixx` / `Izz`** → its rotational response changes.

The agent's action is **4 motor thrusts + 1 fold rate**. The task: fly from a
spawn point, through a wall slot narrower than its unfolded span, and out the
other side — which is only possible if it folds at the right moment and keeps
control with the reduced roll authority.

## Install

You need PyTorch. For your **NVIDIA GPU**, install the CUDA build that matches
your driver (see https://pytorch.org/get-started/locally/), e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

> CPU also works (for the smoke test / tiny runs) but real training wants a GPU.

## Quick sanity check (seconds, CPU is fine)

```bash
python scripts/smoke_test.py
```

This runs the sim, a few PPO updates, and renders a short GIF using a built-in
scripted controller — so you can see the morphing-through-the-gap behavior
*before* training anything.

## Train (this is what loads your GPU)

```bash
# auto-detect GPU and use good defaults
python -m morphquad.train

# crank parallel sims to fully saturate a larger GPU
python -m morphquad.train --envs 16384 --total-steps 80000000
```

**Use your GPU fully:** the single biggest knob is `--envs` — more parallel
simulations = more work per batch = higher GPU utilization. Watch
`nvidia-smi`; raise `--envs` (4096 → 8192 → 16384 → 32768) until you fill VRAM
or utilization pins near 100%. Mixed precision (tensor cores) is on by default
on CUDA; disable with `--no-amp`. Checkpoints land in `checkpoints/`.

Training prints a live line per update:

```
upd   120 | step   15,728,640 |  640,000 sps | ret  18.42 | len 142.3 | pass 61.0% | crash 22.1% | pi -0.012 v 0.318 H 1.84
```

`pass` is the fraction of episodes that cleanly folded through the slot — that's
the number you want climbing.

## Watch it fly (3D)

```bash
# render a trained policy to a GIF
python -m morphquad.viz --ckpt checkpoints/morphquad_final.pt --out flight.gif

# or an interactive, rotatable window
python -m morphquad.viz --ckpt checkpoints/morphquad_final.pt --show

# no checkpoint yet? the scripted controller still demos the behavior:
python -m morphquad.viz --policy scripted --out flight.gif
```

The viewer draws the body, the four arms (which visibly fold as `phi`
increases), the props, the slotted wall, and the flight path.

## Tuning the challenge

In `morphquad/config.py`:

- `TaskConfig.gap_width` — narrow it to force more aggressive folding.
- `DroneConfig.phi_max`, `fold_rate_max` — how far / how fast the arms move.
- `RewardConfig` — balance progress vs. crash vs. control effort.
- `PPOConfig` — `num_envs`, `rollout_len`, `learning_rate`, etc.

## Notes

- "Use my GPU fully" happens on **your** machine — this trains locally against
  your CUDA device. A cloud container without a GPU will fall back to CPU.
- The simulator is a faithful-but-simplified quad model (diagonal inertia,
  first-order motor lag, lumped-mass arms). It's built for fast RL, not for
  certification-grade aerodynamics.
