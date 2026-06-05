"""Render a 3D rollout of the morphing quad: body, folding arms, props, the
slotted wall, and the flight path. Saves a GIF (works headless) and can also
show an interactive window.

Usage
-----
    # use a trained policy
    python -m morphquad.viz --ckpt checkpoints/morphquad_final.pt --out flight.gif

    # no checkpoint yet? a simple scripted controller still demonstrates the
    # fold-through-the-gap behavior so you can see the 3D viewer working:
    python -m morphquad.viz --policy scripted --out flight.gif
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from .config import default_config, Config
from .device import pick_device
from .env import MorphQuadEnv, OBS_DIM, ACT_DIM
from .model import ActorCritic
from . import quatutil as Q


def _load_cfg_from_ckpt(ckpt: dict) -> Config:
    cfg = default_config()
    # checkpoints store a plain dict; defaults are fine for geometry/task here.
    return cfg


def scripted_action(env: MorphQuadEnv) -> torch.Tensor:
    """A small cascaded controller (position -> attitude -> motor mix).

    Stable enough to actually hover, fly toward the slot, fold its arms while
    crossing, and continue out the far side -- so the 3D viewer demonstrates the
    morphing-through-the-gap behavior before any RL training has happened.
    """
    cfg = env.cfg
    d = cfg.drone
    n, dev = env.n, env.device
    g = cfg.sim.gravity
    pos, vel, quat, omega = env.pos, env.vel, env.quat, env.omega
    gap = env.gap

    R = Q.quat_to_rotmat(quat)                 # (n,3,3) body->world
    bz = R[:, :, 2]                            # current body-z in world

    # ---- staged target: line up + fold in front of the slot, then punch
    #      through only once aligned and folded enough to fit ----
    dx = gap[0] - pos[:, 0]
    err_y0 = (pos[:, 1] - gap[1]).abs()
    err_z0 = (pos[:, 2] - gap[2]).abs()
    folded = env.phi.squeeze(-1) > 1.0
    ready = folded & (err_y0 < 0.06) & (err_z0 < 0.12)
    tx = torch.where(ready, torch.full((n,), gap[0].item() + 0.8, device=dev),
                     torch.full((n,), gap[0].item() - 0.30, device=dev))
    target = torch.stack([tx, gap[1].expand(n), gap[2].expand(n)], dim=-1)

    # ---- outer loop: desired world acceleration (PD, per-axis, well damped) ----
    err = target - pos
    kp = torch.tensor([2.2, 4.0, 5.0], device=dev)   # x forward gentler
    kd = torch.tensor([2.6, 3.6, 4.6], device=dev)   # strong vertical damping
    a_des = kp * err - kd * vel
    a_des[:, 0].clamp_(-3.0, 3.0)
    a_des[:, 1].clamp_(-6.0, 6.0)
    a_des[:, 2].clamp_(-6.0, 6.0)
    thrust_world = d.mass * (a_des + torch.tensor([0.0, 0.0, g], device=dev))

    # collective = thrust projected on current body-z, as a fraction of max
    Td = (thrust_world * bz).sum(-1).clamp(min=0.05 * d.max_total_thrust)
    coll = (Td / d.max_total_thrust).clamp(0.05, 0.95)

    # ---- attitude loop: align body-z with desired thrust direction ----
    bz_des = thrust_world / (thrust_world.norm(dim=-1, keepdim=True) + 1e-6)
    e_world = torch.cross(bz, bz_des, dim=-1)              # world-frame att error
    e_body = torch.einsum("nij,nj->ni", R.transpose(1, 2), e_world)
    # gains are in true torque units (N*m); arms are ~0.08 m so these stay small
    k_att, k_rate = 0.9, 0.12
    tau_des = k_att * e_body - k_rate * omega              # desired body torque
    tau_des[:, 2] = -0.02 * omega[:, 2]                    # gently damp yaw
    tau_des = tau_des.clamp(-0.4, 0.4)                     # within motor authority

    # ---- mix desired torque onto the 4 motors using current geometry ----
    phi = env.phi
    ry = env.sy * d.half_span_y * torch.cos(phi)          # (n,4)
    rx = (env.sx * d.half_span_x).expand(n, 4)
    spin = env.spin.expand(n, 4)
    # B maps motor thrusts -> (tau_x, tau_y, tau_z)
    B = torch.stack([ry, -rx, d.yaw_coeff * spin], dim=1)  # (n,3,4)
    Binv = torch.linalg.pinv(B)                            # (n,4,3)
    f_tau = torch.einsum("nkj,nj->nk", Binv, tau_des)     # per-motor thrust (N)
    # convert per-motor thrust delta to a [0,1] command fraction
    f_tau = f_tau / (d.max_total_thrust / 4.0)
    f = (coll.unsqueeze(-1) + f_tau).clamp(0.0, 1.0)
    motor_act = f * 2.0 - 1.0                              # env maps [-1,1]->[0,1]

    # ---- fold the arms while crossing the wall, unfold after ----
    near = (dx < 0.8) & (dx > -0.15)
    fold_cmd = torch.where(near, torch.ones(n, device=dev), -torch.ones(n, device=dev))
    return torch.cat([motor_act, fold_cmd.unsqueeze(-1)], dim=-1)


def rollout(cfg: Config, device, policy="checkpoint", model=None, max_steps=400, seed=0):
    torch.manual_seed(seed)
    env = MorphQuadEnv(cfg, num_envs=1, device=device)
    traj = {"pos": [], "quat": [], "phi": [], "passed": False}
    obs = env.obs()
    for _ in range(max_steps):
        if policy == "checkpoint" and model is not None:
            with torch.no_grad():
                act = model.act_deterministic(obs)
        else:
            act = scripted_action(env)
        traj["pos"].append(env.pos[0].cpu().numpy().copy())
        traj["quat"].append(env.quat[0].cpu().numpy().copy())
        traj["phi"].append(float(env.phi[0].item()))
        obs, rew, terminal, trunc, term_obs, info = env.step(act)
        if bool(info["passed"][0].item()):
            traj["passed"] = True
        if bool((terminal | trunc)[0].item()):
            break
    traj["pos"] = np.array(traj["pos"])
    traj["quat"] = np.array(traj["quat"])
    traj["phi"] = np.array(traj["phi"])
    return cfg, traj


def _rotor_world(cfg: Config, pos, quat, phi):
    """Return (4,3) world rotor positions for one frame."""
    d = cfg.drone
    sx = np.array([+1, -1, +1, -1]) * d.half_span_x
    sy = np.array([+1, -1, -1, +1]) * d.half_span_y * math.cos(phi)
    rb = np.stack([sx, sy, np.zeros(4)], axis=-1)  # (4,3)
    q = torch.tensor(quat, dtype=torch.float32)
    R = Q.quat_to_rotmat(q.unsqueeze(0))[0].numpy()
    return pos[None, :] + rb @ R.T


def _draw_wall(ax, cfg: Config):
    t = cfg.task
    x = t.gap_x
    ylo, yhi, zlo, zhi = -1.5, 1.5, 0.0, 3.0
    gy, gz, gw, gh = t.gap_y, t.gap_z, t.gap_width, t.gap_height
    y0, y1 = gy - gw / 2, gy + gw / 2
    z0, z1 = gz - gh / 2, gz + gh / 2
    panels = [  # (ymin,ymax,zmin,zmax) framing the slot
        (ylo, yhi, zlo, z0),   # bottom
        (ylo, yhi, z1, zhi),   # top
        (ylo, y0, z0, z1),     # left
        (y1, yhi, z0, z1),     # right
    ]
    for (ya, yb, za, zb) in panels:
        Y, Z = np.meshgrid([ya, yb], [za, zb])
        X = np.full_like(Y, x)
        ax.plot_surface(X, Y, Z, color="#8899aa", alpha=0.35, shade=False)


def animate(cfg: Config, traj, out="flight.gif", show=False, fps=30):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    pos = traj["pos"]; quat = traj["quat"]; phi = traj["phi"]
    n = len(pos)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    def frame(i):
        ax.cla()
        _draw_wall(ax, cfg)
        # path so far
        ax.plot(pos[:i + 1, 0], pos[:i + 1, 1], pos[:i + 1, 2],
                color="#1f77b4", lw=1.2, alpha=0.8)
        rw = _rotor_world(cfg, pos[i], quat[i], phi[i])  # (4,3)
        c = pos[i]
        # arms
        for k in range(4):
            ax.plot([c[0], rw[k, 0]], [c[1], rw[k, 1]], [c[2], rw[k, 2]],
                    color="#222222", lw=2.5)
        # rotors
        colors = ["#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
        ax.scatter(rw[:, 0], rw[:, 1], rw[:, 2], c=colors, s=60, depthshade=True)
        ax.scatter([c[0]], [c[1]], [c[2]], c="k", s=30)

        deg = math.degrees(phi[i])
        ax.set_title(f"MorphQuad  |  frame {i+1}/{n}  |  fold {deg:5.1f}deg"
                     + ("   PASSED!" if traj['passed'] and pos[i, 0] > cfg.task.gap_x else ""))
        ax.set_xlim(cfg.task.gap_x - 2.2, cfg.task.gap_x + 1.6)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(0, 3)
        ax.set_xlabel("x (forward)"); ax.set_ylabel("y (lateral)"); ax.set_zlabel("z (up)")
        ax.view_init(elev=18, azim=-60)
        return []

    anim = FuncAnimation(fig, frame, frames=n, interval=1000 / fps, blit=False)
    if out:
        anim.save(out, writer=PillowWriter(fps=fps))
        print(f"saved {out}  ({n} frames)")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Visualize a MorphQuad rollout in 3D.")
    ap.add_argument("--ckpt", default=None, help="path to a trained checkpoint")
    ap.add_argument("--policy", default="auto", choices=["auto", "checkpoint", "scripted"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default="flight.gif")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device(args.device)
    cfg = default_config()
    model = None
    policy = args.policy

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        cfg = _load_cfg_from_ckpt(ckpt)
        model = ActorCritic(OBS_DIM, ACT_DIM, cfg.ppo.hidden_sizes).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        if policy == "auto":
            policy = "checkpoint"
    if policy == "auto":
        policy = "scripted"

    print(f"rollout policy={policy} device={device}")
    cfg, traj = rollout(cfg, device, policy=policy, model=model,
                        max_steps=args.max_steps, seed=args.seed)
    print(f"frames={len(traj['pos'])}  passed={traj['passed']}  "
          f"max_fold={math.degrees(traj['phi'].max()):.1f}deg")
    animate(cfg, traj, out=args.out, show=args.show)


if __name__ == "__main__":
    main()
