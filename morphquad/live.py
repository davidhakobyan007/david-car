"""Live pygame trainer: watch the morphing quad LEARN to fold through the hole.

Runs real PPO training on the GPU and, between updates, renders a handful of
drones flying at the wall with the *current* policy -- so you can watch the
behavior improve and see live counters of how often it makes it through vs.
crashes into the wall.

Two stacked views, both showing the wall + hole and the drones:
  * top:    side view  (x = forward, z = up)   -> shows it fly THROUGH the hole
  * bottom: top-down   (x = forward, y = lateral) -> shows the ARMS FOLD to fit

Run (needs a display; over SSH use X-forwarding or run train.py headless):
    python -m morphquad.live                 # auto GPU
    python -m morphquad.live --envs 8192 --viz-envs 6
    python -m morphquad.live --device cpu --envs 256   # slow, for testing
"""
from __future__ import annotations

import argparse
import collections
import os

import numpy as np
import torch

from .config import default_config
from .device import pick_device, tune_for_throughput, describe
from .env import MorphQuadEnv
from .ppo import PPOTrainer
from . import quatutil as Q

# ------------------------------------------------------------------ colours
BG = (16, 18, 24)
PANEL = (24, 27, 36)
WALL = (90, 100, 120)
HOLE = (16, 18, 24)
GRID = (38, 42, 54)
TEXT = (220, 226, 236)
DIM = (140, 148, 162)
DRONE0 = (80, 220, 255)
DRONE = (90, 150, 200)
ARM = (235, 238, 245)
GOOD = (90, 230, 140)
BAD = (240, 90, 90)
WARN = (240, 200, 90)


def rotor_world(env: MorphQuadEnv):
    """Return (centers (n,3), rotors (n,4,3), phi (n,)) as numpy, world frame."""
    d = env.cfg.drone
    n = env.n
    phi = env.phi  # (n,1)
    rx = (env.sx * d.half_span_x).expand(n, 4)
    ry = env.sy * d.half_span_y * torch.cos(phi)        # (n,4)
    rz = torch.zeros_like(rx)
    rb = torch.stack((rx, ry, rz), dim=-1)              # (n,4,3) body frame
    R = Q.quat_to_rotmat(env.quat)                      # (n,3,3)
    rw = torch.einsum("nij,nkj->nki", R, rb) + env.pos[:, None, :]
    return (env.pos.detach().cpu().numpy(),
            rw.detach().cpu().numpy(),
            phi.squeeze(-1).detach().cpu().numpy())


class Panel:
    """Maps world coords to a screen rectangle (with y/z axis flipped up)."""

    def __init__(self, rect, xr, vr, vaxis_label):
        self.x0, self.y0, self.w, self.h = rect
        self.xr = xr            # (xmin, xmax) world forward axis
        self.vr = vr            # (vmin, vmax) world vertical-on-screen axis
        self.vaxis_label = vaxis_label
        self.pad = 36

    def sx(self, x):
        xmin, xmax = self.xr
        f = (x - xmin) / (xmax - xmin)
        return self.x0 + self.pad + f * (self.w - 2 * self.pad)

    def sy(self, v):
        vmin, vmax = self.vr
        f = (v - vmin) / (vmax - vmin)
        return self.y0 + self.h - self.pad - f * (self.h - 2 * self.pad)

    def pt(self, x, v):
        """Safe integer screen point: handles nan/inf, clamps to the panel."""
        sx = self.sx(x); sv = self.sy(v)
        if not (np.isfinite(sx) and np.isfinite(sv)):
            return None
        sx = int(min(max(sx, self.x0 - 50), self.x0 + self.w + 50))
        sv = int(min(max(sv, self.y0 - 50), self.y0 + self.h + 50))
        return (sx, sv)


def draw_panel(pg, surf, font, panel: Panel, cfg, centers, rotors, phis,
               vidx, title):
    """vidx: which world-coord index is the vertical axis (2=z side, 1=y top)."""
    t = cfg.task
    pg.draw.rect(surf, PANEL, (panel.x0, panel.y0, panel.w, panel.h),
                 border_radius=8)

    # grid
    for gx in np.linspace(panel.xr[0], panel.xr[1], 6):
        x = panel.sx(gx)
        pg.draw.line(surf, GRID, (x, panel.y0 + panel.pad),
                     (x, panel.y0 + panel.h - panel.pad), 1)
    for gv in np.linspace(panel.vr[0], panel.vr[1], 5):
        y = panel.sy(gv)
        pg.draw.line(surf, GRID, (panel.x0 + panel.pad, y),
                     (panel.x0 + panel.w - panel.pad, y), 1)

    # wall with a hole (two bars framing the slot on the vertical axis)
    wx = panel.sx(t.gap_x)
    if vidx == 2:
        gc, gh = t.gap_z, t.gap_height
    else:
        gc, gh = t.gap_y, t.gap_width
    v0, v1 = gc - gh / 2, gc + gh / 2
    top = panel.y0 + panel.pad
    bot = panel.y0 + panel.h - panel.pad
    y_lo = panel.sy(v0)
    y_hi = panel.sy(v1)
    bar_w = 8
    pg.draw.rect(surf, WALL, (wx - bar_w / 2, top, bar_w, y_hi - top))
    pg.draw.rect(surf, WALL, (wx - bar_w / 2, y_lo, bar_w, bot - y_lo))
    # hole markers
    pg.draw.line(surf, GOOD, (wx - 10, y_lo), (wx + 10, y_lo), 2)
    pg.draw.line(surf, GOOD, (wx - 10, y_hi), (wx + 10, y_hi), 2)

    # drones
    n = centers.shape[0]
    for i in range(n):
        col = DRONE0 if i == 0 else DRONE
        c = panel.pt(centers[i, 0], centers[i, vidx])
        if c is None:
            continue
        for k in range(4):
            r = panel.pt(rotors[i, k, 0], rotors[i, k, vidx])
            if r is None:
                continue
            pg.draw.line(surf, ARM if i == 0 else col, c, r,
                         2 if i == 0 else 1)
            pg.draw.circle(surf, col, r, 3 if i == 0 else 2)
        pg.draw.circle(surf, col, c, 3)

    label = font.render(title, True, DIM)
    surf.blit(label, (panel.x0 + panel.pad, panel.y0 + 8))
    vl = font.render(panel.vaxis_label, True, GRID)
    surf.blit(vl, (panel.x0 + 6, panel.y0 + panel.h // 2))


def main():
    ap = argparse.ArgumentParser(description="Live pygame training viewer.")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--envs", type=int, default=4096, help="training envs (GPU load)")
    ap.add_argument("--viz-envs", type=int, default=6, help="drones shown on screen")
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--total-steps", type=int, default=200_000_000)
    ap.add_argument("--updates-per-frame", type=int, default=1)
    ap.add_argument("--frames-per-update", type=int, default=18,
                    help="render steps shown between training updates")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--stochastic", action="store_true",
                    help="show sampled (exploring) actions instead of the mean")
    ap.add_argument("--save", default="checkpoints/morphquad_live.pt")
    args = ap.parse_args()

    import pygame as pg

    cfg = default_config()
    cfg.ppo.num_envs = args.envs
    cfg.ppo.rollout_len = args.rollout
    cfg.ppo.total_steps = args.total_steps
    if args.no_amp:
        cfg.ppo.amp = False

    device = pick_device(args.device)
    tune_for_throughput(device)
    print(describe(device))

    trainer = PPOTrainer(cfg, device)
    viz = MorphQuadEnv(cfg, num_envs=args.viz_envs, device=device)
    viz_obs = viz.obs()

    pg.init()
    W, H = 940, 760
    screen = pg.display.set_mode((W, H))
    pg.display.set_caption("MorphQuad — live training")
    font = pg.font.SysFont("consolas,menlo,monospace", 16)
    big = pg.font.SysFont("consolas,menlo,monospace", 22, bold=True)
    clock = pg.time.Clock()

    xr = (cfg.task.gap_x - 2.3, cfg.task.gap_x + 1.6)
    hud_h = 132
    p_side = Panel((10, hud_h, W - 20, (H - hud_h - 20) // 2),
                   xr, (0.0, 3.0), "z")
    p_top = Panel((10, hud_h + (H - hud_h - 20) // 2 + 10, W - 20,
                   (H - hud_h - 20) // 2),
                  xr, (-0.75, 0.75), "y")

    # counters
    passes = crashes = timeouts = 0
    recent = collections.deque(maxlen=200)   # 1 pass / 0 fail
    flash = {}  # viz env idx -> (color, frames left)
    update = 0
    stats = None
    paused = False
    running = True

    num_updates = trainer.num_updates

    while running:
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                running = False
            elif ev.type == pg.KEYDOWN:
                if ev.key in (pg.K_ESCAPE, pg.K_q):
                    running = False
                elif ev.key == pg.K_SPACE:
                    paused = not paused
                elif ev.key == pg.K_s:
                    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
                    torch.save({"model": trainer.model.state_dict(),
                                "cfg": cfg.to_dict()}, args.save)
                    print(f"saved {args.save}")

        # ---- training: a few PPO updates ----
        if not paused:
            for _ in range(args.updates_per_frame):
                update += 1
                stats = trainer.update_once(update, num_updates)

        # ---- step the on-screen drones with the current policy ----
        for _ in range(args.frames_per_update):
            with torch.no_grad():
                if args.stochastic:
                    act, _, _ = trainer.model.act(viz_obs)
                else:
                    act = trainer.model.act_deterministic(viz_obs)
            viz_obs, rew, terminal, trunc, term_obs, info = viz.step(act.float())
            done = (terminal | trunc)
            if done.any():
                pa = info["passed"]
                cr = info["crash"]
                for i in torch.nonzero(done, as_tuple=False).squeeze(-1).tolist():
                    if bool(pa[i]):
                        passes += 1; recent.append(1); flash[i] = [GOOD, 12]
                    elif bool(cr[i]):
                        crashes += 1; recent.append(0); flash[i] = [BAD, 12]
                    else:
                        timeouts += 1; recent.append(0); flash[i] = [WARN, 12]

            centers, rotors, phis = rotor_world(viz)

            # ---------------- draw ----------------
            screen.fill(BG)
            draw_panel(pg, screen, font, p_side, cfg, centers, rotors, phis,
                       2, "SIDE VIEW  (fly through the hole)")
            draw_panel(pg, screen, font, p_top, cfg, centers, rotors, phis,
                       1, "TOP-DOWN  (arms fold to fit the slot)")

            # flash outcome rings on env 0 in both panels
            for i, fv in list(flash.items()):
                fv[1] -= 1
                if fv[1] <= 0:
                    del flash[i]; continue
                for panel, vidx in ((p_side, 2), (p_top, 1)):
                    c = panel.pt(centers[i, 0], centers[i, vidx])
                    if c is not None:
                        pg.draw.circle(screen, fv[0], c, 16, 2)

            # ---------------- HUD ----------------
            total = passes + crashes + timeouts
            rate = (sum(recent) / len(recent)) if recent else 0.0
            pg.draw.rect(screen, PANEL, (10, 10, W - 20, hud_h - 20),
                         border_radius=8)
            title = big.render("MorphQuad — learning to fold through the hole",
                               True, TEXT)
            screen.blit(title, (28, 20))

            def line(txt, x, y, col=TEXT):
                screen.blit(font.render(txt, True, col), (x, y))

            phi_deg = float(np.degrees(phis[0])) if phis.size else 0.0
            line(f"PASSED  {passes}", 28, 56, GOOD)
            line(f"CRASHED {crashes}", 190, 56, BAD)
            line(f"TIMEOUT {timeouts}", 360, 56, WARN)
            line(f"recent pass-rate {rate:5.1%}  (last {len(recent)})", 540, 56,
                 GOOD if rate > 0.5 else DIM)

            if stats is not None:
                line(f"update {stats.update:>5}/{num_updates}   "
                     f"step {stats.global_step:>11,}   {stats.sps:>7,} sps",
                     28, 84, DIM)
                line(f"ep_return {stats.ep_return:7.2f}   "
                     f"train pass {stats.pass_rate:5.1%}   "
                     f"entropy {stats.entropy:4.2f}   "
                     f"arm-fold(env0) {phi_deg:4.0f}deg",
                     28, 106, DIM)
            line("SPACE pause   S save   Q quit", W - 270, 84, DIM)
            if paused:
                line("[PAUSED]", W - 110, 106, WARN)

            pg.display.flip()
            clock.tick(60)

    # save on exit
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    torch.save({"model": trainer.model.state_dict(), "cfg": cfg.to_dict()},
               args.save)
    print(f"saved {args.save}")
    pg.quit()


if __name__ == "__main__":
    main()
