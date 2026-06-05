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
import math
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


# A real 5-inch quad (~0.16 m span) is tiny next to a 2.8 m wall, so we draw the
# airframe a bit bigger than life. The fold ratio (cos phi) is preserved, so the
# visual still tells the truth: spread looks too wide for the hole, folded fits.
DRONE_VIS_SCALE = 1.8


def rotor_world(env: MorphQuadEnv, scale: float = DRONE_VIS_SCALE):
    """Return (centers (n,3), rotors (n,4,3), phi (n,)) as numpy, world frame."""
    d = env.cfg.drone
    n = env.n
    phi = env.phi  # (n,1)
    rx = (env.sx * d.half_span_x).expand(n, 4)
    ry = env.sy * d.half_span_y * torch.cos(phi)        # (n,4)
    rz = torch.zeros_like(rx)
    rb = torch.stack((rx, ry, rz), dim=-1) * scale      # (n,4,3) body frame
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


# ====================================================================== 3D
class Camera3D:
    """Minimal perspective camera (world: x=forward, y=lateral, z=up)."""

    def __init__(self, w, h, fov_deg=58.0):
        self.w, self.h = w, h
        self.focal = 0.5 * w / math.tan(math.radians(fov_deg) / 2)
        self.eye = np.zeros(3)
        self.R = np.eye(3)

    def look_at(self, eye, target, up=(0.0, 0.0, 1.0)):
        eye = np.asarray(eye, float); target = np.asarray(target, float)
        up = np.asarray(up, float)
        f = target - eye; f /= np.linalg.norm(f) + 1e-9
        r = np.cross(f, up); r /= np.linalg.norm(r) + 1e-9
        u = np.cross(r, f)
        self.eye = eye
        self.R = np.stack([r, u, f], axis=0)      # rows: right, up, forward

    def project(self, pts):
        """pts (...,3) -> (screen (...,2) float, depth (...), valid (...))."""
        shape = pts.shape[:-1]
        cam = (pts.reshape(-1, 3) - self.eye) @ self.R.T
        z = cam[:, 2]
        valid = z > 0.05
        zc = np.where(valid, z, 1.0)
        sx = self.w * 0.5 + self.focal * cam[:, 0] / zc
        sy = self.h * 0.5 - self.focal * cam[:, 1] / zc
        scr = np.stack([sx, sy], axis=-1)
        return scr.reshape(shape + (2,)), z.reshape(shape), valid.reshape(shape)


def _ipt(p, valid):
    if not valid or not (np.isfinite(p[0]) and np.isfinite(p[1])):
        return None
    x = int(min(max(p[0], -2000), 4000))
    y = int(min(max(p[1], -2000), 4000))
    return (x, y)


def _seg(pg, surf, cam, a, b, col, width=1):
    P, _, V = cam.project(np.stack([a, b]))
    pa = _ipt(P[0], V[0]); pb = _ipt(P[1], V[1])
    if pa and pb:
        pg.draw.line(surf, col, pa, pb, width)


def draw_world_3d(pg, surf, cam, cfg, centers, rotors, phis, hud_h):
    t = cfg.task
    gx = t.gap_x

    # ---- ground grid (z = 0) ----
    x0, x1 = gx - 3.0, gx + 1.6
    y0g, y1g = -1.6, 1.6
    for gxx in np.arange(x0, x1 + 1e-3, 0.5):
        _seg(pg, surf, cam, [gxx, y0g, 0], [gxx, y1g, 0], GRID)
    for gyy in np.arange(y0g, y1g + 1e-3, 0.5):
        _seg(pg, surf, cam, [x0, gyy, 0], [x1, gyy, 0], GRID)

    # ---- wall (4 quads framing the slot), painter-sorted with drones ----
    gy, gz, gw, gh = t.gap_y, t.gap_z, t.gap_width, t.gap_height
    yl, yr = gy - gw / 2, gy + gw / 2
    zb, zt = gz - gh / 2, gz + gh / 2
    YL, YR, ZB, ZT = -1.4, 1.4, 0.0, 3.0
    quads = [
        [(gx, YL, ZB), (gx, YR, ZB), (gx, YR, zb), (gx, YL, zb)],   # bottom
        [(gx, YL, zt), (gx, YR, zt), (gx, YR, ZT), (gx, YL, ZT)],   # top
        [(gx, YL, zb), (gx, yl, zb), (gx, yl, zt), (gx, YL, zt)],   # left
        [(gx, yr, zb), (gx, YR, zb), (gx, YR, zt), (gx, yr, zt)],   # right
    ]
    for q in quads:
        pts = np.array(q, float)
        P, Z, V = cam.project(pts)
        if V.all():
            poly = [(_ipt(P[i], V[i])) for i in range(4)]
            if all(p is not None for p in poly):
                pg.draw.polygon(surf, WALL, poly)
                pg.draw.polygon(surf, (60, 68, 84), poly, 1)
    # highlight the hole opening
    hole = np.array([(gx, yl, zb), (gx, yr, zb), (gx, yr, zt), (gx, yl, zt)], float)
    P, _, V = cam.project(hole)
    if V.all():
        poly = [_ipt(P[i], V[i]) for i in range(4)]
        if all(p is not None for p in poly):
            pg.draw.polygon(surf, GOOD, poly, 2)

    # ---- drones, far-to-near ----
    n = centers.shape[0]
    _, depth, _ = cam.project(centers)
    order = np.argsort(-depth)        # far first
    for i in order:
        col = DRONE0 if i == 0 else DRONE
        cP, _, cV = cam.project(centers[i][None])
        c = _ipt(cP[0], cV[0])
        if c is None:
            continue
        # height drop-line to the ground for depth cue
        ground = centers[i].copy(); ground[2] = 0.0
        _seg(pg, surf, cam, centers[i], ground, GRID)
        rP, _, rV = cam.project(rotors[i])     # (4,2)
        for k in range(4):
            r = _ipt(rP[k], rV[k])
            if r is None:
                continue
            pg.draw.line(surf, ARM if i == 0 else col, c, r, 2 if i == 0 else 1)
            rad = 4 if i == 0 else 3
            pg.draw.circle(surf, col, r, rad)
            pg.draw.circle(surf, (255, 255, 255), r, max(1, rad - 2))
        pg.draw.circle(surf, col, c, 3)


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
    ap.add_argument("--view", default="3d", choices=["3d", "2d"],
                    help="3d orbiting camera (default) or stacked 2d panels")
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
    cam = Camera3D(W, H)
    cam_t = 0.0
    gx = cfg.task.gap_x

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
            if args.view == "3d":
                cam_t += 0.012
                side = 2.4 * math.sin(cam_t)           # orbit side-to-side
                cam.look_at(eye=(gx - 3.4, side, 1.7),
                            target=(gx + 0.25, 0.0, 1.45))
                draw_world_3d(pg, screen, cam, cfg, centers, rotors, phis, hud_h)
                # flash outcome rings (project drone centre)
                for i, fv in list(flash.items()):
                    fv[1] -= 1
                    if fv[1] <= 0:
                        del flash[i]; continue
                    P, _, V = cam.project(centers[i][None])
                    c = _ipt(P[0], V[0])
                    if c is not None:
                        pg.draw.circle(screen, fv[0], c, 18, 2)
            else:
                draw_panel(pg, screen, font, p_side, cfg, centers, rotors, phis,
                           2, "SIDE VIEW  (fly through the hole)")
                draw_panel(pg, screen, font, p_top, cfg, centers, rotors, phis,
                           1, "TOP-DOWN  (arms fold to fit the slot)")
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
