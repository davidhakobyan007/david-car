"""Vectorized morphing-quadcopter environment.

Every quantity is a torch tensor with a leading ``num_envs`` dimension, so the
whole batch of simulations advances in lockstep on the GPU. There is no Python
per-env loop anywhere on the hot path -- that is what lets a modern GPU run
thousands of drones at once and stay fully utilized.

Physics
-------
Standard rigid-body quadrotor, but the lateral rotor positions scale with
``cos(phi)`` where ``phi`` is the arm-fold angle. Folding therefore:
  * shrinks the lateral body half-span  (lets it fit a narrow slot),
  * shrinks the roll torque arm          (less roll authority -> must adapt),
  * shrinks Ixx and Izz                  (changes rotational response).
The agent commands 4 motor thrusts plus a fold rate, and must learn to fold at
the right moment to thread a slot that is narrower than its spread-out span.

Observation (24 dims), action (5 dims: 4 motors in [-1,1], 1 fold rate).
"""
from __future__ import annotations

import torch

from .config import Config
from . import quatutil as Q

OBS_DIM = 24
ACT_DIM = 5


class MorphQuadEnv:
    def __init__(self, cfg: Config, num_envs: int, device: torch.device):
        self.cfg = cfg
        self.n = num_envs
        self.device = device
        d, t, s = cfg.drone, cfg.task, cfg.sim

        # static rotor layout (X config), w,x,y,z body frame
        # k: 0 FR(+,+), 1 BL(-,-), 2 FL(+,-), 3 BR(-,+)
        self.sx = torch.tensor([+1.0, -1.0, +1.0, -1.0], device=device)
        self.sy = torch.tensor([+1.0, -1.0, -1.0, +1.0], device=device)
        self.spin = torch.tensor([+1.0, +1.0, -1.0, -1.0], device=device)  # CCW/CW
        self.core_I = torch.tensor(d.core_inertia, device=device)

        self.goal = torch.tensor(
            [t.gap_x + t.target_beyond, t.gap_y, t.gap_z], device=device
        )
        self.gap = torch.tensor([t.gap_x, t.gap_y, t.gap_z], device=device)

        # state tensors
        z = lambda *shape: torch.zeros(self.n, *shape, device=device)
        self.pos = z(3)
        self.vel = z(3)
        self.quat = z(4)
        self.quat[:, 0] = 1.0
        self.omega = z(3)
        self.phi = z(1)              # arm fold angle
        self.motor = z(4)            # actual (lagged) per-motor thrust
        self.steps = torch.zeros(self.n, dtype=torch.long, device=device)
        self.prev_dist = z(1).squeeze(-1)

        self.reset(torch.arange(self.n, device=device))

    # ------------------------------------------------------------------ utils
    def _rotor_pos(self, phi: torch.Tensor) -> torch.Tensor:
        """Body-frame rotor positions (n, 4, 3) for the given fold angle."""
        d = self.cfg.drone
        rx = self.sx * d.half_span_x                      # (4,)
        ry = self.sy * d.half_span_y * torch.cos(phi)     # (n,4)
        rx = rx.expand(self.n, 4)
        rz = torch.zeros_like(rx)
        return torch.stack((rx, ry, rz), dim=-1)

    def _inertia(self, phi: torch.Tensor) -> torch.Tensor:
        """Diagonal inertia (n,3) as a function of fold angle."""
        d = self.cfg.drone
        hx2 = (d.half_span_x ** 2)
        hy2 = (d.half_span_y * torch.cos(phi)) ** 2       # (n,1)
        sum_rx2 = 4.0 * hx2
        sum_ry2 = 4.0 * hy2.squeeze(-1)                   # (n,)
        Ixx = self.core_I[0] + d.rotor_mass * sum_ry2
        Iyy = self.core_I[1] + d.rotor_mass * sum_rx2
        Izz = self.core_I[2] + d.rotor_mass * (sum_rx2 + sum_ry2)
        Iyy = Iyy * torch.ones_like(Ixx)
        return torch.stack((Ixx, Iyy, Izz), dim=-1)

    def _eff_halfwidth(self, phi: torch.Tensor) -> torch.Tensor:
        d = self.cfg.drone
        return d.half_span_y * torch.cos(phi) + d.prop_radius   # (n,1)

    # ------------------------------------------------------------------ reset
    def reset(self, idx: torch.Tensor) -> None:
        t = self.cfg.task
        m = idx.numel()
        if m == 0:
            return
        dev = self.device
        jp, jv, jt = t.spawn_jitter_pos, t.spawn_jitter_vel, t.spawn_jitter_tilt

        pos = torch.zeros(m, 3, device=dev)
        pos[:, 0] = t.gap_x + t.spawn_dx
        pos[:, 1] = t.gap_y + (torch.rand(m, device=dev) * 2 - 1) * jp
        pos[:, 2] = t.gap_z + (torch.rand(m, device=dev) * 2 - 1) * jp
        self.pos[idx] = pos

        self.vel[idx] = (torch.rand(m, 3, device=dev) * 2 - 1) * jv

        # small random tilt around upright
        ax = (torch.rand(m, 3, device=dev) * 2 - 1)
        ax = ax / (ax.norm(dim=-1, keepdim=True) + 1e-8)
        ang = (torch.rand(m, 1, device=dev) * 2 - 1) * jt
        q = torch.cat((torch.cos(ang / 2), ax * torch.sin(ang / 2)), dim=-1)
        self.quat[idx] = Q.quat_normalize(q)

        self.omega[idx] = (torch.rand(m, 3, device=dev) * 2 - 1) * 0.2
        self.phi[idx] = 0.0
        self.motor[idx] = self.cfg.drone.hover_thrust / 4.0
        self.steps[idx] = 0
        self.prev_dist[idx] = (self.goal - self.pos[idx]).norm(dim=-1)

    # ------------------------------------------------------------- observation
    def _obs(self) -> torch.Tensor:
        t = self.cfg.task
        goal_rel = self.goal - self.pos                      # (n,3)
        wall_dx = (self.gap[0] - self.pos[:, 0:1])           # (n,1)
        bodyz = Q.quat_rotate(
            self.quat, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.n, 3)
        )
        hw = self._eff_halfwidth(self.phi)                   # (n,1)
        lat_off = (self.pos[:, 1:2] - self.gap[1])
        vert_off = (self.pos[:, 2:3] - self.gap[2])
        gw = torch.full((self.n, 1), t.gap_width, device=self.device)
        gh = torch.full((self.n, 1), t.gap_height, device=self.device)
        return torch.cat((
            goal_rel, wall_dx, self.vel, bodyz, self.quat,
            self.omega, self.phi, torch.cos(self.phi), hw, gw, gh,
            lat_off, vert_off,
        ), dim=-1)

    def obs(self) -> torch.Tensor:
        return self._obs()

    # -------------------------------------------------------------- physics
    def _integrate(self, motor_target: torch.Tensor, fold_rate: torch.Tensor, dt: float):
        d, s = self.cfg.drone, self.cfg.sim
        g = s.gravity
        # motor first-order lag toward target
        alpha = dt / (d.motor_tau + dt)
        self.motor = self.motor + alpha * (motor_target - self.motor)
        f = self.motor                                      # (n,4) >=0

        # fold angle update
        self.phi = torch.clamp(self.phi + fold_rate * dt, 0.0, d.phi_max)

        rotor = self._rotor_pos(self.phi)                   # (n,4,3)
        # body-frame thrust (each motor along +z)
        thrust_total = f.sum(dim=-1, keepdim=True)          # (n,1)
        Fb = torch.zeros(self.n, 3, device=self.device)
        Fb[:, 2:3] = thrust_total

        # torques in body frame: tau = sum r x (0,0,f)
        ry = rotor[..., 1]                                  # (n,4)
        rx = rotor[..., 0]
        tau_x = (ry * f).sum(dim=-1)
        tau_y = (-rx * f).sum(dim=-1)
        tau_z = (d.yaw_coeff * self.spin * f).sum(dim=-1)
        tau = torch.stack((tau_x, tau_y, tau_z), dim=-1)    # (n,3)
        tau = tau - d.ang_drag * self.omega

        # translational dynamics (world frame)
        Fw = Q.quat_rotate(self.quat, Fb)
        acc = Fw / d.mass
        acc[:, 2] -= g
        acc = acc - (d.lin_drag / d.mass) * self.vel
        self.vel = self.vel + acc * dt
        self.pos = self.pos + self.vel * dt

        # rotational dynamics (body frame, diagonal inertia)
        I = self._inertia(self.phi)                         # (n,3)
        Iw = I * self.omega
        gyro = torch.cross(self.omega, Iw, dim=-1)
        omega_dot = (tau - gyro) / I
        self.omega = self.omega + omega_dot * dt
        self.quat = Q.quat_integrate(self.quat, self.omega, dt)

    # ----------------------------------------------------------------- step
    def step(self, action: torch.Tensor):
        """action: (n,5) roughly in [-1,1].

        Returns obs, reward, done(terminal, no bootstrap), trunc(timeout),
        terminal_obs (state just before reset, for truncation bootstrapping).
        """
        cfg = self.cfg
        d, t, r = cfg.drone, cfg.task, cfg.reward
        action = torch.nan_to_num(action)

        motor_cmd = torch.clamp(action[:, 0:4] * 0.5 + 0.5, 0.0, 1.0)
        motor_target = motor_cmd * (d.max_total_thrust / 4.0)
        fold_rate = torch.clamp(action[:, 4:5], -1.0, 1.0) * d.fold_rate_max

        prev_x = self.pos[:, 0].clone()

        sub_dt = cfg.sim.dt / cfg.sim.substeps
        for _ in range(cfg.sim.substeps):
            self._integrate(motor_target, fold_rate, sub_dt)

        self.steps += 1
        cur_x = self.pos[:, 0]
        up = Q.upright_cos(self.quat)

        # ---- reward shaping ----
        dist = (self.goal - self.pos).norm(dim=-1)
        progress = (self.prev_dist - dist)
        self.prev_dist = dist

        reward = progress * r.progress
        reward = reward + r.alive
        reward = reward + r.upright * up
        reward = reward - r.ctrl_cost * motor_cmd.mean(dim=-1)
        reward = reward - r.fold_cost * fold_rate.abs().squeeze(-1)
        reward = reward - r.spin_cost * (self.omega ** 2).sum(dim=-1)
        reward = reward - r.speed_cost * self.vel.norm(dim=-1)
        reward = reward - r.level_cost * (1.0 - up).clamp(min=0.0)

        # ---- shaping: teach the morphing trick near the wall ----
        hw = self._eff_halfwidth(self.phi).squeeze(-1)
        dx = t.gap_x - cur_x
        near = ((dx > 0.0) & (dx < r.fold_zone)).float()
        phi_frac = (self.phi.squeeze(-1) / d.phi_max)
        # reward folding as it approaches the wall
        reward = reward + r.fold_shaping * near * phi_frac
        # reward actually being narrow enough to clear the slot (centered)
        slack = (t.gap_width * 0.5) - ((self.pos[:, 1] - t.gap_y).abs() + hw)
        reward = reward + r.fit_shaping * near * torch.tanh(8.0 * slack).clamp(min=0)
        # reward vertical alignment with the slot
        z_slack = (t.gap_height * 0.5) - (self.pos[:, 2] - t.gap_z).abs()
        reward = reward + r.align_shaping * near * torch.tanh(6.0 * z_slack).clamp(min=0)

        # ---- gap crossing event ----
        crossing = (prev_x < t.gap_x) & (cur_x >= t.gap_x)
        lat_ok = (self.pos[:, 1] - t.gap_y).abs() + hw < (t.gap_width * 0.5)
        vert_ok = (self.pos[:, 2] - t.gap_z).abs() + d.prop_radius < (t.gap_height * 0.5)
        fit = lat_ok & vert_ok & (up > 0)
        passed = crossing & fit
        clipped = crossing & (~fit)

        # ---- out-of-bounds / flip ----
        oob = (
            (self.pos[:, 1].abs() > t.bound_y)
            | (self.pos[:, 2] < t.bound_z_low)
            | (self.pos[:, 2] > t.bound_z_high)
            | (self.pos[:, 0] < t.gap_x - t.bound_x_back)
            | (up < t.flip_cos_limit)
        )

        reward = reward + r.pass_bonus * passed.float()
        crash = clipped | oob
        reward = reward + r.crash_penalty * crash.float()

        terminal = passed | crash                 # true episode end (no bootstrap)
        trunc = (self.steps >= t.max_steps) & (~terminal)
        done = terminal | trunc

        terminal_obs = self._obs()                # state before any reset

        reset_idx = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if reset_idx.numel() > 0:
            self.reset(reset_idx)

        return self._obs(), reward, terminal, trunc, terminal_obs, {
            "passed": passed, "crash": crash,
        }
