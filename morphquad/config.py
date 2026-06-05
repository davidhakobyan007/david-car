"""Central configuration for MorphQuad.

All physical units are SI (meters, kilograms, seconds, radians).
The defaults describe a ~5-inch racing/freestyle quad (motor-to-motor
diagonal ~0.23 m, thrust-to-weight ~2.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class DroneConfig:
    """Geometry and dynamics of the morphing quad.

    The arms can fold about the body hinges. ``phi`` is the fold angle:
    ``phi = 0``  -> arms fully spread (wide X, max roll authority, widest body)
    ``phi -> phi_max`` -> arms folded toward the fore/aft axis, so the craft
    becomes narrow in the lateral (y) direction and can slip through a slot.

    Lateral half-span scales with ``cos(phi)``: folding shrinks the y extent
    (and the roll torque arm and the Ixx / Izz inertias along with it), which
    is exactly the coupling the agent must learn to exploit.
    """
    mass: float = 0.60              # total mass (kg)
    rotor_mass: float = 0.045       # per-rotor lumped mass (kg) for inertia
    half_span_x: float = 0.080      # fore/aft half-distance to a rotor (m)
    half_span_y: float = 0.080      # lateral half-distance to a rotor at phi=0
    prop_radius: float = 0.0635     # 5 inch prop -> ~0.0635 m radius (collision)
    core_inertia: tuple = (2.0e-3, 2.0e-3, 3.5e-3)  # body-core diag inertia

    thrust_to_weight: float = 2.5   # total max thrust / weight
    yaw_coeff: float = 0.016        # rotor drag-torque / thrust ratio (yaw)
    motor_tau: float = 0.03         # 1st-order motor lag time constant (s)

    phi_max: float = 1.40           # max fold angle (rad) ~ 80 deg
    fold_rate_max: float = 6.0      # max |d phi / dt| (rad/s)

    lin_drag: float = 0.10          # translational drag coeff
    ang_drag: float = 0.010         # rotational drag coeff

    @property
    def hover_thrust(self) -> float:
        return self.mass * 9.81

    @property
    def max_total_thrust(self) -> float:
        return self.thrust_to_weight * self.mass * 9.81


@dataclass
class TaskConfig:
    """The 'fly through a tight gap' task.

    A thin wall sits at ``x = gap_x`` with a rectangular slot of width
    ``gap_width`` (lateral / y) and height ``gap_height`` (vertical / z).
    The slot is deliberately narrower than the unfolded lateral span, so the
    drone *must* fold its arms to pass without clipping the wall.
    """
    gap_x: float = 0.0
    gap_y: float = 0.0
    gap_z: float = 1.5
    gap_width: float = 0.26         # lateral opening (m) -- still forces folding
    gap_height: float = 0.80        # vertical opening (m)

    spawn_dx: float = -2.0          # spawn this far in front of the wall (x)
    spawn_jitter_pos: float = 0.18  # +/- spawn position noise (m)
    spawn_jitter_vel: float = 0.15  # +/- spawn velocity noise (m/s)
    spawn_jitter_tilt: float = 0.08 # +/- initial tilt noise (rad)

    target_beyond: float = 1.2      # goal point this far past the wall (m)

    # episode termination bounds
    max_steps: int = 400
    bound_y: float = 3.0
    bound_z_low: float = 0.20
    bound_z_high: float = 4.0
    bound_x_back: float = 3.0       # too far behind the wall
    bound_x_front: float = 3.0      # too far past the wall (done = success path)
    flip_cos_limit: float = -0.2    # upright cos below this -> flipped/crashed


@dataclass
class RewardConfig:
    progress: float = 0.7           # reward per meter of progress to the goal
    alive: float = 0.02             # small per-step alive bonus
    upright: float = 0.02           # reward * (body_z . world_z)
    pass_bonus: float = 30.0        # big reward for cleanly crossing the slot
    crash_penalty: float = -6.0     # hit wall / out of bounds / flipped
    ctrl_cost: float = 0.002        # penalize thrust magnitude
    fold_cost: float = 0.001        # penalize fold-rate effort
    spin_cost: float = 0.002        # penalize angular velocity (encourage calm)

    # --- shaping so the agent actually discovers the morphing trick ---
    # Near the wall, lining up + folding pays MORE than just barreling forward,
    # so the agent learns to set up the pass instead of crashing through.
    fold_zone: float = 1.2          # start rewarding folding this far before wall
    fold_shaping: float = 0.8       # reward folding (phi up) while near the wall
    fit_shaping: float = 1.6        # reward when narrow enough to fit the slot
    align_shaping: float = 1.0      # reward lining up with the slot near the wall


@dataclass
class PPOConfig:
    num_envs: int = 4096            # parallel sims -> raise to saturate the GPU
    rollout_len: int = 32           # steps per env per update
    total_steps: int = 30_000_000   # total env steps to train
    epochs: int = 4
    num_minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.002
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    learning_rate: float = 3.0e-4
    anneal_lr: bool = True
    hidden_sizes: tuple = (256, 256)
    amp: bool = True                # mixed precision (uses tensor cores)
    seed: int = 0


@dataclass
class SimConfig:
    dt: float = 0.01                # control step (s)
    substeps: int = 2               # physics substeps per control step
    gravity: float = 9.81


@dataclass
class Config:
    drone: DroneConfig = field(default_factory=DroneConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    sim: SimConfig = field(default_factory=SimConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def default_config() -> Config:
    return Config()
