"""Batched quaternion and rigid-body helpers (pure torch, GPU friendly).

Quaternions are stored as (..., 4) tensors in **w, x, y, z** order and are
assumed to represent a body->world rotation.
"""
from __future__ import annotations

import torch


def quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two (..., 4) quaternions (w,x,y,z)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((w, x, y, z), dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors ``v`` (..., 3) by quaternions ``q`` (..., 4)."""
    qw = q[..., 0:1]
    qv = q[..., 1:4]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by the inverse (conjugate) of ``q``: world->body."""
    qw = q[..., 0:1]
    qv = -q[..., 1:4]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


def quat_integrate(q: torch.Tensor, omega_body: torch.Tensor, dt: float) -> torch.Tensor:
    """Integrate a quaternion given a body-frame angular velocity.

    q_dot = 0.5 * q (x) [0, omega_body]
    """
    zeros = torch.zeros_like(omega_body[..., :1])
    omega_quat = torch.cat((zeros, omega_body), dim=-1)
    q_dot = 0.5 * quat_mul(q, omega_quat)
    return quat_normalize(q + q_dot * dt)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert (..., 4) quaternions to (..., 3, 3) rotation matrices."""
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    m = torch.stack((
        1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),
        2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy),
    ), dim=-1)
    return m.reshape(q.shape[:-1] + (3, 3))


def upright_cos(q: torch.Tensor) -> torch.Tensor:
    """Cosine of the tilt angle = (body +z axis) . (world +z axis)."""
    bz = quat_rotate(q, torch.tensor([0.0, 0.0, 1.0], device=q.device).expand_as(q[..., :3]))
    return bz[..., 2]
