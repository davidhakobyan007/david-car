"""MorphQuad: a GPU-accelerated reinforcement-learning environment for a
morphing 5-inch quadcopter that folds its arms (X <-> H <-> folded) to fly
through tight gaps.

Modules
-------
config   : all tunable parameters (drone geometry, task, PPO).
quatutil : batched quaternion / rigid-body math (pure torch).
env      : the vectorized morphing-quad simulator (thousands of envs on GPU).
model    : actor-critic network.
ppo      : PPO trainer (mixed precision, GPU saturating).
train    : CLI entry point for training.
viz      : load a trained policy and render a 3D rollout (drone + folding arms).
"""

__all__ = ["config", "quatutil", "env", "model", "ppo"]
__version__ = "0.1.0"
