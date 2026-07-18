"""Temporally-correlated exploration noise for continuous-time actor-critic / value-gradient control.

The exploration perturbation must be a genuine continuous-time process, not white noise. White
noise is not a process (it is the formal derivative of Brownian motion, with no measurable sample
path), and its effect on the plant over a horizon T scales as sigma*sqrt(T*dt) -> 0 as dt -> 0: it
does not explore the state space, it averages out. Doya (2000) therefore uses low-pass-filtered
(Ornstein-Uhlenbeck) noise, and so does this module.

Ornstein-Uhlenbeck:  dn = -(1/tau_n) n dt + sqrt(2/tau_n) sigma dW,
stationary variance sigma^2, correlation over a lag Delta t equal to exp(-Delta t / tau_n) -- a
function of PHYSICAL TIME, independent of the discretisation dt. The exact discretisation on a grid
of step dt is

    n_{k+1} = a n_k + sqrt(1 - a^2) sigma xi_k,     a = exp(-dt / tau_n),  xi_k ~ N(0, I),

which reproduces both the stationary variance sigma^2 and the adjacent-step correlation a exactly
(not an Euler approximation). Contrast the previous squared-exponential Gaussian-process sampler,
whose adjacent-step correlation exp(-dt^2 / 2 l^2) was 0.000 to float64 at the shipped length_scale
= 0.002, i.e. white, and which cost an O(n^3) Cholesky per episode; this is O(n).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def sample_ou_trajectory(
    n_points: int,
    action_dim: int,
    dt: float,
    tau_n: float,
    sigma: float,
    key: jax.Array,
) -> jax.Array:
    """Sample one Ornstein-Uhlenbeck exploration path of shape (n_points, action_dim).

    Parameters
    ----------
    n_points
        Number of control steps in the episode (plus any buffer).
    action_dim
        Control dimension; each component is an independent OU process.
    dt
        Control step size (physical time between consecutive samples).
    tau_n
        OU correlation time (Doya 2000 uses 1.0). Adjacent-step correlation is exp(-dt/tau_n),
        so it is set by the ratio dt/tau_n and is independent of dt for fixed physical tau_n.
    sigma
        Stationary standard deviation of the process (the effective exploration scale).
    key
        JAX PRNG key, for reproducibility under the shared-seed policy.

    Returns
    -------
    A (n_points, action_dim) array; the path is started from its stationary distribution
    N(0, sigma^2), so it is stationary from the first step (no burn-in transient).
    """
    a = jnp.exp(-dt / tau_n)
    innovation_std = sigma * jnp.sqrt(jnp.maximum(1.0 - a * a, 0.0))
    key0, key_rest = jax.random.split(key)
    # Start from the stationary distribution so there is no initial-transient bias.
    n0 = sigma * jax.random.normal(key0, shape=(action_dim,))
    innovations = innovation_std * jax.random.normal(key_rest, shape=(n_points, action_dim))

    def step(prev, xi):
        cur = a * prev + xi
        return cur, cur

    _, path = jax.lax.scan(step, n0, innovations)
    return path
