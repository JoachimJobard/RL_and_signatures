"""Optimizer construction helpers shared across agents."""

import jax.numpy as jnp
import optax


def robbins_monro_schedule(base_lr: float, decay_power: float):
    """Return a Robbins-Monro learning-rate schedule alpha_k = base_lr / (1 + k)^p.

    k is the optimiser's own update count. The Robbins-Monro conditions for stochastic-
    approximation convergence are sum_k alpha_k = infinity and sum_k alpha_k^2 < infinity, satisfied
    exactly for p in (1/2, 1]; the LARGEST-step admissible schedule is p -> 1/2^+ (slowest decay).
    A constant learning rate (decay_power = 0) fails the second condition and converges only to a
    noise ball of radius proportional to the gradient variance, not to the optimum.

    NOTE on cadence: k counts THIS optimiser's updates. For the averaged actor-critic and the
    policy gradient the actor optimiser steps exactly once per episode, so this is a per-EPISODE
    schedule. A per-step optimiser (the critic) would decay per step, which over ~1e6 steps is far
    more aggressive -- hence the critic is left constant by default and only the actor is scheduled.
    """
    p = float(decay_power)
    lr0 = float(base_lr)
    return lambda count: lr0 / jnp.power(1.0 + count, p)


def build_adam(learning_rate, clip_gradient=None, decay_power=0.0, **adam_kwargs):
    """Build an Adam optimizer with optional global-norm gradient clipping.

    Gradient clipping is opt-in and disabled by default. When ``clip_gradient``
    is a positive, finite number, gradients are rescaled so that their global
    L2 norm does not exceed ``clip_gradient`` *before* the Adam update; this is
    direction-preserving. When ``clip_gradient`` is ``None`` or non-positive, no
    clipping is applied.

    This replaces the previous per-element clipping (``jnp.clip(g, -c, c)``),
    which distorted the gradient direction and was applied with a hard-coded
    bound that ignored the configured value.

    Args:
        learning_rate: Adam learning rate.
        clip_gradient: Maximum global gradient norm, or ``None`` / non-positive
            to disable clipping (the default).
        **adam_kwargs: Forwarded to ``optax.adam`` (e.g. ``b1``, ``b2``).

    Returns:
        An ``optax.GradientTransformation``.
    """
    lr = robbins_monro_schedule(learning_rate, decay_power) if (decay_power and decay_power > 0) \
        else learning_rate
    adam = optax.adam(lr, **adam_kwargs)
    if clip_gradient is not None and clip_gradient > 0:
        return optax.chain(optax.clip_by_global_norm(clip_gradient), adam)
    return adam
