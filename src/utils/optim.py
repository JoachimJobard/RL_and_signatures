"""Optimizer construction helpers shared across agents."""

import optax


def build_adam(learning_rate, clip_gradient=None, **adam_kwargs):
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
    adam = optax.adam(learning_rate, **adam_kwargs)
    if clip_gradient is not None and clip_gradient > 0:
        return optax.chain(optax.clip_by_global_norm(clip_gradient), adam)
    return adam
