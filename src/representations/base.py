"""Representation interface for the history-dependent control study.

A *representation* maps a windowed state history to a feature vector that a LINEAR
functional (critic / actor) consumes. Keeping that functional linear is what ties the
study to the Arribas linear-approximation theorem (Thm 4.2) and isolates the effect
of the representation (see documents/methodology/experimental_design.md).

The single primitive every representation exposes is a **pure, JAX-differentiable**
``feature_fn(window) -> features``:

- ``window`` has shape ``(L, n)``: the discretised history at the control cadence,
  **chronological with the current state last** (``window[-1] = x(t)``). This
  convention lets the value-gradient agent obtain the control-law gradient
  ``dV/dx(t)`` uniformly as ``d feature_fn / d window[-1]`` for ANY representation.
- ``features`` has shape ``(feature_dim,)``.

Implementations: ``MarkovianRepresentation`` (current state only),
``RawHistoryRepresentation`` (polynomial monomials of the flattened window), and
``SignatureRepresentation`` (depth-m signature of the Arribas-augmented path).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax.numpy as jnp


@runtime_checkable
class Representation(Protocol):
    """A feature map from a windowed history to a feature vector for a linear functional."""

    #: Dimension of the feature vector returned by ``feature_fn``.
    feature_dim: int

    def feature_fn(self, window: jnp.ndarray) -> jnp.ndarray:
        """Map a history window ``(L, n)`` (current state last) to ``(feature_dim,)``."""
        ...
