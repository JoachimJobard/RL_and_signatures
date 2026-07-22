"""Representation factory + a window buffer with a SlidingSignatureJAX-compatible
surface, so any representation drops into the value-gradient agent unchanged.

``make_representation`` selects the representation by kind; ``RepresentationBuffer``
holds the discretised history window and exposes the same attributes the agent uses
on ``SlidingSignatureJAX`` (``signature_size``, ``window_size``, ``buffer``,
``reset``, ``append``, ``current_signature``, ``compute_signature``,
``_jit_compute_sig``) — the last is the pure, differentiable feature map the
value-gradient control law differentiates.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import DTypeLike

from src.representations.base import Representation
from src.representations.polynomial import (
    MarkovianRepresentation,
    RawHistoryRepresentation,
)
from src.representations.signature import SignatureRepresentation
from src.utils.dynamic_signature import DequeBuffer


def signature_window_size(env) -> int:
    """Representation window (in control-cadence taps) that covers the plant's maximum delay:
    ``ceil(max_delay / step_size) + 3``, or ``10`` for a non-delayed plant. THE single source for
    the window auto-derivation. Used at config time (main_unified, so the logged window equals the
    one actually run) and by every learner's agent, so two learners never derive different windows
    -- the previous value-gradient(+3)/actor-critic(+1) split was a cross-learner confound at fixed
    representation."""
    max_delay = float(env.max_delay) if getattr(env, "max_delay", None) is not None else 0.0
    return int(np.ceil(max_delay / float(env.step_size))) + 3 if max_delay > 0 else 10


def make_representation(
    kind: str,
    window_length: int,
    n_state: int,
    depth: int = 2,
    degree: int = 2,
    time_augmentation: bool = True,
    origin_augmentation: bool = True,
    bias: bool = False,
    dtype: DTypeLike = np.float64,
) -> Representation:
    """Build a representation by name: ``signature`` (depth-m), ``raw_history``
    (degree-m polynomial of the window), or ``markovian`` (degree-m polynomial of
    the current state)."""
    if kind == "signature":
        return SignatureRepresentation(
            depth=depth, window_length=window_length, n_state=n_state,
            time_augmentation=time_augmentation, origin_augmentation=origin_augmentation,
            bias=bias, dtype=dtype,
        )
    if kind == "raw_history":
        return RawHistoryRepresentation(window_length=window_length, n_state=n_state, degree=degree)
    if kind == "markovian":
        return MarkovianRepresentation(n_state=n_state, degree=degree)
    raise ValueError(f"Unknown representation kind: {kind!r} "
                     "(expected 'signature', 'raw_history', or 'markovian').")


class RepresentationBuffer:
    """Discretised-history window buffer + a :class:`Representation`, exposing the
    ``SlidingSignatureJAX`` surface the value-gradient agent depends on."""

    def __init__(
        self,
        representation: Representation,
        window_length: int,
        n_state: int,
        dtype: DTypeLike = np.float64,
        actor_representation: Representation | None = None,
    ) -> None:
        self.representation = representation
        self.window_length = window_length
        self.window_size = window_length - 1  # capacity - 1, matching SlidingSignatureJAX
        self.signature_size = int(representation.feature_dim)
        self.n_state = n_state
        self.dtype = dtype
        self.buffer = DequeBuffer(size=window_length, dtype=dtype)
        # Pure, differentiable feature map (jitted) — what the control law differentiates.
        self._jit_compute_sig = jax.jit(representation.feature_fn)
        # Optional SEPARATE actor feature map on the SAME window. The actor network eats the raw
        # path/state directly (no polynomial/signature lift): the lift is a value-function device
        # for the linear-in-features critic, whereas the linear control on a linear plant is a
        # linear functional of the path. None => the actor reuses the critic's feature map.
        self.actor_representation = actor_representation
        if actor_representation is not None:
            self.actor_feature_dim = int(actor_representation.feature_dim)
            self._jit_compute_actor_features = jax.jit(actor_representation.feature_fn)
        else:
            self.actor_feature_dim = self.signature_size
            self._jit_compute_actor_features = self._jit_compute_sig
        self._empty_sig = jnp.zeros(self.signature_size)
        self._current_signature = self._empty_sig
        self._signature_dirty = True

    def reset(self, prefill_zeros: bool = True) -> None:
        self.buffer = DequeBuffer(size=self.window_length, dtype=self.dtype)
        if prefill_zeros:
            zero = np.zeros(self.n_state, dtype=self.dtype)
            for _ in range(self.window_length):
                self.buffer.append(zero)
        self._signature_dirty = True

    def append(self, item: np.ndarray | jax.Array) -> None:
        self.buffer.append(np.asarray(item, dtype=self.dtype).reshape(self.n_state))
        self._signature_dirty = True

    @property
    def current_signature(self) -> jnp.ndarray:
        if self._signature_dirty:
            self._current_signature = self.compute_signature()
            self._signature_dirty = False
        return self._current_signature

    @current_signature.setter
    def current_signature(self, value: jnp.ndarray) -> None:
        self._current_signature = value
        self._signature_dirty = False

    @property
    def current_actor_features(self) -> jnp.ndarray:
        """Actor-side features from the SAME window. With a separate ``actor_representation`` (the
        'raw'/no-feature-map actor) this is the raw path/state fed straight to the actor network;
        otherwise it equals :attr:`current_signature`."""
        if self.actor_representation is None:
            return self.current_signature
        if len(self.buffer) < 1:
            return jnp.zeros(self.actor_feature_dim)
        return self._jit_compute_actor_features(self._full_window())

    def _full_window(self) -> jnp.ndarray:
        """The window as a fixed-size ``(window_length, n_state)`` array (front-padded
        with the oldest sample if the buffer is not yet full — needed because the
        polynomial features require a fixed input dimension)."""
        data = np.asarray(self.buffer.to_array(), dtype=self.dtype).reshape(-1, self.n_state)
        if data.shape[0] < self.window_length:
            pad = np.repeat(data[:1], self.window_length - data.shape[0], axis=0)
            data = np.vstack([pad, data])
        return jnp.asarray(data)

    def compute_signature(self) -> jnp.ndarray:
        if len(self.buffer) < 1:
            return self._empty_sig
        return self._jit_compute_sig(self._full_window())
