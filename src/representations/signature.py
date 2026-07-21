"""Signature representation: depth-m signature of the Arribas-augmented path.

Wraps the existing ``SlidingSignatureJAX`` augmentation/compute (Arribas 2018,
Definition p.5) as a pure ``feature_fn(window) -> signature`` so it shares the
representation interface with the polynomial baselines. The signature features feed
a LINEAR functional, instantiating Arribas Thm 4.2 (linear functionals of the signature
are dense in the continuous functionals of the path).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpy.typing import DTypeLike

from src.utils.dynamic_signature import SlidingSignatureJAX


class SignatureRepresentation:
    """Phi(x_t) = S^depth( augmented path of the window )."""

    def __init__(
        self,
        depth: int,
        window_length: int,
        n_state: int,
        time_augmentation: bool = True,
        origin_augmentation: bool = True,
        bias: bool = False,
        dtype: DTypeLike = np.float64,
    ) -> None:
        # window_size is the buffer capacity minus one; here the window has
        # ``window_length`` samples, so window_size = window_length - 1.
        self._sig = SlidingSignatureJAX(
            depth=depth,
            window_size=window_length - 1,
            d=n_state,
            time_augmentation=time_augmentation,
            origin_augmentation=origin_augmentation,
            bias=bias,
            dtype=dtype,
        )
        self.depth = depth
        self.window_length = window_length
        self.n_state = n_state
        self.feature_dim = int(self._sig.signature_size)

    def feature_fn(self, window: jnp.ndarray) -> jnp.ndarray:
        # window is (window_length, n_state), chronological (current last) — exactly
        # the path the signature transform consumes (data[0] = earliest = basepoint).
        return self._sig._jit_compute_sig(window)
