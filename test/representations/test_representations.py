"""Tests for the representation interface and its three implementations."""

import jax
import jax.numpy as jnp
import numpy as np

from src.representations.base import Representation
from src.representations.polynomial import (
    MarkovianRepresentation,
    RawHistoryRepresentation,
)
from src.representations.signature import SignatureRepresentation


def _window(L: int, n: int) -> jnp.ndarray:
    return jnp.asarray(np.random.default_rng(0).standard_normal((L, n)))


def test_all_satisfy_protocol():
    for rep in (MarkovianRepresentation(2, 2),
                RawHistoryRepresentation(4, 1, 2),
                SignatureRepresentation(depth=2, window_length=5, n_state=1)):
        assert isinstance(rep, Representation)


def test_markovian_degree2_dim_and_uses_current_state_only():
    rep = MarkovianRepresentation(n_state=2, degree=2)
    # monomials of [x0, x1] up to degree 2: x0, x1, x0^2, x0x1, x1^2 -> 5
    assert rep.feature_dim == 5
    w = jnp.array([[9.0, 9.0], [2.0, 3.0]])  # current state is the LAST row
    feats = np.asarray(rep.feature_fn(w))
    assert np.allclose(feats, [2.0, 3.0, 4.0, 6.0, 9.0])  # uses [2,3] only, not [9,9]


def test_raw_history_degree1_is_flat_window():
    rep = RawHistoryRepresentation(window_length=3, n_state=2, degree=1)
    assert rep.feature_dim == 6  # 3 taps * 2 states
    w = jnp.arange(6.0).reshape(3, 2)
    assert np.allclose(np.asarray(rep.feature_fn(w)), np.arange(6.0))


def test_raw_history_degree2_dim():
    L, n, d = 2, 2, 2  # input_dim = L*n = 4
    rep = RawHistoryRepresentation(window_length=L, n_state=n, degree=d)
    # degree1: 4, degree2: C(4+1,2) = 10  -> 14
    assert rep.feature_dim == 4 + 10


def test_signature_feature_dim_matches_formula():
    rep = SignatureRepresentation(depth=2, window_length=5, n_state=1,
                                  time_augmentation=True, origin_augmentation=True)
    d_eff = 1 + 1 + 1  # n + time(1) + origin(n)
    assert rep.feature_dim == d_eff + d_eff ** 2


def test_value_gradient_prerequisite_dPhi_dcurrent_is_finite_and_nonzero():
    # The value-gradient control u ~ R^-1 B' dV/dx(t) needs dPhi/dwindow[-1] to exist,
    # be finite, AND be non-zero (else the current state cannot move the value).
    for rep, L in ((MarkovianRepresentation(1, 2), 4),
                   (RawHistoryRepresentation(4, 1, 2), 4),
                   (SignatureRepresentation(depth=2, window_length=5, n_state=1), 5)):
        w = _window(L, 1)
        J = np.asarray(jax.jacobian(rep.feature_fn)(w))  # (feature_dim, L, n)
        dphi_dcurrent = J[:, -1, :]
        assert np.all(np.isfinite(dphi_dcurrent))
        assert np.linalg.norm(dphi_dcurrent) > 1e-8


def test_markovian_is_insensitive_to_history():
    # A built-in check that the Markovian representation really is Markovian:
    # its features do not depend on any lagged state.
    rep = MarkovianRepresentation(1, 2)
    w = _window(4, 1)
    J = np.asarray(jax.jacobian(rep.feature_fn)(w))
    assert np.allclose(J[:, :-1, :], 0.0)  # zero gradient w.r.t. all earlier taps


def test_feature_fns_are_differentiable_wrt_current_state():
    # The value-gradient control needs d feature_fn / d window[-1] to exist.
    for rep in (MarkovianRepresentation(1, 2),
                RawHistoryRepresentation(4, 1, 2),
                SignatureRepresentation(depth=2, window_length=5, n_state=1)):
        w = _window(rep.__dict__.get("window_length", 4) if hasattr(rep, "window_length") else 4, 1)
        # scalar function of the window -> gradient w.r.t. the whole window is finite
        g = jax.grad(lambda win: jnp.sum(rep.feature_fn(win)))(w)
        assert np.all(np.isfinite(np.asarray(g)))
