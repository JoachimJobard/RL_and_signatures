"""Tests for build_adam (optional global-norm gradient clipping, off by default)."""

import jax.numpy as jnp
import optax

from src.utils.optim import build_adam


def _apply_once(tx, params, grads):
    state = tx.init(params)
    updates, _ = tx.update(grads, state, params)
    return updates


def test_no_clip_matches_plain_adam():
    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    grads = {"w": jnp.array([10.0, -20.0, 30.0])}
    got = _apply_once(build_adam(0.1), params, grads)
    ref = _apply_once(optax.adam(0.1), params, grads)
    assert jnp.allclose(got["w"], ref["w"])


def test_none_and_nonpositive_disable_clipping():
    params = {"w": jnp.array([1.0])}
    grads = {"w": jnp.array([100.0])}
    ref = _apply_once(optax.adam(0.1), params, grads)
    for clip in (None, 0.0, -1.0):
        got = _apply_once(build_adam(0.1, clip_gradient=clip), params, grads)
        assert jnp.allclose(got["w"], ref["w"])


def test_positive_clip_matches_global_norm_chain():
    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    grads = {"w": jnp.array([10.0, -20.0, 30.0])}
    got = _apply_once(build_adam(0.1, clip_gradient=2.0), params, grads)
    ref = _apply_once(
        optax.chain(optax.clip_by_global_norm(2.0), optax.adam(0.1)),
        params, grads,
    )
    assert jnp.allclose(got["w"], ref["w"])


def test_clip_actually_bounds_global_norm():
    # The clip transform alone must bound the global norm to the requested value.
    clip = optax.clip_by_global_norm(2.0)
    grads = {"w": jnp.array([30.0, 40.0])}  # norm 50
    state = clip.init(grads)
    clipped, _ = clip.update(grads, state)
    norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in clipped.values()))
    assert float(norm) <= 2.0 + 1e-6
