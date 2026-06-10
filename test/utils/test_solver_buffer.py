"""Tests for the delay-history ring buffer and its linear interpolation.

These pin the delay-DDE history retrieval that the RK4 integrator depends on
(code_review F-D2): integer delays return exact past samples and fractional
delays linearly interpolate between the two straddling samples.
"""

import jax.numpy as jnp

from src.utils.solver_buffer_jax import (
    BufferState,
    buffer_append,
    get_current,
    get_delayed_interpolated,
)


def _fill(values, capacity=16, dim=1):
    state = BufferState.create(capacity, dim)
    for v in values:
        state = buffer_append(state, jnp.array([float(v)]))
    return state


def test_get_current_returns_last_appended():
    state = _fill([0, 1, 2, 3, 4])
    assert jnp.allclose(get_current(state), jnp.array([4.0]))


def test_integer_delay_returns_exact_past_sample():
    state = _fill([0, 1, 2, 3, 4])
    # delay_steps=1 -> one step before the newest (which is 4) -> 3
    assert jnp.allclose(get_delayed_interpolated(state, 1.0), jnp.array([3.0]))
    # delay_steps=2 -> 2
    assert jnp.allclose(get_delayed_interpolated(state, 2.0), jnp.array([2.0]))


def test_fractional_delay_linearly_interpolates():
    state = _fill([0, 1, 2, 3, 4])
    # delay_steps=1.5 -> between sample 1-step back (3, weight 0.5) and 2-steps back (2, weight 0.5)
    assert jnp.allclose(get_delayed_interpolated(state, 1.5), jnp.array([2.5]))
    # delay_steps=0.25 -> between newest (4, weight 0.75) and 1-step back (3, weight 0.25)
    assert jnp.allclose(get_delayed_interpolated(state, 0.25), jnp.array([3.75]))


def test_zero_delay_returns_current():
    state = _fill([0, 1, 2, 3, 4])
    assert jnp.allclose(get_delayed_interpolated(state, 0.0), get_current(state))
