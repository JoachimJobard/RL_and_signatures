"""Tests for evaluation helpers."""

import numpy as np

from src.training.evaluate import conform_initial_state


def test_conform_truncates_when_longer():
    # 2D default [1, 1] on a 1D env -> [1]
    out = conform_initial_state([1.0, 1.0], 1)
    assert out.shape == (1,)
    assert np.allclose(out, [1.0])


def test_conform_tiles_when_shorter():
    # 1D [0.8] on a 2D env -> [0.8, 0.8]
    out = conform_initial_state([0.8], 2)
    assert out.shape == (2,)
    assert np.allclose(out, [0.8, 0.8])


def test_conform_unchanged_when_matching():
    out = conform_initial_state([1.0, 2.0], 2)
    assert np.allclose(out, [1.0, 2.0])


def test_conform_noop_when_env_dim_none():
    out = conform_initial_state([1.0, 2.0, 3.0], None)
    assert np.allclose(out, [1.0, 2.0, 3.0])


def test_conform_accepts_scalar_and_array():
    assert conform_initial_state(0.5, 1).shape == (1,)
    assert conform_initial_state(np.array([1.0, 2.0]), 1).shape == (1,)
