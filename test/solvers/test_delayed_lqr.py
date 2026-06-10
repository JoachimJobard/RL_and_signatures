"""Tests for the augmented delayed-LQR oracle (Kolmanovskii 6.2, option A)."""

import numpy as np
from scipy.linalg import solve_discrete_are

from src.solvers.delayed_lqr import augmented_discrete_lqr, simulate_closed_loop_cost


def test_no_delay_reduces_to_ordinary_lqr():
    # As A1 -> 0 the gain on the current-state block equals the ordinary discrete
    # LQR gain and the lagged-block gains vanish.
    A = np.array([[-0.5]]); B = np.array([[1.0]]); A1 = np.array([[0.0]])
    Q = np.array([[1.0]]); R = np.array([[1.0]]); dt = 0.05
    lqr = augmented_discrete_lqr(A, A1, B, Q, R, delay=0.1, dt=dt)  # K = 2 taps

    Ad = np.eye(1) + dt * A
    Bd = dt * B
    P = solve_discrete_are(Ad, Bd, Q, R)
    k_ref = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)

    assert np.allclose(lqr.gain[:, :1], k_ref, atol=1e-6)
    assert np.allclose(lqr.gain[:, 1:], 0.0, atol=1e-6)


def test_closed_loop_is_schur_stable():
    A = np.array([[0.0]]); A1 = np.array([[0.3]]); B = np.array([[1.0]])
    Q = np.array([[1.0]]); R = np.array([[0.1]])
    lqr = augmented_discrete_lqr(A, A1, B, Q, R, delay=0.2, dt=0.05)
    assert lqr.closed_loop_spectral_radius() < 1.0


def test_delay_controller_uses_history():
    A = np.array([[0.0]]); A1 = np.array([[0.5]]); B = np.array([[1.0]])
    Q = np.array([[1.0]]); R = np.array([[0.1]])
    lqr = augmented_discrete_lqr(A, A1, B, Q, R, delay=0.2, dt=0.05)  # K = 4 taps
    assert lqr.gain.shape == (1, 5)
    # The feedback genuinely uses lagged states (non-trivial gain beyond block 0).
    assert np.max(np.abs(lqr.gain[:, 1:])) > 1e-3


def test_closed_loop_cost_finite_and_stabilising():
    A = np.array([[0.1]]); A1 = np.array([[0.2]]); B = np.array([[1.0]])
    Q = np.array([[1.0]]); R = np.array([[0.1]])
    lqr = augmented_discrete_lqr(A, A1, B, Q, R, delay=0.2, dt=0.05)
    cost = simulate_closed_loop_cost(
        lqr, A, A1, B, Q, R, x0=np.array([1.0]), n_steps=400,
    )
    assert np.isfinite(cost) and cost > 0.0
    assert lqr.closed_loop_spectral_radius() < 1.0


def test_multidim_shapes_and_control():
    A = np.array([[0.0, 1.0], [-1.0, -0.5]])
    A1 = 0.1 * np.eye(2)
    B = np.array([[0.0], [1.0]])
    Q = np.eye(2); R = np.array([[0.1]])
    lqr = augmented_discrete_lqr(A, A1, B, Q, R, delay=0.1, dt=0.05)  # K = 2 taps
    assert lqr.gain.shape == (1, 2 * 3)
    u = lqr.control(np.ones((3, 2)))  # window: 3 taps x 2 states
    assert u.shape == (1,)
