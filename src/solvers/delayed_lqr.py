"""Analytic delayed-LQR oracle via the augmented finite-dimensional LQR.

For a linear delay differential equation with quadratic cost,

    x_dot(t) = A x(t) + A1 x(t - tau) + B u(t),
    J = integral_0^inf ( x' Q x + u' R u ) dt,

Kolmanovskii & Myshkis (1992), §6.2, show the optimal control is a LINEAR
functional of the history (eq. 2.7),

    u*(t) = -N1^{-1} B' [ P x(t) + integral_{-tau}^0 Q(theta) x(t+theta) dtheta ].

This module computes the *discretised* version of that controller by the method
of steps: discretise the history on the control grid with K = round(tau/dt) taps,
stack the augmented state xi_k = [x_k, x_{k-1}, ..., x_{k-K}], write the linear
shift-register map xi_{k+1} = Aaug xi_k + Baug u_k, and solve the standard
discrete-time algebraic Riccati equation. The resulting feedback gain is, by
construction, a linear functional of the discretised history, and it is the
*optimal linear-on-raw-history controller* for the discretised plant — hence both
the performance-ceiling oracle for the linear delayed cell and the fairness anchor
of the representation comparison (see documents/methodology/experimental_design.md).

The discretisation is forward Euler (first order in dt); the controller converges
to the continuous optimum as dt -> 0 (the grid-convergence test). As A1 -> 0 the
gain reduces to the ordinary LQR on the current state (the no-delay-limit test).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are


@dataclass
class DelayedLQR:
    """Result of the augmented delayed-LQR synthesis.

    Attributes:
        gain: feedback gain ``K``, shape ``(m, n*(K_taps+1))``. The optimal control
            is ``u = -gain @ xi`` where ``xi = [x_k, x_{k-1}, ..., x_{k-K_taps}]``
            (current state first), a linear functional of the discretised history.
        n_state: state dimension ``n``.
        n_control: control dimension ``m``.
        k_taps: number of delay taps ``K = round(tau/dt)``.
        dt: control time step.
        Aaug: augmented closed-loop-free dynamics matrix.
        Baug: augmented input matrix.
        P: discrete Riccati solution for the augmented system.
    """
    gain: np.ndarray
    n_state: int
    n_control: int
    k_taps: int
    dt: float
    Aaug: np.ndarray
    Baug: np.ndarray
    P: np.ndarray

    def control(self, history_window: np.ndarray) -> np.ndarray:
        """Optimal control for a history window ``[x(t), x(t-dt), ..., x(t-K dt)]``.

        Args:
            history_window: shape ``(K_taps+1, n)``, newest state first.

        Returns:
            The control ``u = -gain @ xi`` of shape ``(m,)``.
        """
        xi = np.asarray(history_window, dtype=float).reshape(-1)
        return -self.gain @ xi

    def closed_loop_spectral_radius(self) -> float:
        """Spectral radius of the closed-loop augmented system ``Aaug - Baug @ gain``
        (must be < 1 for a stabilising controller)."""
        eig = np.linalg.eigvals(self.Aaug - self.Baug @ self.gain)
        return float(np.max(np.abs(eig)))


def augmented_discrete_lqr(
    A: np.ndarray,
    A1: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    delay: float,
    dt: float,
) -> DelayedLQR:
    """Synthesise the delayed-LQR feedback via the augmented finite-dimensional LQR.

    Args:
        A, A1, B: system matrices ``x_dot = A x + A1 x(t-delay) + B u``.
        Q, R: quadratic-cost weights (``Q`` symmetric PSD, ``R`` symmetric PD).
        delay: time lag ``tau >= 0`` (``delay = 0`` gives ordinary LQR).
        dt: control time step; ``K = round(delay/dt)`` history taps are used.

    Returns:
        A :class:`DelayedLQR` with the feedback gain and the augmented system.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    A1 = np.atleast_2d(np.asarray(A1, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))

    n = A.shape[0]
    m = B.shape[1]
    k_taps = int(round(float(delay) / dt)) if delay and delay > 0 else 0
    n_blocks = k_taps + 1
    big = n * n_blocks

    # Forward-Euler local discretisation of the non-delay dynamics.
    Ad = np.eye(n) + dt * A
    A1d = dt * A1
    Bd = dt * B

    # Augmented shift-register dynamics: xi_{k+1} = Aaug xi_k + Baug u_k,
    # xi = [x_k, x_{k-1}, ..., x_{k-K}].
    Aaug = np.zeros((big, big))
    Baug = np.zeros((big, m))
    Aaug[0:n, 0:n] = Ad
    if k_taps > 0:
        Aaug[0:n, n * k_taps:n * (k_taps + 1)] = A1d
    Baug[0:n, :] = Bd
    for j in range(1, n_blocks):
        Aaug[n * j:n * (j + 1), n * (j - 1):n * j] = np.eye(n)

    # Cost penalises only the current-state block.
    Qaug = np.zeros((big, big))
    Qaug[0:n, 0:n] = Q

    P = solve_discrete_are(Aaug, Baug, Qaug, R)
    gain = np.linalg.solve(R + Baug.T @ P @ Baug, Baug.T @ P @ Aaug)

    return DelayedLQR(
        gain=gain, n_state=n, n_control=m, k_taps=k_taps, dt=float(dt),
        Aaug=Aaug, Baug=Baug, P=P,
    )


def simulate_closed_loop_cost(
    lqr: DelayedLQR,
    A: np.ndarray,
    A1: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    n_steps: int,
) -> float:
    """Integrated cost of the discretised closed loop from a constant initial
    history ``x(theta) = x0``, used to test the oracle (grid convergence) and as
    the achievable ceiling for the linear delayed cell."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    A1 = np.atleast_2d(np.asarray(A1, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))
    n, dt, K = lqr.n_state, lqr.dt, lqr.k_taps
    x0 = np.asarray(x0, dtype=float).reshape(n)

    # Initial history window (newest first), constant x0.
    window = np.tile(x0, (K + 1, 1))
    total = 0.0
    for _ in range(n_steps):
        u = lqr.control(window)
        x = window[0]
        x_delayed = window[K]
        total += float((x @ Q @ x + u @ R @ u) * dt)
        x_next = x + dt * (A @ x + A1 @ x_delayed + B @ u)
        window = np.vstack([x_next, window[:-1]])
    return total
