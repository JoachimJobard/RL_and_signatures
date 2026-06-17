"""Delay-coupled Hopfield network as the linear/nonlinear-in-delay H1/H2 testbed.

A continuous-time graded-response Hopfield / cellular neural network with a *delayed*
synaptic coupling (Marcus & Westervelt, "Stability of analog neural networks with
delay", Phys. Rev. A 39 (1989) 347):

    x_i'(t) = -x_i(t) + sum_j W_ij phi_eps( x_j(t - tau) ) + (B u)_i ,

with the tunable activation

    phi_eps(xi) = (1 - eps) * xi  +  eps * tanh(kappa * xi) / kappa ,      eps in [0, 1].

WHY THIS IS THE RIGHT TESTBED FOR H1 AND H2 (the eps knob).
The delay sits on the ENTIRE coupling term, so the present state does not summarise the
relevant past: the optimal controller is genuinely a functional of the history and H1
(markovian vs raw-history) holds. The single parameter eps then interpolates between two
control-theoretic regimes WITHOUT changing the linearisation:

* eps = 0  (LINEAR in the delay): phi_0(xi) = xi, so the plant is the linear delay
  system x' = -x + W x(t-tau) + Bu. The optimal value is exactly quadratic in the
  finite-dimensional method-of-steps augmented state, hence its gradient is LINEAR in the
  history window. The raw-history representation already spans that window, so it attains
  the delayed-LQR oracle gradient exactly and the signature can add only ill-conditioned
  directions: H2 is predicted to FAIL ("raw-history is optimal") -- a provable negative,
  not merely an empirical null.
* eps > 0  (NONLINEAR in the delay): the optimal value is a genuinely nonlinear
  functional of the history; raw-history (linear-in-window) under-fits it and the
  signature's nonlinear path features can help: H2 may HOLD.

H1 IS HELD FIXED ACROSS THE SWEEP. Because tanh'(0) = 1 (and tanh(kappa*0)/... has unit
slope at the origin for every kappa), phi_eps'(0) = 1 for ALL eps, so the linearisation at
the regulation setpoint x* = 0 is

    A = -I ,    A1 = W ,    B = B           (independent of eps, kappa).

The structural H1 effect (the delayed-LQR history-kernel ratio ||K_hist||/||K_now|| and
the markovian-oracle cost gap) is therefore identical for every eps; passing the
linearised delay-impact gate once (run/study/hopfield_delay_impact.py) certifies history
matters for the whole eps sweep, and the nonlinearity acts only at finite amplitude --
exactly where the signature's path features earn their keep. This factorises the design:
H1 is set by (W, tau); H2 is swept by eps.

The default coupling W is a scaled 2-D rotation (eigenvalues rho*exp(+/- i theta)); with
the unit leak -x and the delay tau this produces a delay-induced (Hopf) oscillatory mode,
so control is genuinely required and the history weight is substantial. The matrices
(A = -I, A1 = W, B) are the EXACT linearisation, so build_delayed_oracle / the delayed-LQR
is available as the analytic reference controller for the eps = 0 arm.
"""

import jax.numpy as jnp
import numpy as np

from src.envs.env_rk_jax import JAXDDEEnv
from src.utils.solver_buffer_jax import get_delayed_interpolated


def rotational_coupling(rho: float, theta: float) -> np.ndarray:
    """2x2 scaled-rotation coupling W = rho * [[cos, -sin], [sin, cos]] (eigenvalues
    rho*exp(+/- i theta)) -- a delayed rotational coupling that, against the unit leak,
    yields a delay-induced Hopf oscillation."""
    c, s = np.cos(theta), np.sin(theta)
    return rho * np.array([[c, -s], [s, c]])


class DelayedHopfieldNetwork(JAXDDEEnv):
    def __init__(
        self,
        W: np.ndarray | None = None,
        delay: float = 0.5,
        eps: float = 0.0,
        kappa: float = 1.0,
        nonlinearity: str = "tanh",
        damping: float = 0.0,
        B: np.ndarray | None = None,
        Q: jnp.ndarray | None = None,
        R: jnp.ndarray | None = None,
        step_size: float = 0.1,
        resolution: int = 4,
        x_target: float = 0.0,  # unused; equilibrium is the origin
    ):
        if W is None:
            W = rotational_coupling(rho=2.0, theta=np.pi / 2.0)   # antisymmetric, eig = +/- 2i
        W = np.asarray(W, dtype=float)
        n = W.shape[0]
        assert W.shape == (n, n), "W must be square"
        self.W = jnp.array(W)
        self.eps = float(eps)
        self.kappa = float(kappa)
        self.nonlinearity = str(nonlinearity)
        self.damping = float(damping)        # instantaneous cubic confinement -d x^3 (Duffing): bounds

        if B is None:
            B = np.eye(n)                                          # full, per-neuron control
        B = np.asarray(B, dtype=float)
        m = B.shape[1]
        A = -np.eye(n)                                             # unit leak -> A = -I
        if Q is None:
            Q = jnp.eye(n)
        if R is None:
            R = 0.1 * jnp.eye(m)
        super().__init__(
            A=A, B=B, A1=W, delay=jnp.array([delay]),
            Q=Q, R=R, step_size=step_size, resolution=resolution,
        )

    def _phi(self, xd):
        """Tunable activation, phi'(0) = 1 for every (eps, kappa) so the linearisation A1=W is fixed.
        'tanh'  : (1-eps) xd + eps tanh(kappa xd)/kappa  -- SATURATING (Hopfield); strong nonlinearity
                  saturates the delayed-state sensitivity (phi'->0), which erodes H1.
        'cubic' : xd + kappa xd^3  -- NON-SATURATING (Duffing/hardening); phi'=1+3 kappa xd^2 GROWS with
                  amplitude, so history sensitivity is preserved (H1 held) while the value is nonlinear
                  (H2). Not the literal Hopfield sigmoid (opposite cubic sign); a phi^4/bistable variant."""
        if self.nonlinearity == "cubic":
            return xd + self.kappa * xd ** 3
        return (1.0 - self.eps) * xd + self.eps * jnp.tanh(self.kappa * xd) / self.kappa

    def dynamics(self, x, buffer, u, dt_offset_fraction):
        x_delayed = x
        if self.has_delay:
            base_delay_steps = self.delay / self.solver_step_size
            adjusted_delay_steps = base_delay_steps - dt_offset_fraction
            x_delayed = get_delayed_interpolated(buffer, adjusted_delay_steps)
        return -x - self.damping * x ** 3 + self.W @ self._phi(x_delayed) + self.B @ u


# TESTING =========================================================================
if __name__ == "__main__":
    import jax
    from src.envs.env_rk_jax import JAXEnvWrapper

    for eps in (0.0, 1.0):
        env = DelayedHopfieldNetwork(delay=0.5, eps=eps, step_size=0.1, resolution=4)
        n, m = env.N, env.B.shape[1]
        print(f"\neps={eps}: state_dim={n} control_dim={m}  "
              f"eig(A1=W)={np.round(np.linalg.eigvals(np.array(env.W)), 3)}")
        x0 = np.array([0.5, 0.0])
        w = JAXEnvWrapper(env, rng_key=0)
        w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
        mx = 0.0
        for _ in range(150):                                       # tf = 15 at dt = 0.1
            _, x, _ = w.step(w.state, jnp.zeros(m))
            mx = max(mx, float(jnp.max(jnp.abs(x))))
        print(f"   open-loop max|x| over T=15 (zero control): {mx:.3f}  finite={np.isfinite(mx)}")
