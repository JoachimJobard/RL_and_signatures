"""Connected-cruise-control vehicle platoon as a controlled nonlinear delay system.

A chain of N vehicles following the Optimal-Velocity Model (Bando et al. 1995) with a
driver/communication reaction delay sigma, in the connected-cruise-control form of
Ge, Orosz & Stepan, "To Delay or Not to Delay -- Stability of Connected Cruise
Control" (in: Delay Systems, Springer 2016). Per vehicle i (following vehicle i-1):

    s_i' = v_i,
    v_i' = alpha [ V(h_i(t-sigma)) - v_i(t-sigma) ]
           + beta [ W(v_{i-1}(t-sigma)) - v_i(t-sigma) ] + u_i,

with headway h_i = s_{i-1} - s_i - l, the smooth range policy (Ge-Orosz-Stepan eq. 5)

    V(h) = 0                                           if h <= h_st,
           (v_max/2) (1 - cos(pi (h-h_st)/(h_go-h_st)))  if h_st < h < h_go,
           v_max                                       if h >= h_go,

and the saturation W(v) = min(v, v_max). Parameters (human-driver traffic data,
Ge-Orosz-Stepan): h_st=5 m, h_go=35 m, v_max=30 m/s; uniform-flow equilibrium
v*=15 m/s, h*=20 m, slope f* = V'(h*) = pi/2.

WHY THIS IS A GENUINE HISTORY-DEPENDENT CONTROL CELL (unlike the Dadebo CSTR).
The delay sits on EVERY feedback term: the entire restoring force in v_i' is delayed,
only the kinematics s_i' = v_i is instantaneous. So the present state does NOT
summarise the relevant past and the optimal controller is genuinely a functional of
the history. The linearise -> delayed-LQR gate (delayed_lqr on this env's A, A1, B)
gives a history-kernel ratio ||K_hist||/||K_now|| ~= 0.17 and a markovian-oracle cost
gap of +12.5% under a propagating (lead-vehicle) disturbance at the default regime
(N=5, alpha=1.0, beta=0.5, sigma=0.5) -- vs the Dadebo CSTR's 0.009 / +1%. The gap
grows with the delay sigma (to +43% at sigma=1.0, though the open loop then becomes
mildly unstable). The nonlinear range policy V(h) makes the optimal control a NONLINEAR
functional of the history, so this cell tests H2 as well.

State (deviations from the uniform-flow REFERENCE TRAJECTORY, equilibrium at the
origin):
    x = [ s_1, v_1, s_2, v_2, ..., s_N, v_N ]  in R^{2N},
    s_i = (position of vehicle i) - (its uniform-flow reference),  v_i = (velocity) - v*.
The cost penalises the position-tracking error s_i (string-stability / formation
keeping: cumulative spacing error, which is what makes history genuinely matter) and
the velocity error v_i. The headway deviation is dh_i = s_{i-1} - s_i. The leader of
vehicle 1 is a prescribed head travelling at the uniform-flow velocity (deviation 0),
so s_0 := 0, v_0 := 0 (open-chain configuration).

Control: the value-gradient agent commands an added acceleration u_i on every vehicle
(B is constant, full column rank N) -- the all-vehicles-controlled configuration.

The linear matrices (A, A1, B) passed to the base class are the EXACT linearisation of
the nonlinear dynamics() at the origin, so the delayed-LQR oracle (delayed_lqr_for_env)
is available as the analytic reference controller for this cell.
"""

import jax
import jax.numpy as jnp
import numpy as np

from src.envs.env_rk_jax import JAXDDEEnv
from src.utils.solver_buffer_jax import get_delayed_interpolated


class PlatoonEnv(JAXDDEEnv):
    def __init__(
        self,
        n_vehicles: int = 5,
        alpha: float = 1.0,
        beta: float = 0.5,
        delay: float = 0.5,
        v_star: float = 15.0,
        v_max: float = 30.0,
        h_st: float = 5.0,
        h_go: float = 35.0,
        Q: jnp.ndarray | None = None,
        R: jnp.ndarray | None = None,
        step_size: float = 0.05,
        resolution: int = 4,
        x_target: float = 0.0,  # unused; equilibrium is the origin in perturbation coords
    ):
        self.n_vehicles = int(n_vehicles)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.v_max = float(v_max)
        self.h_st = float(h_st)
        self.h_go = float(h_go)
        self.v_star = float(v_star)
        # Equilibrium headway and slope from the smooth range policy V(h*) = v*.
        self.h_star = h_st + (h_go - h_st) / np.pi * float(np.arccos(1.0 - 2.0 * v_star / v_max))
        self.f_star = (v_max / 2.0) * (np.pi / (h_go - h_st)) * float(
            np.sin(np.pi * (self.h_star - h_st) / (h_go - h_st)))

        n = 2 * self.n_vehicles
        m = self.n_vehicles
        A, A1, B = self._linearised_matrices()
        if Q is None:
            Q = jnp.eye(n)
        if R is None:
            R = 0.1 * jnp.eye(m)
        super().__init__(
            A=A, B=B, A1=A1, delay=jnp.array([delay]),
            Q=Q, R=R, step_size=step_size, resolution=resolution,
        )

    def _linearised_matrices(self):
        """Exact linearisation of :meth:`dynamics` at the origin (uniform flow), in
        position-deviation coordinates. Kinematics s_i' = v_i is instantaneous (-> A);
        the OVM restoring force is fully delayed (-> A1). Control enters v_i' (-> B)."""
        N = self.n_vehicles
        n = 2 * N
        A = np.zeros((n, n))
        A1 = np.zeros((n, n))
        B = np.zeros((n, N))
        a, b, fs = self.alpha, self.beta, self.f_star
        for i in range(N):
            isp, iv = 2 * i, 2 * i + 1
            A[isp, iv] = 1.0                       # s_i' = v_i
            A1[iv, isp] = -a * fs                  # -alpha f* s_i (delayed; from -V(h_i))
            A1[iv, iv] = -(a + b)                  # -(alpha+beta) v_i (delayed)
            if i >= 1:
                A1[iv, 2 * (i - 1)] = a * fs        # +alpha f* s_{i-1} (leader position)
                A1[iv, 2 * (i - 1) + 1] = b         # +beta v_{i-1} (leader velocity)
            B[iv, i] = 1.0                          # control acceleration on v_i
        return jnp.array(A), jnp.array(A1), jnp.array(B)

    def _range_policy(self, h):
        """Smooth optimal-velocity / range policy V(h) (Ge-Orosz-Stepan eq. 5)."""
        F = (self.v_max / 2.0) * (1.0 - jnp.cos(jnp.pi * (h - self.h_st) / (self.h_go - self.h_st)))
        return jnp.where(h <= self.h_st, 0.0, jnp.where(h >= self.h_go, self.v_max, F))

    def dynamics(self, x, buffer, u, dt_offset_fraction):
        N = self.n_vehicles
        # Delayed full state (all components delayed by sigma), RK4 sub-step aware.
        x_delayed = x
        if self.has_delay:
            base_delay_steps = self.delay / self.solver_step_size
            adjusted_delay_steps = base_delay_steps - dt_offset_fraction
            x_delayed = get_delayed_interpolated(buffer, adjusted_delay_steps)

        s, v = x[0::2], x[1::2]                      # position & velocity deviations
        s_d, v_d = x_delayed[0::2], x_delayed[1::2]  # delayed deviations
        # Leader (vehicle ahead) deviations; the head (vehicle 1's leader) is at the
        # reference, so s_lead[0] = v_lead[0] = 0.
        s_lead_d = jnp.concatenate([jnp.zeros(1), s_d[:-1]])
        v_lead_d = jnp.concatenate([jnp.zeros(1), v_d[:-1]])

        # Instantaneous kinematics: position-error rate = velocity error.
        s_dot = v
        # Delayed OVM restoring force + instantaneous control acceleration. The headway
        # deviation seen by vehicle i is the delayed s_{i-1} - s_i.
        headway = self.h_star + (s_lead_d - s_d)
        V_term = self._range_policy(headway)
        W_term = jnp.minimum(self.v_star + v_lead_d, self.v_max)
        v_dot = (self.alpha * (V_term - self.v_star - v_d)
                 + self.beta * (W_term - self.v_star - v_d) + u)

        return jnp.stack([s_dot, v_dot], axis=1).reshape(-1)


# TESTING =========================================================================
if __name__ == "__main__":
    env = PlatoonEnv(n_vehicles=5, delay=1.0)
    print(f"N={env.n_vehicles}  state_dim={env.N}  control_dim={env.B.shape[1]}")
    print(f"h*={env.h_star:.3f}  v*={env.v_star}  f*={env.f_star:.4f}  (expect h*=20, f*=pi/2={np.pi/2:.4f})")
    from src.envs.env_rk_jax import JAXEnvWrapper
    x0 = np.zeros(env.N)
    x0[1] = 2.0  # perturb vehicle 1 velocity by +2 m/s
    w = JAXEnvWrapper(env, rng_key=0)
    w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    mx = 0.0
    for _ in range(400):
        _, x, _ = w.step(w.state, jnp.zeros(env.n_vehicles))
        mx = max(mx, float(jnp.max(jnp.abs(x))))
    print(f"open-loop max|x| over T=20 (zero control): {mx:.3f}  finite={np.isfinite(mx)}")
