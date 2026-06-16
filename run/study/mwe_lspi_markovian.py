"""Does LEAST-SQUARES POLICY ITERATION fix the online-LSTD divergence?

Online LSTD diverged even on the trivial markovian cell (double integrator): the data
distribution chases a policy that changes every step, so it is effectively off-policy and
the projected-Bellman fixed point is not stabilising (deadly triad).

LSPI decouples the two: hold the policy FIXED, collect transitions UNDER that fixed policy
(genuinely on-policy), LSTD-evaluate it on that batch (n >> D), THEN improve (greedy
value-gradient), and repeat. This restores (i) the on-policy norm-matching that makes the
LSTD operator a contraction and (ii) the n > D over-determination.

We run discounted LSPI on the double integrator and track the closed-loop cost per
iteration against no-control and the (undiscounted) LQR oracle.

Usage:
    uv run python run/study/mwe_lspi_markovian.py [--debug]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.solvers.continuous_oracle import build_markovian_oracle

# markovian cell = double integrator (the case that diverged under online LSTD)
A = np.array([[0.0, 1.0], [0.0, 0.0]]); B = np.array([[0.0], [1.0]])
Q = np.eye(2); R = np.array([[1.0]])
DT = 0.1; RESOLUTION = 4; WIN = 5; N = 2
TAU = 1.0           # discount time-constant (finite -> value finite even for a bad policy)
REG = 1e-3          # LSTD Tikhonov (minimal)
CLIP = 10.0
TF_EVAL = 15.0      # closed-loop evaluation horizon
T_COLLECT = 4.0     # short rollouts keep the data bounded for the unstable plant
N_ROLL = 25         # rollouts per iteration (n_transitions ~ N_ROLL * T_COLLECT/DT >> D)
N_ITER = 8
X0C = np.array([1.0, 0.5])


def make_env():
    from src.envs.env_rk_jax import JAXDDEEnv
    return JAXDDEEnv(A=A, B=B, A1=np.zeros((2, 2)), delay=None, Q=Q, R=R,
                     step_size=DT, resolution=RESOLUTION)


def make_feat():
    import jax
    from src.representations.factory import make_representation
    rep = make_representation("markovian", window_length=WIN, n_state=N, degree=2)
    return jax.jit(rep.feature_fn), int(rep.feature_dim)


def rollout(env, feat, control_fn, x0, tf, collect=False):
    """Roll the env under control_fn(window)->u. If collect, also return TD transitions
    (phi_t, phi_next, r_t); always return the closed-loop cost I."""
    import jax, jax.numpy as jnp
    from src.envs.env_rk_jax import JAXEnvWrapper
    from src.representations.factory import make_representation, RepresentationBuffer
    rep = make_representation("markovian", window_length=WIN, n_state=N, degree=1)
    buf = RepresentationBuffer(rep, window_length=WIN, n_state=N)
    w = JAXEnvWrapper(env, rng_key=42); w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    buf.reset()
    for _ in range(WIN):
        buf.append(np.asarray(x0, dtype=np.float32))
    phis, phins, rs = [], [], []
    I, t = 0.0, 0.0
    W = np.array(buf.buffer.to_array()).reshape(WIN, N)
    while t < tf - 1e-9:
        u = np.clip(np.asarray(control_fn(W)).reshape(-1), -CLIP, CLIP)
        if not np.all(np.isfinite(u)) or abs(I) > 1e10:
            return (phis, phins, rs, float("inf")) if collect else float("inf")
        t, x, r = w.step(w.state, jnp.array(u)); x = np.array(x).reshape(-1)
        I += -float(r) * DT
        buf.append(x.astype(np.float32))
        Wn = np.array(buf.buffer.to_array()).reshape(WIN, N)
        if collect:
            phis.append(np.asarray(feat(jnp.asarray(W))))
            phins.append(np.asarray(feat(jnp.asarray(Wn))))
            rs.append(float(r))
        W = Wn
    return (phis, phins, rs, I) if collect else I


def collect_dataset(env, feat, control_fn, seed):
    rng = np.random.default_rng(seed)
    P, PN, RW = [], [], []
    for _ in range(N_ROLL):
        x0 = rng.standard_normal(2) * np.array([1.0, 0.5])
        ph, pn, rw, _ = rollout(env, feat, control_fn, x0, T_COLLECT, collect=True)
        P += ph; PN += pn; RW += rw
    return np.array(P), np.array(PN), np.array(RW)


def lstd_solve(P, PN, RW, D):
    """Continuous-TD fixed point E[phi*delta]=0, delta = r + theta^T psi,
    psi = (phi_next - phi)/dt - phi/tau.  theta = -(M + reg I)^-1 b."""
    psi = (PN - P) / DT - P / TAU
    M = P.T @ psi
    b = P.T @ RW
    return -np.linalg.solve(M + REG * np.eye(D), b)


def greedy(feat, theta, half_RinvBT):
    import jax, jax.numpy as jnp
    th = jnp.asarray(theta)
    Vg = jax.jit(jax.grad(lambda p: jnp.dot(th, feat(p))))
    return lambda W: half_RinvBT @ np.array(Vg(jnp.asarray(W)))[-1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--debug", action="store_true")
    ap.parse_args()
    env = make_env(); feat, D = make_feat()
    oc = build_markovian_oracle(A, B, Q, R)
    K_or = oc["K"]; half_RinvBT = 0.5 * np.linalg.inv(R) @ B.T

    I_nc = rollout(env, feat, lambda W: np.zeros(1), X0C, TF_EVAL)
    I_or = rollout(env, feat, lambda W: -K_or @ W[-1], X0C, TF_EVAL)
    u_star = lambda W: -K_or @ W[-1]

    print(f"\nLSPI on the double-integrator markovian cell | D={D} | discount tau={TAU}")
    print(f"reference: no-control I={I_nc:.3f}   undiscounted-LQR oracle I*={I_or:.3f}")
    print(f"\n{'iter':>5}{'n_trans':>9}{'I_cl':>12}{'cos(u,u*)':>12}")

    policy = lambda W: np.zeros(1)            # start from the zero (no-control) policy
    for it in range(N_ITER):
        P, PN, RW = collect_dataset(env, feat, policy, seed=it)
        theta = lstd_solve(P, PN, RW, D)
        policy = greedy(feat, theta, half_RinvBT)
        # diagnostics
        dep = X0C
        I = rollout(env, feat, policy, dep, TF_EVAL)
        # cos vs oracle on a small state cloud
        rng = np.random.default_rng(123)
        us, uss = [], []
        for _ in range(60):
            W = np.tile((rng.standard_normal(2) * np.array([1.0, 0.5])).astype(np.float32), (WIN, 1))
            us.append(np.asarray(policy(W)).reshape(-1)); uss.append(np.asarray(u_star(W)).reshape(-1))
        us, uss = np.array(us), np.array(uss)
        cos = float(np.sum(us * uss) / (np.linalg.norm(us) * np.linalg.norm(uss) + 1e-12))
        print(f"{it:>5}{len(P):>9}{I:>12.4f}{cos:>12.3f}")

    print(f"\n(undiscounted oracle I*={I_or:.3f}; LSPI uses discount tau={TAU} so its fixed point "
          f"is the DISCOUNTED-optimal control, slightly above I*.)")


if __name__ == "__main__":
    main()
