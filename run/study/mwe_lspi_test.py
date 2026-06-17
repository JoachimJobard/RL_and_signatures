"""LSPI (Least-Squares Policy Iteration) as the STABLE realisation of the RL critic.

Online LSTD diverges (off-policy moving target + n<D). LSPI fixes both: freeze the policy,
collect transitions UNDER it from many short rollouts over diverse initial conditions
(on-policy, bounded, n>>D), LSTD-evaluate that one policy, THEN improve (greedy
value-gradient), repeat. Validated on the double integrator (mwe_lspi_markovian.py); this
generalises the test to any harness cell + representation.

  iteration k:
    D_k = { (phi_t, phi_{t+dt}, r_t) }  collected under the FIXED policy pi_k (diverse ICs)
    theta_{k+1} = -(M + reg I)^-1 b,   M = sum phi psi^T,  psi = (phi'-phi)/dt - phi/tau,  b = sum phi r
    pi_{k+1}(x) = 1/2 R^-1 B^T d/dx (theta_{k+1}^T phi)

Usage:
    uv run python run/study/mwe_lspi_test.py --cell linear_dde --rep signature [--tau 2.0]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))         # for the sibling harness module
from h1h2_evaluation import make_cell, label_fns, mg_linear_feedback, REPS   # noqa: E402

REG = 1e-3
N_ROLL = 20
N_ITER = 8


def make_feat(cell, rep_cfg):
    import jax
    from src.representations.factory import make_representation
    kw = {k: v for k, v in rep_cfg.items() if k != "kind"}
    rep = make_representation(rep_cfg["kind"], window_length=cell["window_length"],
                              n_state=cell["n"], **kw)
    return jax.jit(rep.feature_fn), int(rep.feature_dim)


def ic_sampler(cell, rng):
    if cell.get("nonlinear"):
        return np.array([rng.uniform(0.6, 1.35)])          # MG constant-history level
    return cell["x0"](rng)


def reference_control(cell):
    if cell.get("nonlinear"):
        return mg_linear_feedback(cell)
    _, us = label_fns(cell)
    return us


def rollout(cell, feat, control_fn, x0, tf, collect=False):
    """Roll the cell env under control_fn(window)->u. Returns closed-loop cost I, and (if collect)
    the TD transitions (phi_t, phi_next, r_t)."""
    import jax, jax.numpy as jnp
    from src.envs.env_rk_jax import JAXEnvWrapper
    from src.representations.factory import make_representation, RepresentationBuffer
    env, win, n, dt, clip = cell["env"], cell["window_length"], cell["n"], cell["dt"], cell["clip"]
    rep = make_representation("markovian", window_length=win, n_state=n, degree=1)
    buf = RepresentationBuffer(rep, window_length=win, n_state=n)
    w = JAXEnvWrapper(env, rng_key=42); w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    buf.reset()
    for _ in range(win):
        buf.append(np.asarray(x0, dtype=np.float32))
    P, PN, RW = [], [], []
    I, t = 0.0, 0.0
    W = np.array(buf.buffer.to_array()).reshape(win, n)
    while t < tf - 1e-9:
        u = np.clip(np.asarray(control_fn(W)).reshape(-1), -clip, clip)
        if not np.all(np.isfinite(u)) or abs(I) > 1e10:
            return (P, PN, RW, float("inf")) if collect else float("inf")
        t, x, r = w.step(w.state, jnp.array(u)); x = np.array(x).reshape(-1)
        I += -float(r) * dt
        buf.append(x.astype(np.float32))
        Wn = np.array(buf.buffer.to_array()).reshape(win, n)
        if collect:
            P.append(np.asarray(feat(jnp.asarray(W)))); PN.append(np.asarray(feat(jnp.asarray(Wn)))); RW.append(float(r))
        W = Wn
    return (P, PN, RW, I) if collect else I


def collect_dataset(cell, feat, control_fn, seed, t_collect, sigma=0.0):
    """Collect transitions under control_fn. sigma>0 adds exploration noise to the control during
    collection (the RL analog of off-sheet perturbation: it spreads the data OFF the greedy manifold
    so the value-gradient is constrained -- without it the signature gradient is under-determined)."""
    rng = np.random.default_rng(seed)
    m = cell["oracle"]["B"].shape[1]
    ctrl = control_fn
    if sigma > 0:
        ctrl = lambda W: np.asarray(control_fn(W)).reshape(-1) + sigma * rng.standard_normal(m)
    P, PN, RW = [], [], []
    for _ in range(N_ROLL):
        p, pn, rw, _ = rollout(cell, feat, ctrl, ic_sampler(cell, rng), t_collect, collect=True)
        P += p; PN += pn; RW += rw
    return np.array(P), np.array(PN), np.array(RW)


def lstd_solve(P, PN, RW, D, dt, tau, rank=0):
    """CENTRED LSTD (gauge fix: subtract the running feature mean -- removes the constant/time
    direction that dominates the high-dim Gram) with optional TRUNCATED-SVD: solve the fixed point
    in the top-`rank` principal subspace of the centred features, dropping the near-null directions
    that make the value-gradient under-determined (the platoon ker-G pathology)."""
    mu = P.mean(0)
    Pc = P - mu
    psi = (PN - P) / dt - Pc / tau
    M = Pc.T @ psi; b = Pc.T @ RW
    if 0 < rank < D:
        C = Pc.T @ Pc
        _, evecs = np.linalg.eigh((C + C.T) / 2.0)
        U = evecs[:, -rank:]                                   # top-`rank` principal directions
        Mk, bk = U.T @ M @ U, U.T @ b
        reg = REG * max(float(np.abs(np.diag(Mk)).max()), 1e-12)
        return U @ (-np.linalg.solve(Mk + reg * np.eye(rank), bk))
    return -np.linalg.solve(M + REG * np.eye(D), b)


def greedy(feat, theta, half_RinvBT):
    import jax, jax.numpy as jnp
    th = jnp.asarray(theta)
    Vg = jax.jit(jax.grad(lambda p: jnp.dot(th, feat(p))))
    return lambda W: half_RinvBT @ np.array(Vg(jnp.asarray(W)))[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--rep", default="signature", choices=list(REPS))
    ap.add_argument("--tau", type=float, default=2.0, help="discount time-constant for LSTD")
    ap.add_argument("--explore", type=float, default=0.0, help="exploration noise sigma during data collection")
    ap.add_argument("--damp", type=float, default=1.0,
                    help="critic damping alpha: theta <- (1-alpha) theta + alpha theta_lstd (1=full LSPI)")
    ap.add_argument("--n-iter", type=int, default=N_ITER)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cell = make_cell(args.cell)
    feat, D = make_feat(cell, REPS[args.rep])
    oc = cell["oracle"]; B, R = oc["B"], oc["R"]
    half_RinvBT = 0.5 * np.linalg.inv(R) @ B.T
    dt, tf = cell["dt"], cell["tf"]
    t_collect = tf if cell.get("nonlinear") else tf * 0.3        # MG bounded; short for unstable linear
    u_star = reference_control(cell)

    I_nc = rollout(cell, feat, lambda W: np.zeros(B.shape[1]), cell["x0c"], tf)
    I_or = rollout(cell, feat, u_star, cell["x0c"], tf)
    print(f"\nLSPI | cell='{args.cell}' rep='{args.rep}' D={D} | discount tau={args.tau} | t_collect={t_collect:.1f}")
    print(f"reference: no-control I={I_nc:.4f}   oracle-feedback I*={I_or:.4f}")
    print(f"\n{'iter':>5}{'n_trans':>9}{'n/D':>7}{'I_cl':>12}{'cos(u,u*)':>12}")

    # deployment windows for cos: roll under the oracle and keep the raw windows
    def deployment_windows():
        import jax, jax.numpy as jnp
        from src.envs.env_rk_jax import JAXEnvWrapper
        from src.representations.factory import make_representation, RepresentationBuffer
        env, win, n = cell["env"], cell["window_length"], cell["n"]
        rep = make_representation("markovian", window_length=win, n_state=n, degree=1)
        buf = RepresentationBuffer(rep, window_length=win, n_state=n)
        w = JAXEnvWrapper(env, rng_key=42); w.reset(jax.random.PRNGKey(0), x0=jnp.array(cell["x0c"]), t0=0.0)
        buf.reset()
        for _ in range(win):
            buf.append(np.asarray(cell["x0c"], dtype=np.float32))
        Ws, t = [], 0.0
        while t < tf - 1e-9:
            W = np.array(buf.buffer.to_array()).reshape(win, n); Ws.append(W)
            u = np.clip(np.asarray(u_star(W)).reshape(-1), -cell["clip"], cell["clip"])
            t, x, _ = w.step(w.state, jnp.array(u)); buf.append(np.array(x).reshape(-1).astype(np.float32))
        return Ws
    depW = deployment_windows()
    u_star_dep = np.array([np.asarray(u_star(W)).reshape(-1) for W in depW])

    theta = np.zeros(D)                                          # zero critic -> zero (no-control) policy
    policy = lambda W: np.zeros(B.shape[1])
    for it in range(args.n_iter):
        P, PN, RW = collect_dataset(cell, feat, policy, seed=it, t_collect=t_collect, sigma=args.explore)
        if len(P) == 0:
            print(f"{it:>5}  (no transitions — policy diverged data collection)"); break
        theta_lstd = lstd_solve(P, PN, RW, D, dt, args.tau)
        theta = (1.0 - args.damp) * theta + args.damp * theta_lstd      # damped (soft) policy-iteration step
        policy = greedy(feat, theta, half_RinvBT)
        I = rollout(cell, feat, policy, cell["x0c"], tf)
        u_rep = np.array([np.asarray(policy(W)).reshape(-1) for W in depW])
        cos = float(np.sum(u_rep * u_star_dep) / (np.linalg.norm(u_rep) * np.linalg.norm(u_star_dep) + 1e-12))
        print(f"{it:>5}{len(P):>9}{len(P)/D:>7.0f}{I:>12.4f}{cos:>12.3f}")


if __name__ == "__main__":
    main()
