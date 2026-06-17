"""H1/H2 evaluation harness (LINEAR cells): markovian, linear-DDE, platoon-linearised.

For one cell it builds the gated continuous-time oracle, generates on+off-manifold data with
n_train > feature_dim, labels every window analytically with (V*, u*), fits the markovian /
raw-history / signature critics by PLAIN least-squares to V*, and reports the three axes:
value-fit R^2 (expressivity), gradient cos(u_theta, u*) (extraction), closed-loop I (control).

  H1 := raw-history vs markovian      (does history help?)
  H2 := signature  vs raw-history     (does the signature structure help, natural config?)

The cells populate the falsifiability grid:
  markovian   -> H1 should FAIL (history irrelevant; V* depends on x(t) only)
  linear_dde  -> H1 should HOLD, H2 should FAIL (history matters; V* is exactly quadratic)
  platoon     -> H1 HOLD, H2 FAIL (nonlinear plant, linear-delay oracle)

Array-ready: one task per (--cell, --rep, --seed) writes a per-task summary; --rep all runs the
three representations together and prints the H1/H2 read-out for local validation.

Usage:
    uv run python run/study/h1h2_evaluation.py --cell linear_dde --rep all --seed 0 --debug
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.solvers.continuous_oracle import (
    build_markovian_oracle, build_delayed_oracle, z_from_window, doya_gate)

# ----- representations under test (natural config) -----
REPS = {
    "markovian":   dict(kind="markovian",   degree=2),
    "raw_history": dict(kind="raw_history", degree=2),
    "signature":   dict(kind="signature",   depth=2),
}
# UNIFORM data-sampling parameters (identical procedure across every environment):
#   on-sheet    = windows along the optimal/oracle trajectory from a set of initial conditions;
#   off-sheet   = perturb an on-sheet window by EPS and RE-LABEL it through the cell's own oracle
#                 (analytic V* for the linear/linearised cells, a Pontryagin BVP for Mackey-Glass).
# Same EPS and same off-sheet count N_OFF everywhere; only the oracle's label function and the
# trajectory integrator are cell-specific (intrinsic to the env), never the sampling logic.
N_IC = 12          # oracle-trajectory initial conditions (linear/linearised cells)
N_IC_MG = 30       # constant-history initial conditions (MG cells; more, for n>d at win~25)
SUBSAMPLE = 1      # keep every window of each trajectory
N_OFF = 500        # off-sheet windows (uniform count; instant for linear cells, 1 BVP each for MG)
EPS = 0.25         # off-sheet window perturbation (the state is O(1) in every cell)


# ============================== cell registry ==============================
def make_cell(name):
    """Return a cell: env, gated oracle, window/cadence, x0 sampler, deployment IC."""
    from src.envs.env_rk_jax import JAXDDEEnv
    if name == "markovian":
        A0 = np.array([[0.0, 1.0], [0.0, 0.0]]); A1 = np.zeros((2, 2)); B = np.array([[0.0], [1.0]])
        Q = np.eye(2); R = np.array([[1.0]]); dt = 0.1
        env = JAXDDEEnv(A=A0, B=B, A1=A1, delay=None, Q=Q, R=R, step_size=dt, resolution=4)
        oracle = build_markovian_oracle(A0, B, Q, R)
        return dict(name=name, env=env, oracle=oracle, n=2, dt=dt, window_length=5, tf=15.0,
                    clip=10.0, x0=lambda rng: rng.standard_normal(2) * np.array([1.0, 0.5]),
                    x0c=np.array([1.0, 0.5]))
    if name == "linear_dde":
        A0 = np.array([[0.0]]); A1 = np.array([[-1.5]]); B = np.array([[1.0]])
        Q = np.array([[1.0]]); R = np.array([[0.1]]); tau = 1.0; dt = 0.2
        env = JAXDDEEnv(A=A0, B=B, A1=A1, delay=np.array([tau]), Q=Q, R=R, step_size=dt, resolution=4)
        oracle = build_delayed_oracle(A0, A1, B, Q, R, tau, N=16)
        win = int(round(tau / dt)) + 1
        return dict(name=name, env=env, oracle=oracle, n=1, dt=dt, window_length=win, tf=20.0,
                    clip=10.0, x0=lambda rng: rng.standard_normal(1) * 0.8, x0c=np.array([0.8]))
    if name == "platoon":
        from src.envs.platoon import PlatoonEnv
        env = PlatoonEnv(n_vehicles=5, alpha=3.0, beta=1.0, delay=0.2, step_size=0.05, resolution=4)
        A0, A1, B = np.array(env.A), np.array(env.A1), np.array(env.B)
        Q, R = np.array(env.Q), np.array(env.R); tau = float(env.max_delay); dt = env.step_size
        oracle = build_delayed_oracle(A0, A1, B, Q, R, tau, N=12)
        x0c = np.zeros(env.N); x0c[1] = 0.5
        def x0s(rng):
            z = np.zeros(env.N); z[1::2] = 0.4 * rng.standard_normal(env.N // 2); return z
        return dict(name=name, env=env, oracle=oracle, n=env.N, dt=dt, window_length=5, tf=15.0,
                    clip=3.0, x0=x0s, x0c=x0c)
    if name == "hopfield_linear":
        # LINEAR-in-delay arm (eps=0): x' = -x + W x(t-tau) + Bu. Delayed-LQR is exact;
        # the optimal gradient is linear in the history window, so raw-history is optimal
        # and H2 is predicted to FAIL (provable negative). Gate: kernel ratio 0.30, H1 gap
        # +55% at tau=0.5 (run/study/hopfield_delay_impact.py) -- a genuine H1 cell.
        from src.envs.delayed_hopfield_network import DelayedHopfieldNetwork, rotational_coupling
        W = rotational_coupling(rho=2.0, theta=np.pi / 2.0); tau = 0.5; dt = 0.1
        env = DelayedHopfieldNetwork(W=W, delay=tau, eps=0.0, step_size=dt, resolution=4)
        A0, A1, B = np.array(env.A), np.array(env.A1), np.array(env.B)
        Q, R = np.array(env.Q), np.array(env.R)
        oracle = build_delayed_oracle(A0, A1, B, Q, R, tau, N=12)
        win = int(round(tau / dt)) + 1                              # >= tau/dt+1 so raw-history spans the augmented state
        return dict(name=name, env=env, oracle=oracle, n=2, dt=dt, window_length=win, tf=15.0,
                    clip=10.0, x0=lambda rng: rng.standard_normal(2) * 0.4,
                    x0c=np.array([0.5, 0.0]))
    if name in ("hopfield_nonlinear", "hopfield_duffing"):
        # NONLINEAR-in-delay arm: x' = -x - d x^3 + W phi(x(t-tau)) + Bu. SAME linearisation
        # (A=-I, A1=W) as hopfield_linear (phi'(0)=1, the -d x^3 derivative at 0 is 0), so the H1
        # structure is identical; the value is a NONLINEAR functional of the history => H2 may HOLD.
        #   hopfield_nonlinear : SATURATING tanh (Hopfield), damping=0. Strong nonlinearity (large
        #       kappa) saturates the delayed-state sensitivity (phi'->0), eroding H1 -- a tradeoff.
        #   hopfield_duffing   : NON-SATURATING cubic phi=x+kappa x^3 (phi'=1+3 kappa x^2 GROWS, so
        #       history sensitivity is preserved => H1 held) + instantaneous -d x^3 confinement that
        #       bounds the otherwise-explosive plant. A phi^4/Duffing variant (not literal Hopfield).
        # kappa/nonlinearity/damping overridable via HOPFIELD_KAPPA / HOPFIELD_NONLIN / HOPFIELD_DAMPING
        # (path-A kappa sweep on hopfield_nonlinear); Pontryagin oracle gates verified.
        from src.envs.delayed_hopfield_network import DelayedHopfieldNetwork, rotational_coupling
        W = rotational_coupling(rho=2.0, theta=np.pi / 2.0); dt = 0.1; eps = 1.0
        # HOPFIELD_TAU sweeps the DELAY (=> window length round(tau/dt)+1): the operative knob for H2.
        # raw_history degree-2 dim ~ (window*n)^2/2 EXPLODES with tau while the signature dim is
        # window-independent, so longer delay -> raw_history ill-conditioned -> signature should win.
        tau = float(os.environ.get("HOPFIELD_TAU", "0.5"))
        Ncheb = int(os.environ.get("HOPFIELD_NCHEB", str(max(8, round(3 * tau)))))   # scale collocation w/ tau (MG rule)
        if name == "hopfield_duffing":
            kappa = float(os.environ.get("HOPFIELD_KAPPA", "0.5"))
            nonlin = os.environ.get("HOPFIELD_NONLIN", "cubic")
            damping = float(os.environ.get("HOPFIELD_DAMPING", "2.0"))   # d>~rho*kappa bounds the plant
        else:
            kappa = float(os.environ.get("HOPFIELD_KAPPA", "1.0"))
            nonlin = os.environ.get("HOPFIELD_NONLIN", "tanh")
            damping = float(os.environ.get("HOPFIELD_DAMPING", "0.0"))
        env = DelayedHopfieldNetwork(W=W, delay=tau, eps=eps, kappa=kappa, nonlinearity=nonlin,
                                     damping=damping, step_size=dt, resolution=4)
        A0, A1, B = np.array(env.A), np.array(env.A1), np.array(env.B)
        Q, R = np.array(env.Q), np.array(env.R)
        oracle = build_delayed_oracle(A0, A1, B, Q, R, tau, Ncheb)
        win = int(round(tau / dt)) + 1
        return dict(name=name, env=env, oracle=oracle, n=2, dt=dt, window_length=win, tf=15.0,
                    clip=10.0, x0c=np.array([0.5, 0.0]), nonlinear=True, family="hopfield", xstar=0.0,
                    hopfield=dict(W=W, eps=eps, kappa=kappa, nonlinearity=nonlin, damping=damping))
    if name in ("mg_limit_cycle", "mg_chaotic"):
        from src.envs.mackey_glass_1D import MackeyGlass1DEnv
        from src.solvers.mackey_glass_pontryagin import mg_gp
        tau = 6.0 if name == "mg_limit_cycle" else 17.0
        mu, p, n_exp, xs, dt = 0.1, 0.2, 10, 1.0, 0.25
        Ncheb = 18 if tau == 6.0 else 28
        Q = np.array([[1.0]]); R = np.array([[0.1]])
        env = MackeyGlass1DEnv(delay=tau, step_size=dt, resolution=5, Q=Q, R=R, n=n_exp,
                               p=p, mu=mu, x_target=xs)
        A0 = np.array([[-mu]]); A1 = np.array([[p * mg_gp(xs, n_exp)]]); B = np.array([[1.0]])
        oracle = build_delayed_oracle(A0, A1, B, Q, R, tau, Ncheb)
        win = int(round(tau / dt)) + 1
        return dict(name=name, env=env, oracle=oracle, n=1, dt=dt, window_length=win, tf=40.0,
                    clip=5.0, x0c=np.array([0.8]), nonlinear=True, xstar=xs,
                    mg=dict(mu=mu, p=p, n_exp=n_exp, xs=xs, tau=tau, m=Ncheb + 1))
    raise ValueError(f"unknown cell {name}")


def label_fns(cell):
    """Analytic (V*, u*) of a window for a LINEAR cell (target = origin, so delta z = z)."""
    oc, dt, n = cell["oracle"], cell["dt"], cell["n"]
    if oc["kind"] == "markovian":
        P, K = oc["P"], oc["K"]
        return (lambda W: -float(W[-1] @ P @ W[-1]), lambda W: -K @ W[-1])
    Pc, Kc, theta = oc["Pc"], oc["Kc"], oc["theta"]
    def Vs(W): z = z_from_window(W, dt, theta, n); return -float(z @ Pc @ z)
    def us(W): z = z_from_window(W, dt, theta, n); return -Kc @ z
    return Vs, us


def mg_linear_feedback(cell):
    """Linear-oracle feedback u = -Kc (z - x* 1) on the MG window (the deployable near-optimal
    feedback reference for I_or)."""
    oc, dt, xs = cell["oracle"], cell["dt"], cell["xstar"]
    Kc, theta = oc["Kc"], oc["theta"]
    return lambda W: -Kc @ (z_from_window(W, dt, theta, 1) - xs)


def oracle_label(cell):
    """Return a function W -> V*(W) (reward convention). UNIFORM interface; the underlying oracle is
    analytic for the linear/linearised cells and a per-window Pontryagin BVP for Mackey-Glass. The MG
    label returns None if the BVP fails to converge for that perturbed window."""
    oc, dt, n = cell["oracle"], cell["dt"], cell["n"]
    if cell.get("family") == "hopfield":
        from src.solvers.delayed_hopfield_pontryagin import solve_pontryagin_hopfield, value_along_hopfield
        hp = cell["hopfield"]
        def lab(W):
            z0 = z_from_window(W, dt, oc["theta"], n)            # window -> augmented collocation IC
            sol, _ = solve_pontryagin_hopfield(z0, oc, hp["W"], hp["eps"], hp["kappa"], T=12.0,
                                               nonlinearity=hp["nonlinearity"], damping=hp["damping"])
            if not sol.success:
                return None
            _, _, _, Vcost = value_along_hopfield(sol, oc)
            return -float(Vcost[0])                              # reward convention
        return lab
    if cell.get("nonlinear"):
        from src.solvers.mackey_glass_pontryagin import solve_pontryagin, value_along
        mg = cell["mg"]
        D, Pc, Kc, Nmat, M, theta = oc["D"], oc["Pc"], oc["Kc"], oc["Nmat"], oc["M"], oc["theta"]
        R = float(oc["R"][0, 0]); m = D.shape[0]; A_cl = M - Nmat @ Kc; xs = mg["xs"]
        def lab(W):
            z0 = z_from_window(W, dt, theta, 1)                  # window -> collocation IC
            sol, _ = solve_pontryagin(z0, D, Pc, A_cl, xs, mg["mu"], mg["p"], mg["n_exp"], R, T=20.0)
            if not sol.success:
                return None
            _, _, _, Vcost = value_along(sol, m, xs, Pc, R)
            return -float(Vcost[0])                             # reward convention
        return lab
    if oc["kind"] == "markovian":
        P = oc["P"]; return lambda W: -float(np.asarray(W)[-1] @ P @ np.asarray(W)[-1])
    Pc, theta = oc["Pc"], oc["theta"]
    return lambda W: -float((lambda z: z @ Pc @ z)(z_from_window(W, dt, theta, n)))


def on_sheet_windows(cell, rng):
    """On-sheet data: windows along the optimal/oracle trajectory from a set of initial conditions.
    Linear cells roll the env under u*; MG cells integrate the Pontryagin optimum. Returns (Ws, Vs)."""
    lab = oracle_label(cell)
    if cell.get("family") == "hopfield":
        from src.solvers.delayed_hopfield_pontryagin import build_hopfield_pontryagin_dataset
        hp = cell["hopfield"]
        n_ic = int(os.environ.get("H1H2_N_IC", str(N_IC_MG)))                  # on-sheet IC count (data-scaling knob)
        ics = [0.5 * rng.standard_normal(cell["n"]) for _ in range(n_ic)]      # cloud of initial states
        Ws, Vs, _ = build_hopfield_pontryagin_dataset(cell["oracle"], hp["W"], hp["eps"], hp["kappa"],
                                                      cell["dt"], cell["window_length"], ics,
                                                      T=12.0, stride=SUBSAMPLE,
                                                      nonlinearity=hp["nonlinearity"], damping=hp["damping"])
        return list(Ws), list(Vs.tolist())
    if cell.get("nonlinear"):
        from src.solvers.mackey_glass_pontryagin import build_pontryagin_dataset
        mg = cell["mg"]; m = mg["m"]
        ics = [x0 * np.ones(m) for x0 in np.linspace(0.5, 1.45, N_IC_MG)]
        Ws, Vs, _ = build_pontryagin_dataset(cell["oracle"], mg["mu"], mg["p"], mg["n_exp"], mg["xs"],
                                             cell["dt"], cell["window_length"], ics, T=20.0, stride=SUBSAMPLE)
        return list(Ws), list(Vs.tolist())
    _, us = label_fns(cell)
    Ws, Vs = [], []
    for _ in range(N_IC):
        traj, _ = rollout(cell, us, cell["x0"](rng))
        for W in traj[::SUBSAMPLE]:
            Ws.append(W); Vs.append(lab(W))
    return Ws, Vs


# --- parallel off-sheet BVP labelling (module-level workers, picklable for spawn) ---
# Each off-sheet window needs one Pontryagin BVP (~2.5 s, single-threaded scipy). For the BVP
# cells (hopfield / MG) this dominates the wall-clock (n_off solves), so we fan it out across the
# task's cores. The solvers are pure numpy/scipy (no JAX), so a 'spawn' pool is safe even though
# JAX is already initialised in the parent (fork would risk an XLA-thread deadlock; spawn does not).
def _off_label_hopfield(args):
    Wp, oracle, hp, dt, n = args
    from src.solvers.continuous_oracle import z_from_window
    from src.solvers.delayed_hopfield_pontryagin import solve_pontryagin_hopfield, value_along_hopfield
    z0 = z_from_window(Wp, dt, oracle["theta"], n)
    sol, _ = solve_pontryagin_hopfield(z0, oracle, hp["W"], hp["eps"], hp["kappa"], T=12.0,
                                       nonlinearity=hp["nonlinearity"], damping=hp["damping"])
    if not sol.success:
        return None
    _, _, _, Vcost = value_along_hopfield(sol, oracle)
    return -float(Vcost[0])


def _off_label_mg(args):
    Wp, oracle, mg, dt = args
    from src.solvers.continuous_oracle import z_from_window
    from src.solvers.mackey_glass_pontryagin import solve_pontryagin, value_along
    D, Pc, Kc, Nmat, M = oracle["D"], oracle["Pc"], oracle["Kc"], oracle["Nmat"], oracle["M"]
    theta = oracle["theta"]; R = float(oracle["R"][0, 0]); m = D.shape[0]
    A_cl = M - Nmat @ Kc; xs = mg["xs"]
    z0 = z_from_window(Wp, dt, theta, 1)
    sol, _ = solve_pontryagin(z0, D, Pc, A_cl, xs, mg["mu"], mg["p"], mg["n_exp"], R, T=20.0)
    if not sol.success:
        return None
    _, _, _, Vcost = value_along(sol, m, xs, Pc, R)
    return -float(Vcost[0])


def _parallel_labels(worker, tasks):
    """Map ``worker`` over ``tasks`` across BVP_NPROC / SLURM_CPUS_PER_TASK cores via a spawn pool."""
    import multiprocessing as mp
    nproc = int(os.environ.get("BVP_NPROC", os.environ.get("SLURM_CPUS_PER_TASK", "0"))) or min(8, mp.cpu_count())
    if nproc <= 1 or len(tasks) < 4:
        return [worker(t) for t in tasks]
    with mp.get_context("spawn").Pool(nproc) as pool:
        return pool.map(worker, tasks)


def generate_data(cell, seed, off_sheet, n_off=N_OFF):
    """UNIFORM sampler across EVERY environment. on-sheet = optimal-trajectory windows; if off_sheet,
    add n_off windows perturbed by EPS and RE-LABELLED through the cell's oracle (analytic / BVP). The
    only per-cell parts are the env integrator and the oracle label -- the sampling logic is identical.
    For BVP cells the off-sheet relabelling is parallelised across cores (the dominant cost)."""
    rng = np.random.default_rng(seed)
    on_W, on_V = on_sheet_windows(cell, rng)
    W, V = list(on_W), list(on_V)
    if off_sheet:
        perturbed = [np.asarray(on_W[i]) + EPS * rng.standard_normal(np.asarray(on_W[i]).shape)
                     for i in rng.integers(0, len(on_W), n_off)]
        if cell.get("family") == "hopfield":
            tasks = [(Wp, cell["oracle"], cell["hopfield"], cell["dt"], cell["n"]) for Wp in perturbed]
            labels = _parallel_labels(_off_label_hopfield, tasks)
        elif cell.get("nonlinear"):                                 # MG (BVP)
            tasks = [(Wp, cell["oracle"], cell["mg"], cell["dt"]) for Wp in perturbed]
            labels = _parallel_labels(_off_label_mg, tasks)
        else:                                                       # linear cells: analytic label, fast
            lab = oracle_label(cell); labels = [lab(Wp) for Wp in perturbed]
        for Wp, v in zip(perturbed, labels):
            if v is not None:
                W.append(Wp); V.append(v)
    return np.array(W), np.array(V)


def deploy_mg(cell):
    """Nonlinear optimal trajectory + control from x0c (the cos reference)."""
    from src.solvers.mackey_glass_pontryagin import build_pontryagin_dataset
    mg = cell["mg"]; ic = [cell["x0c"][0] * np.ones(mg["m"])]
    Ws, _, Us = build_pontryagin_dataset(cell["oracle"], mg["mu"], mg["p"], mg["n_exp"], mg["xs"],
                                         cell["dt"], cell["window_length"], ic, T=20.0, stride=1)
    return np.array(Ws), Us


def deploy_hopfield(cell):
    """Nonlinear optimal trajectory + control from x0c (the cos reference) for the Hopfield cell."""
    from src.solvers.delayed_hopfield_pontryagin import build_hopfield_pontryagin_dataset
    hp = cell["hopfield"]
    Ws, _, Us = build_hopfield_pontryagin_dataset(cell["oracle"], hp["W"], hp["eps"], hp["kappa"],
                                                  cell["dt"], cell["window_length"], [cell["x0c"]],
                                                  T=12.0, stride=1, nonlinearity=hp["nonlinearity"],
                                                  damping=hp["damping"])
    return np.array(Ws), Us


def hopfield_linear_feedback(cell):
    """Deployable linear-oracle feedback u = -Kc z (origin equilibrium) -- the I_or reference."""
    oc, dt, n = cell["oracle"], cell["dt"], cell["n"]
    Kc, theta = oc["Kc"], oc["theta"]
    return lambda W: -Kc @ z_from_window(W, dt, theta, n)


# ============================== rollout / closed loop ==============================
def rollout(cell, control_fn, x0):
    """Roll the env under control_fn(window)->u; return the window sequence and cost I."""
    import jax, jax.numpy as jnp
    from src.envs.env_rk_jax import JAXEnvWrapper
    from src.representations.factory import make_representation, RepresentationBuffer
    env, win, n, dt, tf, clip = (cell["env"], cell["window_length"], cell["n"], cell["dt"],
                                 cell["tf"], cell["clip"])
    rep = make_representation("markovian", window_length=win, n_state=n, degree=1)
    buf = RepresentationBuffer(rep, window_length=win, n_state=n)
    w = JAXEnvWrapper(env, rng_key=42); w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    buf.reset()
    for _ in range(win):
        buf.append(np.asarray(x0, dtype=np.float32))
    Ws, I, t = [], 0.0, 0.0
    while t < tf - 1e-9:
        W = np.array(buf.buffer.to_array()).reshape(win, n)
        u = np.clip(np.asarray(control_fn(W)).reshape(-1), -clip, clip)
        if not np.all(np.isfinite(u)) or abs(I) > 1e8:
            return np.array(Ws), float("inf")
        Ws.append(W)
        t, x, r = w.step(w.state, jnp.array(u)); x = np.array(x).reshape(-1)
        I += -float(r) * dt; buf.append(x.astype(np.float32))
    return np.array(Ws), I


# ============================== fit + evaluate one representation ==============================
def evaluate_rep(cell, rep_cfg, Wtr, Vtr, dep_W, u_star_dep, half_RinvBT):
    import jax, jax.numpy as jnp
    from src.representations.factory import make_representation
    kw = {k: v for k, v in rep_cfg.items() if k != "kind"}
    rep = make_representation(rep_cfg["kind"], window_length=cell["window_length"],
                              n_state=cell["n"], **kw)
    feat = jax.jit(rep.feature_fn)
    Phi = np.stack([np.asarray(feat(jnp.asarray(W))) for W in Wtr])
    ridge = float(os.environ.get("H1H2_RIDGE", "0.0"))     # tiny Tikhonov on the oracle fit (0 = plain lstsq)
    if ridge > 0.0:
        G = Phi.T @ Phi
        lam = ridge * (np.trace(G) / G.shape[0])           # relative ridge (scale-aware)
        theta = np.linalg.solve(G + lam * np.eye(G.shape[1]), Phi.T @ Vtr)
    else:
        theta = np.linalg.lstsq(Phi, Vtr, rcond=None)[0]
    r2 = 1.0 - np.sum((Vtr - Phi @ theta) ** 2) / (np.sum((Vtr - Vtr.mean()) ** 2) + 1e-18)
    th = jnp.asarray(theta)
    Vgrad = jax.jit(jax.grad(lambda p: jnp.dot(th, feat(p))))
    u_rep = np.array([half_RinvBT @ np.array(Vgrad(jnp.asarray(W)))[-1] for W in dep_W])
    cos = float(np.sum(u_rep * u_star_dep) /
                (np.linalg.norm(u_rep) * np.linalg.norm(u_star_dep) + 1e-18))
    _, I = rollout(cell, lambda W: half_RinvBT @ np.array(Vgrad(jnp.asarray(W)))[-1], cell["x0c"])
    return dict(dim=int(Phi.shape[1]), n_train=int(Phi.shape[0]), r2=float(r2), cos=cos, I=float(I))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True,
                    choices=["markovian", "linear_dde", "hopfield_linear", "hopfield_nonlinear",
                             "hopfield_duffing", "platoon", "mg_limit_cycle", "mg_chaotic"])
    ap.add_argument("--rep", default="all", choices=["all", *REPS.keys()])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-mode", default="on_off_sheet", choices=["on_sheet", "on_off_sheet"],
                    help="on_sheet = optimal-trajectory windows only; on_off_sheet = + off-sheet "
                         "windows (perturb + re-label through the cell's oracle). Uniform across envs.")
    ap.add_argument("--n-off", type=int, default=N_OFF, help="number of off-sheet windows (on_off_sheet mode)")
    ap.add_argument("--out-dir", default=None, help="explicit output dir (job arrays); else auto")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cell = make_cell(args.cell)
    oc = cell["oracle"]; R, B = oc["R"], oc["B"]
    half_RinvBT = 0.5 * np.linalg.inv(R) @ B.T

    gcos, gmag = doya_gate(oc, cell["dt"])
    print(f"\n=== H1/H2 cell '{cell['name']}' | seed {args.seed} ===")
    print(f"oracle gate: cos={gcos:.4f}  |u|/|u*|={gmag:.4f}  (must be 1.000)")

    # deployment (cos reference) is cell-specific; the DATA SAMPLING is uniform (generate_data).
    if cell.get("family") == "hopfield":
        dep_W, u_star_dep = deploy_hopfield(cell)
        _, I_or = rollout(cell, hopfield_linear_feedback(cell), cell["x0c"])
    elif cell.get("nonlinear"):
        dep_W, u_star_dep = deploy_mg(cell)
        _, I_or = rollout(cell, mg_linear_feedback(cell), cell["x0c"])
    else:
        Vs, us = label_fns(cell)
        dep_W, I_or = rollout(cell, us, cell["x0c"])
        u_star_dep = np.array([us(W) for W in dep_W])
    Wtr, Vtr = generate_data(cell, args.seed, off_sheet=(args.data_mode == "on_off_sheet"), n_off=args.n_off)
    _, I_nc = rollout(cell, lambda W: np.zeros(B.shape[1]), cell["x0c"])
    print(f"data_mode: {args.data_mode}   reference: no-control I={I_nc:.4f}   "
          f"oracle-feedback I*={I_or:.4f}  (n_train={len(Wtr)})")
    reps = list(REPS) if args.rep == "all" else [args.rep]
    rows = {}
    print(f"\n{'rep':>12}{'dim':>6}{'n/d':>7}{'valR2':>9}{'gradcos':>9}{'I_cl':>10}")
    for name in reps:
        res = evaluate_rep(cell, REPS[name], Wtr, Vtr, dep_W, u_star_dep, half_RinvBT)
        rows[name] = res
        print(f"{name:>12}{res['dim']:>6}{res['n_train']/res['dim']:>7.1f}"
              f"{res['r2']:>9.3f}{res['cos']:>9.3f}{res['I']:>10.4f}")

    if args.rep == "all":
        # a hypothesis "holds" only if the contender lowers cost by more than MARGIN (relative);
        # a near-tie is the negative result (e.g. history that does not help). Single-seed verdict
        # is indicative; the falsifiable test is the seed-CI comparison in the aggregation step.
        MARGIN = 0.03
        def margin(test, base):
            return (rows[base]["I"] - rows[test]["I"]) / (abs(rows[base]["I"]) + 1e-18)
        m1, m2 = margin("raw_history", "markovian"), margin("signature", "raw_history")
        print(f"\nH1 (raw_history > markovian): {'HOLDS' if m1 > MARGIN else 'fails'} "
              f"(I {rows['raw_history']['I']:.4f} vs {rows['markovian']['I']:.4f}; "
              f"cost reduction {100*m1:+.1f}%)")
        print(f"H2 (signature > raw_history): {'HOLDS' if m2 > MARGIN else 'fails'} "
              f"(I {rows['signature']['I']:.4f} vs {rows['raw_history']['I']:.4f}; "
              f"cost reduction {100*m2:+.1f}%)")

    from src.utils.run_context import script_data_dir
    tag = f"{cell['name']}_{args.data_mode}_seed{args.seed}"
    if args.out_dir is not None:                                # job-array mode: per-task path
        out = Path(args.out_dir) / tag
    else:
        debug = "_debug_" if args.debug else ""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = script_data_dir(__file__) / f"{debug}{ts}_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(
        cell=cell["name"], seed=args.seed, data_mode=args.data_mode, gate=dict(cos=gcos, mag=gmag),
        I_nc=I_nc, I_or=I_or, reps=rows), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
