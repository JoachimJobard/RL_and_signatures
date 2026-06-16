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
# data-generation defaults (n_train kept comfortably > feature_dim)
N_IC = 8
AUG_PER = 3
EPS_TRAIN = 0.3
SUBSAMPLE = 2


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


def generate_data(cell, Vs, us, seed):
    """On-manifold (oracle rollouts, sub-sampled) + off-manifold (perturbed) windows, n>d."""
    rng = np.random.default_rng(seed)
    on = []
    for _ in range(N_IC):
        Ws, _ = rollout(cell, us, cell["x0"](rng))
        on.append(Ws[::SUBSAMPLE])
    onman = np.concatenate(on, 0)
    items = [onman]
    for _ in range(AUG_PER):
        items.append(onman + EPS_TRAIN * rng.standard_normal(onman.shape))
    Wtr = np.concatenate(items, 0)
    Vtr = np.array([Vs(W) for W in Wtr])
    return Wtr, Vtr


# ============================== fit + evaluate one representation ==============================
def evaluate_rep(cell, rep_cfg, Wtr, Vtr, dep_W, u_star_dep, half_RinvBT):
    import jax, jax.numpy as jnp
    from src.representations.factory import make_representation
    kw = {k: v for k, v in rep_cfg.items() if k != "kind"}
    rep = make_representation(rep_cfg["kind"], window_length=cell["window_length"],
                              n_state=cell["n"], **kw)
    feat = jax.jit(rep.feature_fn)
    Phi = np.stack([np.asarray(feat(jnp.asarray(W))) for W in Wtr])
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
    ap.add_argument("--cell", required=True, choices=["markovian", "linear_dde", "platoon"])
    ap.add_argument("--rep", default="all", choices=["all", *REPS.keys()])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cell = make_cell(args.cell)
    oc = cell["oracle"]; R, B = oc["R"], oc["B"]
    half_RinvBT = 0.5 * np.linalg.inv(R) @ B.T
    Vs, us = label_fns(cell)

    # gate the oracle before using it
    gcos, gmag = doya_gate(oc, cell["dt"])
    print(f"\n=== H1/H2 cell '{cell['name']}' | seed {args.seed} ===")
    print(f"oracle gate: cos={gcos:.4f}  |u|/|u*|={gmag:.4f}  (must be 1.000)")

    # references and deployment trajectory (for cos)
    dep_W, I_or = rollout(cell, us, cell["x0c"])
    _, I_nc = rollout(cell, lambda W: np.zeros(B.shape[1]), cell["x0c"])
    u_star_dep = np.array([us(W) for W in dep_W])
    print(f"reference: no-control I={I_nc:.4f}   continuous oracle I*={I_or:.4f}")

    Wtr, Vtr = generate_data(cell, Vs, us, args.seed)
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

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    from src.utils.run_context import script_data_dir
    out = script_data_dir(__file__) / f"{debug}{ts}_{cell['name']}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(
        cell=cell["name"], seed=args.seed, gate=dict(cos=gcos, mag=gmag),
        I_nc=I_nc, I_or=I_or, reps=rows), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
