"""RL half of the H1/H2 matrix: damped LSPI (least-squares value-gradient LEARNING).

The learning counterpart of the oracle harness, using the SAME least-squares estimator (LSTD =
the TD analog of the oracle half's lstsq) but stabilised into a convergent loop:

  per iteration: freeze the policy -> collect transitions UNDER it from diverse-IC rollouts
  (on-policy, n>>D) -> LSTD-evaluate -> DAMPED update theta <- (1-a) theta + a theta_lstd
  -> greedy value-gradient improve. Exploration noise keeps the data off-manifold; the discount
  tau is the gentleness lever for delicate (near-Hopf) plants.

Reports, per (cell, rep, seed), the BEST-iterate metrics: corr^2 of the learned value with the
linear-oracle value, gradient cos(u, u*), and closed-loop I -- the same three axes and summary.json
shape as h1h2_evaluation.py, so h1h2_aggregate.py reads both halves. Array-ready via --out-dir.

Usage:
    uv run python run/study/h1h2_rl_lspi.py --cell mg_limit_cycle --rep all --seed 0 --debug
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))                  # sibling study modules
from h1h2_evaluation import make_cell, REPS                                # noqa: E402
from src.solvers.continuous_oracle import z_from_window                    # noqa: E402
from mwe_lspi_test import (make_feat, reference_control, rollout,           # noqa: E402
                           collect_dataset, lstd_solve, greedy)

# per-cell LSPI config: tau (discount gentleness lever -- smaller for delicate plants), damping,
# exploration. Validated: mg tau=2, linear_dde signature tau=1, markovian full-ish; platoon gentler.
CFG = {
    "markovian":      dict(tau=1.0, damp=0.3, explore=0.2, rank=0),
    "linear_dde":     dict(tau=1.0, damp=0.1, explore=0.3, rank=0),
    # n_roll=50 (not the default 20): dim-90 raw_history needs n/D~25 for a well-conditioned LSTD;
    # at n_roll=20 it diverges. The fix is DATA, not truncation/damping (sweep 2026-06-17).
    # reg=1e-4 (not the default 1e-3): for dim-90 raw_history the ABSOLUTE ridge is non-monotonic --
    # a tiny ridge (<=1e-4) keeps the LSTD solve in the stable basin, the default 1e-3 sits ON a
    # policy-stability boundary where the solve is thread-fragile (diverges), 1e-1 is stable again
    # (ridge sweep 2026-06-17). Tiny ridge = thread-robust convergence, no truncation.
    "hopfield_linear":    dict(tau=1.0, damp=0.1, explore=0.3, rank=0, n_roll=50, reg=1e-4),
    "hopfield_nonlinear": dict(tau=1.0, damp=0.1, explore=0.3, rank=0, n_roll=50, reg=1e-4),
    "hopfield_duffing":   dict(tau=1.0, damp=0.1, explore=0.3, rank=0, n_roll=50, reg=1e-4),
    "platoon":        dict(tau=0.5, damp=0.1, explore=0.5, rank=20),   # high-dim: centred + truncated LSTD
    "mg_limit_cycle": dict(tau=2.0, damp=0.1, explore=0.3, rank=0),
    "mg_chaotic":     dict(tau=2.0, damp=0.1, explore=0.3, rank=0),
}
N_ITER = 12


def linear_value(cell, W):
    """The linear-oracle value -delta z^T Pc delta z (analytic, cheap) for the corr^2 diagnostic."""
    oc, dt, n = cell["oracle"], cell["dt"], cell["n"]
    if oc["kind"] == "markovian":
        x = np.asarray(W)[-1]; return -float(x @ oc["P"] @ x)
    z = z_from_window(W, dt, oc["theta"], n) - cell.get("xstar", 0.0)
    return -float(z @ oc["Pc"] @ z)


def deployment_windows(cell, u_star):
    import jax, jax.numpy as jnp
    from src.envs.env_rk_jax import JAXEnvWrapper
    from src.representations.factory import make_representation, RepresentationBuffer
    env, win, n, dt, tf, clip = (cell["env"], cell["window_length"], cell["n"], cell["dt"],
                                 cell["tf"], cell["clip"])
    rep = make_representation("markovian", window_length=win, n_state=n, degree=1)
    buf = RepresentationBuffer(rep, window_length=win, n_state=n)
    w = JAXEnvWrapper(env, rng_key=42); w.reset(jax.random.PRNGKey(0), x0=jnp.array(cell["x0c"]), t0=0.0)
    buf.reset()
    for _ in range(win):
        buf.append(np.asarray(cell["x0c"], dtype=np.float32))
    Ws, t = [], 0.0
    while t < tf - 1e-9:
        W = np.array(buf.buffer.to_array()).reshape(win, n); Ws.append(W)
        u = np.clip(np.asarray(u_star(W)).reshape(-1), -clip, clip)
        t, x, _ = w.step(w.state, jnp.array(u)); buf.append(np.array(x).reshape(-1).astype(np.float32))
    return Ws


def evaluate_rep(cell, rep_cfg, seed, cfg, depW, u_star_dep):
    """Run damped LSPI; return the BEST-iterate (lowest I) {dim, r2, cos, I}."""
    import jax, jax.numpy as jnp
    feat, D = make_feat(cell, rep_cfg)
    oc = cell["oracle"]; B, R = oc["B"], oc["R"]
    half_RinvBT = 0.5 * np.linalg.inv(R) @ B.T
    dt, tf = cell["dt"], cell["tf"]
    t_collect = tf if cell.get("nonlinear") else tf * 0.3
    Vlin_dep = np.array([linear_value(cell, W) for W in depW])

    theta = np.zeros(D)
    policy = lambda W: np.zeros(B.shape[1])
    best = dict(dim=int(D), r2=0.0, cos=0.0, I=float("inf"))
    for it in range(N_ITER):
        P, PN, RW = collect_dataset(cell, feat, policy, seed=seed * 1000 + it,
                                    t_collect=t_collect, sigma=cfg["explore"],
                                    n_roll=cfg.get("n_roll", 20))
        if len(P) == 0:
            break
        theta = (1.0 - cfg["damp"]) * theta + cfg["damp"] * lstd_solve(
            P, PN, RW, D, dt, cfg["tau"], rank=cfg.get("rank", 0), reg=cfg.get("reg", 1e-3))
        policy = greedy(feat, theta, half_RinvBT)
        I = rollout(cell, feat, policy, cell["x0c"], tf)
        if not np.isfinite(I):
            continue
        u_rep = np.array([np.asarray(policy(W)).reshape(-1) for W in depW])
        cos = float(np.sum(u_rep * u_star_dep) / (np.linalg.norm(u_rep) * np.linalg.norm(u_star_dep) + 1e-12))
        Vth = np.array([float(np.dot(theta, np.asarray(feat(jnp.asarray(W))))) for W in depW])
        r2 = float(np.corrcoef(Vth, Vlin_dep)[0, 1] ** 2) if np.std(Vth) > 1e-12 else 0.0
        if I < best["I"]:
            best = dict(dim=int(D), r2=r2, cos=cos, I=float(I))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=list(CFG))
    ap.add_argument("--rep", default="all", choices=["all", *REPS])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=None, help="override the per-cell discount tau")
    ap.add_argument("--rank", type=int, default=None, help="override the per-cell truncated-SVD rank (0 = none)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cell = make_cell(args.cell)
    cfg = dict(CFG[args.cell])
    if args.tau is not None:
        cfg["tau"] = args.tau
    if args.rank is not None:
        cfg["rank"] = args.rank
    B = cell["oracle"]["B"]
    u_star = reference_control(cell)
    depW = deployment_windows(cell, u_star)
    u_star_dep = np.array([np.asarray(u_star(W)).reshape(-1) for W in depW])

    I_nc = rollout(cell, None, lambda W: np.zeros(B.shape[1]), cell["x0c"], cell["tf"])
    I_or = rollout(cell, None, u_star, cell["x0c"], cell["tf"])
    print(f"\n=== RL-LSPI cell '{cell['name']}' | seed {args.seed} | tau={cfg['tau']} damp={cfg['damp']} explore={cfg['explore']} ===")
    print(f"reference: no-control I={I_nc:.4f}   oracle-feedback I*={I_or:.4f}")
    print(f"\n{'rep':>12}{'dim':>6}{'corr2':>9}{'gradcos':>9}{'I_cl':>11}")

    reps = list(REPS) if args.rep == "all" else [args.rep]
    rows = {}
    for name in reps:
        res = evaluate_rep(cell, REPS[name], args.seed, cfg, depW, u_star_dep)
        rows[name] = res
        print(f"{name:>12}{res['dim']:>6}{res['r2']:>9.3f}{res['cos']:>9.3f}{res['I']:>11.4f}")

    if args.rep == "all":
        MARGIN = 0.03
        def m(t, b): return (rows[b]["I"] - rows[t]["I"]) / (abs(rows[b]["I"]) + 1e-18)
        h1, h2 = m("raw_history", "markovian"), m("signature", "raw_history")
        print(f"\nH1 (raw_history > markovian): {'HOLDS' if h1 > MARGIN else 'fails'} "
              f"(I {rows['raw_history']['I']:.4f} vs {rows['markovian']['I']:.4f})")
        print(f"H2 (signature > raw_history): {'HOLDS' if h2 > MARGIN else 'fails'} "
              f"(I {rows['signature']['I']:.4f} vs {rows['raw_history']['I']:.4f})")

    tag = f"{cell['name']}_seed{args.seed}"
    from src.utils.run_context import script_data_dir
    if args.out_dir is not None:
        out = Path(args.out_dir) / tag
    else:
        debug = "_debug_" if args.debug else ""
        out = script_data_dir(__file__) / f"{debug}{datetime.now():%Y%m%d_%H%M%S}_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(
        cell=cell["name"], seed=args.seed, solver="damped_lspi", tau=cfg["tau"],
        I_nc=I_nc, I_or=I_or, reps=rows), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
