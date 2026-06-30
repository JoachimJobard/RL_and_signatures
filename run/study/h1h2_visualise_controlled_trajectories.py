"""Visualisation: closed-loop controlled trajectories and their control signals, per cell.

For one H1/H2 cell this fits the markovian / raw-history / signature critics on EXACTLY the
benchmark data (run/study/h1h2_evaluation.py: same generate_data, same seed, same
H1H2_RIDGE / H1H2_N_IC / --n-off knobs), then deploys each critic's value-gradient feedback
from the cell's reference initial condition and records the state trajectory x(t) and the
control signal u(t). The analytic oracle feedback (the I_or reference) and the no-control
response are recorded alongside. The local closed-loop cost I is printed next to the cluster
value so the figure is a faithful regeneration (consistency check), not a fresh experiment.

Stroke convention (repo-wide): solid = learned critics; dashed = analytic oracle reference;
dotted = no-control / auxiliary. Distinct colours per critic (not a hyperparameter sweep).

Saves a .npz of every trajectory + a two-panel figure (state ; control). Replot-safe: the
.npz regenerates the figure without re-fitting (--from-npz).

Usage:
    uv run python run/study/h1h2_visualise_controlled_trajectories.py --cell linear_dde
    HOPFIELD_TAU=2 uv run python run/study/h1h2_visualise_controlled_trajectories.py \
        --cell hopfield_nonlinear --tag tau2_starved
    HOPFIELD_TAU=2 H1H2_N_IC=120 uv run python ... --cell hopfield_nonlinear --n-off 2000 --tag tau2_controlled
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

_REP_COLOR = {"markovian": "#1b9e77", "raw_history": "#d95f02", "signature": "#7570b3"}


def capture_rollout(cell, control_fn, x0):
    """Roll the env under control_fn(window)->u from x0; record (t, x[T,n], u[T,m], I)."""
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
    ts, xs, us, I, t = [], [], [], 0.0, 0.0
    while t < tf - 1e-9:
        W = np.array(buf.buffer.to_array()).reshape(win, n)
        u = np.clip(np.asarray(control_fn(W)).reshape(-1), -clip, clip)
        if not np.all(np.isfinite(u)) or abs(I) > 1e8:
            break
        t, x, r = w.step(w.state, jnp.array(u)); x = np.array(x).reshape(-1)
        ts.append(float(t)); xs.append(x.copy()); us.append(np.asarray(u).copy())
        I += -float(r) * dt; buf.append(x.astype(np.float32))
    return np.array(ts), np.array(xs), np.array(us), float(I)


def fit_policy(cell, rep_cfg, Wtr, Vtr, half_RinvBT):
    """Fit one critic on the benchmark data; return its value-gradient feedback control_fn."""
    import jax, jax.numpy as jnp
    from src.representations.factory import make_representation
    kw = {k: v for k, v in rep_cfg.items() if k != "kind"}
    rep = make_representation(rep_cfg["kind"], window_length=cell["window_length"],
                              n_state=cell["n"], **kw)
    feat = jax.jit(rep.feature_fn)
    Phi = np.stack([np.asarray(feat(jnp.asarray(W))) for W in Wtr])
    ridge = float(os.environ.get("H1H2_RIDGE", "0.0"))
    if ridge > 0.0:
        G = Phi.T @ Phi
        lam = ridge * (np.trace(G) / G.shape[0])
        theta = np.linalg.solve(G + lam * np.eye(G.shape[1]), Phi.T @ Vtr)
    else:
        theta = np.linalg.lstsq(Phi, Vtr, rcond=None)[0]
    th = jnp.asarray(theta)
    Vgrad = jax.jit(jax.grad(lambda p: jnp.dot(th, feat(p))))
    return lambda W: half_RinvBT @ np.array(Vgrad(jnp.asarray(W)))[-1], int(Phi.shape[1])


def reference_control(cell):
    """The analytic oracle feedback (the I_or deployable reference) for this cell."""
    from h1h2_evaluation import (label_fns, hopfield_linear_feedback, mg_linear_feedback)
    if cell.get("family") == "hopfield":
        return hopfield_linear_feedback(cell)
    if cell.get("nonlinear"):
        return mg_linear_feedback(cell)
    _, us = label_fns(cell)
    return us


def generate(cell_name, seed, n_off, data_mode):
    from h1h2_evaluation import make_cell, generate_data, REPS
    cell = make_cell(cell_name)
    oc = cell["oracle"]; R, B = oc["R"], oc["B"]
    half_RinvBT = 0.5 * np.linalg.inv(R) @ B.T
    Wtr, Vtr = generate_data(cell, seed, off_sheet=(data_mode == "on_off_sheet"), n_off=n_off)
    traj = {}
    # reference (oracle feedback) and no-control
    tr, xr, ur, Ir = capture_rollout(cell, reference_control(cell), cell["x0c"])
    traj["oracle"] = dict(t=tr, x=xr, u=ur, I=Ir)
    tnc, xnc, unc, Inc = capture_rollout(cell, lambda W: np.zeros(B.shape[1]), cell["x0c"])
    traj["no_control"] = dict(t=tnc, x=xnc, u=unc, I=Inc)
    dims = {}
    for name in REPS:
        ctrl, dim = fit_policy(cell, REPS[name], Wtr, Vtr, half_RinvBT)
        t, x, u, I = capture_rollout(cell, ctrl, cell["x0c"])
        traj[name] = dict(t=t, x=x, u=u, I=I); dims[name] = dim
    print(f"  cell={cell_name} n_train={len(Wtr)}  "
          f"I: oracle={Ir:.4g} nc={Inc:.4g} "
          + " ".join(f"{k}={traj[k]['I']:.4g}(d{dims[k]})" for k in REPS))
    return cell, traj, dims


def _data_ylim(curves_ref, curves_rep, n_off):
    """y-limits from the reference curves, widened to include only the converged reps
    (within 5x the reference amplitude) so a diverged critic is clipped, not scale-crushing."""
    ref = np.concatenate(curves_ref)
    lo, hi = float(np.min(ref)), float(np.max(ref))
    amp = max(abs(lo), abs(hi), 1e-3)
    for c in curves_rep:
        c = np.asarray(c)
        if np.all(np.isfinite(c)) and np.max(np.abs(c)) < 5.0 * amp:
            lo, hi = min(lo, float(np.min(c))), max(hi, float(np.max(c)))
    m = 0.15 * (hi - lo + 1e-6)
    return lo - m, hi + m


def make_figure(cell_name, cell, traj, dims, out_png, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axx, axu) = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True)
    reps = ("markovian", "raw_history", "signature")
    # references first (background): dotted = no-control, dashed = analytic oracle
    axx.plot(traj["no_control"]["t"], traj["no_control"]["x"][:, 0], ls=":", color="0.5",
             lw=1.3, label=f"no control (I={traj['no_control']['I']:.3g})")
    axx.plot(traj["oracle"]["t"], traj["oracle"]["x"][:, 0], ls="--", color="k", lw=1.7,
             label=f"oracle reference (I={traj['oracle']['I']:.3g})")
    axu.plot(traj["oracle"]["t"], traj["oracle"]["u"][:, 0], ls="--", color="k", lw=1.7)
    for name in reps:
        c = _REP_COLOR[name]; I = traj[name]["I"]
        lab = (f"{name} (dim {dims[name]}, I={I:.3g})" if np.isfinite(I) and abs(I) < 1e6
               else f"{name} (dim {dims[name]}, diverged)")
        axx.plot(traj[name]["t"], traj[name]["x"][:, 0], ls="-", color=c, lw=1.6, label=lab)
        axu.plot(traj[name]["t"], traj[name]["u"][:, 0], ls="-", color=c, lw=1.6)
    axx.set_ylabel(r"state $x_0(t)$"); axu.set_ylabel(r"control $u_0(t)$"); axu.set_xlabel(r"time $t$")
    axx.axhline(0.0, color="0.85", lw=0.6, zorder=0)
    axx.set_ylim(*_data_ylim([traj["oracle"]["x"][:, 0], traj["no_control"]["x"][:, 0]],
                             [traj[n]["x"][:, 0] for n in reps], 0))
    axu.set_ylim(*_data_ylim([traj["oracle"]["u"][:, 0]],
                             [traj[n]["u"][:, 0] for n in reps], 0))
    axx.set_title(title or f"Controlled trajectory and control — {cell_name}", fontsize=11)
    handles, labels = axx.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=3, fontsize=8, frameon=True)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-off", type=int, default=500)
    ap.add_argument("--data-mode", default="on_off_sheet", choices=["on_sheet", "on_off_sheet"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cell, traj, dims = generate(args.cell, args.seed, args.n_off, args.data_mode)
    from src.utils.run_context import script_data_dir
    tag = f"{args.cell}" + (f"_{args.tag}" if args.tag else "")
    out = Path(args.out_dir) if args.out_dir else (
        script_data_dir(__file__) / f"{datetime.now():%Y%m%d_%H%M%S}_{tag}")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{tag}.npz", **{f"{k}__{q}": traj[k][q] for k in traj for q in ("t", "x", "u")},
             **{f"{k}__I": traj[k]["I"] for k in traj},
             dims=json.dumps(dims), cell=args.cell, tag=tag)
    make_figure(args.cell, cell, traj, dims, out / f"{tag}.png",
                title=f"Controlled trajectory and control — {args.cell}" + (f" [{args.tag}]" if args.tag else ""))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
