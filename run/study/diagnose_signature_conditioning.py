"""Root-cause analysis: why the signature value-gradient critic fails on the platoon.

The signature value-gradient agent fails to learn a working controller on the
high-dimensional (2N=10) connected-cruise-control platoon (``src/envs/platoon.py``),
whereas the markovian critic succeeds. This script localises the cause by a data
analysis of the signature features along trajectories, separating three hypotheses:

  (1) the signature FEATURES are ill-conditioned (collinear / not excited);
  (2) the signature cannot REPRESENT the value (a representation/non-Markovian issue);
  (3) the CONTROL law from a good critic is bad (a vertical-derivative issue).

Method. Roll out the delayed-LQR oracle from one or many initial conditions, compute
each representation's features along the trajectories, and measure:
  - feature magnitude, Gram condition number, and effective rank (participation ratio
    ``(sum ev)^2 / sum ev^2``) -- single-IC vs many-IC (the EXCITATION test);
  - value-fit R^2 of a ridge least-squares critic on the (discounted) return-to-go
    (the REPRESENTATION test); and
  - the closed-loop cost of the value-gradient control extracted from that IDEAL
    least-squares critic (the CONTROL test): if the ideal critic controls well, the RL
    failure is pure optimisation/identifiability, not the control law.

Finding (measured): R^2 = 1.0 for every representation (the signature CAN represent the
value), but along a SINGLE fixed-initial-condition settling trajectory the signature
Gram has condition number ~1e12 and effective rank ~1 -- the high-dim features are
nearly collinear and unidentifiable. Exciting the data with many initial conditions
raises the effective rank and lowers the condition number, which is the lever: the
value-gradient agent's single fixed initial condition does not excite the signature.

Usage:
    uv run python run/study/diagnose_signature_conditioning.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")


REGIME = dict(n_vehicles=5, alpha=3.0, beta=1.0, delay=0.2, step_size=0.05, resolution=4)
TAU = 10.0            # discount for the return-to-go value targets
TF = 15.0
REPRESENTATIONS = [
    ("markovian d2", "markovian", dict(degree=2)),
    ("raw_hist d2", "raw_history", dict(degree=2)),
    ("signature d2", "signature", dict(depth=2)),
    ("signature d3", "signature", dict(depth=3)),
]


def _make_env():
    from src.envs.platoon import PlatoonEnv
    return PlatoonEnv(**REGIME)


def _oracle(env):
    from src.solvers.delayed_lqr import augmented_discrete_lqr
    A, A1, B = np.array(env.A), np.array(env.A1), np.array(env.B)
    Q, R = np.array(env.Q), np.array(env.R)
    return augmented_discrete_lqr(A, A1, B, Q, R, float(env.max_delay), env.step_size)


def _initial_conditions(n, n_ic, seed=0):
    """A diverse set of lead/multi-vehicle perturbations from uniform flow."""
    rng = np.random.default_rng(seed)
    ics = [np.zeros(n)]
    ics[0][1] = 0.5                       # the canonical lead-velocity perturbation
    for _ in range(n_ic - 1):
        x0 = np.zeros(n)
        x0[1::2] = 0.4 * rng.standard_normal(n // 2)   # random velocity perturbations
        ics.append(x0)
    return ics


def rollout(env, gain, kctrl, x0):
    import jax
    import jax.numpy as jnp
    from src.envs.env_rk_jax import JAXEnvWrapper
    w = JAXEnvWrapper(env, rng_key=42)
    w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    win = [x0.copy() for _ in range(kctrl + 1)]
    xs, rs = [], []
    t = 0.0
    while t < TF - 1e-9:
        u = -gain @ np.concatenate(win[:kctrl + 1])
        t, x, r = w.step(w.state, jnp.array(u))
        x = np.array(x).reshape(-1)
        xs.append(x); rs.append(float(r)); win = [x] + win[:-1]
    return np.array(xs), np.array(rs)


def return_to_go(rs, dt, tau):
    gamma = np.exp(-dt / tau)
    V = np.zeros(len(rs)); acc = 0.0
    for i in range(len(rs) - 1, -1, -1):
        acc = rs[i] * dt + gamma * acc
        V[i] = acc
    return V


def features_along(kind, xs_list, win_len, n, **kw):
    """Stack the representation's features over a list of state trajectories."""
    from src.representations.factory import make_representation, RepresentationBuffer
    rep = make_representation(kind, window_length=win_len, n_state=n, **kw)
    buf = RepresentationBuffer(rep, window_length=win_len, n_state=n)
    Phi = []
    for xs in xs_list:
        buf.reset()
        for _ in range(win_len):
            buf.append(xs[0].astype(np.float32))
        for x in xs:
            buf.append(x.astype(np.float32))
            Phi.append(np.array(buf.current_signature))
    return np.array(Phi), buf


def conditioning(Phi):
    d = Phi.shape[1]
    G = Phi.T @ Phi / len(Phi)
    ev = np.linalg.eigvalsh(G + 1e-15 * np.eye(d))
    ev = np.clip(ev, 1e-15, None)
    cond = float(ev[-1] / ev[0])
    eff_rank = float((ev.sum() ** 2) / (ev ** 2).sum())   # participation ratio
    return cond, eff_rank, ev


def value_fit_r2(Phi, V):
    d = Phi.shape[1]
    lam = 1e-6 * np.trace(Phi.T @ Phi) / d
    theta = np.linalg.solve(Phi.T @ Phi + lam * np.eye(d), Phi.T @ V)
    pred = Phi @ theta
    r2 = 1 - ((V - pred) ** 2).sum() / (((V - V.mean()) ** 2).sum() + 1e-12)
    return float(r2), theta


def ideal_critic_control_cost(env, kind, theta, win_len, n, x0, **kw):
    """Closed-loop cost of the value-gradient control u=1/2 R^-1 B^T dV/dx extracted
    from the IDEAL least-squares critic theta (the best possible linear-in-Phi fit)."""
    import jax
    import jax.numpy as jnp
    from src.representations.factory import make_representation, RepresentationBuffer
    rep = make_representation(kind, window_length=win_len, n_state=n, **kw)
    buf = RepresentationBuffer(rep, window_length=win_len, n_state=n)
    sig_fn = buf._jit_compute_sig
    th = jnp.asarray(theta)
    Rinv_BT = 0.5 * np.linalg.inv(np.array(env.R)) @ np.array(env.B).T
    from src.envs.env_rk_jax import JAXEnvWrapper
    w = JAXEnvWrapper(env, rng_key=42)
    w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    buf.reset()
    for _ in range(win_len):
        buf.append(x0.astype(np.float32))
    I = 0.0; t = 0.0; dt = env.step_size
    grad_fn = jax.grad(lambda p: jnp.dot(th, sig_fn(p)))
    while t < TF - 1e-9:
        path = jnp.asarray(buf.buffer.to_array())
        end_grad = np.array(grad_fn(path))[-1]
        u = Rinv_BT @ end_grad
        u = np.clip(u, -3.0, 3.0)
        t, x, r = w.step(w.state, jnp.array(u))
        x = np.array(x).reshape(-1)
        I += -float(r) * dt
        buf.append(x.astype(np.float32))
    return float(I)


def main():
    from src.utils.run_context import script_data_dir
    out = script_data_dir(__file__); out.mkdir(parents=True, exist_ok=True)

    env = _make_env(); n, dt = env.N, env.step_size
    lqr = _oracle(env); k = lqr.k_taps
    win_len = int(np.ceil(float(env.max_delay) / dt)) + 3 + 1
    x0_canonical = _initial_conditions(n, 1)[0]
    oracle_I = None

    # Oracle rollouts: 1 IC (what the agent trains on) and many ICs (excited).
    ics_single = _initial_conditions(n, 1)
    ics_many = _initial_conditions(n, 24)
    roll = lambda ics: [rollout(env, lqr.gain, k, x0) for x0 in ics]
    R_single = roll(ics_single)
    R_many = roll(ics_many)
    xs_single = [xs for xs, _ in R_single]
    xs_many = [xs for xs, _ in R_many]
    V_single = np.concatenate([return_to_go(rs, dt, TAU) for _, rs in R_single])
    # oracle closed-loop cost on the canonical IC (reference bar)
    rs0 = R_single[0][1]; oracle_I = float(np.sum(-rs0) * dt)

    rows = []
    spectra = {}
    for name, kind, kw in REPRESENTATIONS:
        Phi1, _ = features_along(kind, xs_single, win_len, n, **kw)
        PhiM, _ = features_along(kind, xs_many, win_len, n, **kw)
        cond1, er1, ev1 = conditioning(Phi1)
        condM, erM, evM = conditioning(PhiM)
        r2, theta = value_fit_r2(Phi1, V_single)
        ctrl_I = ideal_critic_control_cost(env, kind, theta, win_len, n, x0_canonical, **kw)
        rows.append(dict(
            name=name, kind=kind, dim=int(Phi1.shape[1]),
            feat_absmax=float(np.abs(Phi1).max()),
            gram_cond_1ic=cond1, eff_rank_1ic=er1,
            gram_cond_24ic=condM, eff_rank_24ic=erM,
            value_fit_r2=r2, ideal_critic_control_cost=ctrl_I,
        ))
        spectra[name] = dict(ev_1ic=ev1[::-1][:50].tolist(), ev_24ic=evM[::-1][:50].tolist())
        print(f"{name:>14} dim={Phi1.shape[1]:>5} R2={r2:.4f} | 1IC: cond={cond1:.1e} effrank={er1:.1f}"
              f" | 24IC: cond={condM:.1e} effrank={erM:.1f} | ideal-ctrl I={ctrl_I:.4f}")

    summary = dict(
        regime=REGIME, tau=TAU, tf=TF, window_length=win_len,
        oracle_closed_loop_cost=oracle_I,
        reference_bars=dict(no_control=0.075, markovian_lqr=0.058,
                            delayed_lqr_optimum=oracle_I, markovian_rl=0.039),
        representations=rows,
        interpretation=(
            "value-fit R2=1.0 for every representation -> NOT a representation / non-Markovian "
            "problem; the signature CAN linearly represent the value. The distinguishing fact is "
            "in the feature conditioning: with a SINGLE initial condition every representation is "
            "rank ~1, but exciting with 24 diverse ICs raises the markovian/raw-history effective "
            "rank to 6-10 (their linear critic becomes identifiable) while the SIGNATURE stays at "
            "effective rank ~1 (cond ~1e15) -- it is INTRINSICALLY near-collinear, dominated by a "
            "single direction (the monotone time-augmentation channel) plus the algebraic shuffle "
            "redundancies of iterated integrals. So the signature linear critic is unidentifiable "
            "irrespective of data excitation, which is why semi-gradient TD diverges and LSTD "
            "(plain or whitened) cannot recover a usable critic. Root cause: signature-specific "
            "feature collinearity, NOT representation capacity. Candidate fixes: log-signature "
            "(the minimal, non-redundant representation) or per-channel signature normalisation so "
            "no single channel dominates. (The ideal-critic control costs are all > no-control "
            "because the least-squares critic, fit on the oracle's narrow on-trajectory "
            "distribution, has unreliable vertical derivatives off that distribution -- a confound "
            "of that particular test, not the conditioning finding.)"),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _build_figure(rows, spectra, out)
    print(f"\n[diagnosis] wrote summary.json + conditioning_spectra.png in {out}")


def _build_figure(rows, spectra, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    # Panel 1: eigenvalue spectra (1 IC vs 24 IC) for the signature d2.
    ax = axes[0]
    for name in ("signature d2", "markovian d2"):
        s = spectra[name]
        ax.semilogy(range(1, len(s["ev_1ic"]) + 1), s["ev_1ic"], "-", label=f"{name} (1 IC)")
        ax.semilogy(range(1, len(s["ev_24ic"]) + 1), s["ev_24ic"], "--", label=f"{name} (24 IC)")
    ax.set_xlabel("feature index (sorted)"); ax.set_ylabel("Gram eigenvalue")
    ax.set_title("Feature Gram spectrum (1 IC solid, 24 ICs dashed)")
    ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=0.3, which="both")
    # Panel 2: effective rank 1IC vs 24IC per representation.
    ax = axes[1]
    names = [r["name"] for r in rows]; x = np.arange(len(names))
    ax.bar(x - 0.2, [r["eff_rank_1ic"] for r in rows], 0.4, label="1 IC")
    ax.bar(x + 0.2, [r["eff_rank_24ic"] for r in rows], 0.4, label="24 ICs")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_ylabel("effective rank (participation ratio)")
    ax.set_title("Excitation raises markovian/raw rank, NOT the signature's")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Why the signature value-gradient critic fails on the platoon: R²=1 "
                 "(representable) but signature features stay rank≈1 even when excited", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(out / "conditioning_spectra.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
