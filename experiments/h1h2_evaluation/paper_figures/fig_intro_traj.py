"""Intro-teaser data (paper Figure 1). For one cell and one learner, re-simulate from the saved
checkpoints, on the cell's fixed clean initial condition at the representative seed, the controlled
state of EACH representation (markovian / raw history / signature), together with the analytic
delayed-LQR oracle and the no-control baseline. Reuses the tested _figlib.reeval_run / find_oracle, so
no evaluation logic is re-implemented and nothing is retrained.

Emits a tidy CSV: time, no_control, oracle, markovian_oracle, markovian, raw_history, signature -- the
illustrative curves of Figure 1 (target is the constant 0). The paper figure is rebuilt from this CSV.
The markovian_oracle column is the delay-blind LQR reference (see below), the others come from the
saved checkpoints / the analytic delayed-LQR oracle.

Usage: python fig_intro_traj.py <out_dir> [cell_name=linear_dde] [learner=value_gradient]
"""
import os
import sys
import glob
import csv as _csv
import numpy as np
import campaign as C
import _figlib as F


def find_oracle_any(cell, name):
    """The delayed-LQR reference run: primarily in the cell's run group; falling back to the
    oracle_audit tree, where the analytic-oracle reference actually lives in this snapshot."""
    orc = F.find_oracle(C.DATA_ROOT, cell["group_glob"])
    if orc is not None:
        return orc
    hits = sorted(glob.glob(f"{C.DATA_ROOT}/oracle_audit/*{name}*/*oracle_reference*"))
    return hits[0] if hits else None

out_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PAPER_FIGURES_OUT", "figures_out")
name = sys.argv[2] if len(sys.argv) > 2 else "linear_dde"
learner = sys.argv[3] if len(sys.argv) > 3 else "value_gradient"
cell = C.CELLS[name]
SEED = C.REPRESENTATIVE_SEED

t_ref = None
xn_ref = None
cfg_ref = None
controlled = {}
for kind in C.KINDS:
    rd = F.find_run(C.DATA_ROOT, cell["group_glob"], learner, kind, SEED)
    if rd is None:
        print(f"  MISSING run: {learner} {kind} seed{SEED}")
        continue
    ed, cfg_ref = F.reeval_run(rd, cell["x0"], cell["burn_steps"])
    t, xa, xn = F.state0(ed)
    controlled[kind] = xa
    t_ref, xn_ref = t, xn
    print(f"  {kind:12s} eta={F.eta_of(ed, cell['oracle_cost_reduction']):.1f}%  ({rd.split('/')[-1]})")

if t_ref is None:
    sys.exit("no runs found -- check DATA_ROOT and the cell group_glob")

# Analytic delayed-LQR oracle on the same IC (closed-form controller; no checkpoint to restore).
oracle = np.full_like(t_ref, np.nan)
orc = find_oracle_any(cell, name)
if orc is not None:
    edo, _ = F.reeval_run(orc, cell["x0"], cell["burn_steps"], load_checkpoint=False)
    _, oracle, _ = F.state0(edo)
    print(f"  oracle       ({orc.split('/')[-1]})")
else:
    print("  oracle       MISSING")

# Markovian-LQR oracle (the canonical solvers.delayed_lqr.markovian_oracle_gap definition): the
# delay-blind LQR -- synthesised for the instantaneous dynamics alone (the delayed matrix set to
# zero), discounted with the same rate as the delayed oracle -- rolled on the delayed plant from
# the same fixed IC. Closed-form and deterministic (no checkpoint); defined only for the linear
# JAXDDEEnv cell, whose A, A1, B, Q, R are read from the run config. Euler at the control step,
# exactly as markovian_oracle_gap, so the value equals that diagnostic's Markovian rollout.
markovian_oracle = np.full_like(t_ref, np.nan)
ep = cfg_ref.env.environment_params if cfg_ref is not None else None
if ep is not None and str(ep.get("_target_", "")).endswith("JAXDDEEnv"):
    from src.solvers.delayed_lqr import augmented_discrete_lqr
    from omegaconf import OmegaConf
    mat = lambda key: np.atleast_2d(np.array(OmegaConf.to_container(ep[key].object), dtype=float))
    A, A1, B, Q, R = (mat(k) for k in ("A", "A1", "B", "Q", "R"))
    delay = float(np.array(OmegaConf.to_container(ep.delay.object), dtype=float).ravel()[0])
    dt = float(ep.step_size)
    gamma = float(OmegaConf.select(cfg_ref, "agent.discount.gamma") or 0.0)
    beta = float(np.exp(-gamma * dt))
    gain = augmented_discrete_lqr(A, np.zeros_like(A), B, Q, R, 0.0, dt, discount_beta=beta).gain
    k = int(round(delay / dt))
    window = np.tile(np.array(cell["x0"], dtype=float).reshape(-1), (k + 1, 1))
    traj = [float(window[0, 0])]
    for _ in range(len(t_ref) - 1):
        x = window[0]; x_delayed = window[k]; u = -gain @ x
        x_next = x + dt * (A @ x + A1 @ x_delayed + B @ u)
        window = np.vstack([x_next.reshape(1, -1), window[:-1]]); traj.append(float(x_next[0]))
    markovian_oracle = np.array(traj)
    print(f"  markovian_oracle  (delay-blind LQR K={float(gain.ravel()[0]):.3f}, beta={beta:.4f})")

cols = [("time", t_ref), ("no_control", xn_ref), ("oracle", oracle),
        ("markovian_oracle", markovian_oracle)]
for kind in C.KINDS:
    cols.append((kind, controlled.get(kind, np.full_like(t_ref, np.nan))))
header = [h for h, _ in cols]

os.makedirs(out_dir, exist_ok=True)
stem = f"{out_dir}/fig_intro_{name}_{learner}_seed{SEED}"
with open(f"{stem}.csv", "w", newline="") as fh:
    w = _csv.writer(fh)
    w.writerow(header)
    for i in range(len(t_ref)):
        w.writerow([f"{c[i]:.6g}" for _, c in cols])
print(f"wrote {stem}.csv")

# Figure (matplotlib, reusing _figlib's rcParams). Stroke convention: solid = trained
# representation, dashed = analytic delayed-LQR oracle, dotted = setpoint; the uncontrolled
# baseline is a muted grey solid. Palette matches the other paper figures.
PALETTE = {"markovian": "#2a78d6", "raw_history": "#eb6834", "signature": "#1baf7a"}
fig, ax = F.plt.subplots(figsize=(6.4, 3.5))
ax.plot(t_ref, xn_ref, "-", lw=1.2, color="#9aa0a6", label="uncontrolled")
if np.isfinite(oracle).any():
    ax.plot(t_ref, oracle, "--", lw=1.4, color="#333333", label="delayed-LQR oracle")
if np.isfinite(markovian_oracle).any():
    ax.plot(t_ref, markovian_oracle, "--", lw=1.4, color="#2a78d6", label="markovian-LQR oracle")
for kind in C.KINDS:
    if kind in controlled:
        ax.plot(t_ref, controlled[kind], "-", lw=1.7, color=PALETTE[kind], label=C.KIND_LABEL[kind])
ax.axhline(0.0, ls=":", lw=1.0, color="#333333")
ax.set_xlabel("time")
ax.set_ylabel("state")
ax.set_title(f"{cell['title']}  ·  seed {SEED}  ·  {C.LEARNER_LABEL[learner]}", fontsize=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=8, frameon=True)
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(f"{stem}.pdf")
fig.savefig(f"{stem}.png")
F.plt.close(fig)
print(f"wrote {stem}.pdf")
