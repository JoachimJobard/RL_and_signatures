"""3x3 comparison grid on the CLEAN fixed initial condition, re-simulated from checkpoints (not read
from the buggy random-IC eval.pkl). Title per panel = relative optimality eta = (J0 - J)/(J0 - Jstar)
rendered as a percentage; blue if the controller helps (eta>0), red if it diverges (eta<=0). Cost is
minimised (sign convention): J is the discounted cost, J0 the no-control cost, Jstar the oracle.
Usage: comparison_clean.py <data_root> <env> <seed> <x0 comma-sep> <burn> <oracle_costred> <out.png>"""
import sys, os, glob
import numpy as np
from omegaconf import OmegaConf
import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from src.training.train import build_agent
from src.training.evaluate import collect_evaluation_data

D, ENV = sys.argv[1], sys.argv[2]
SEED = int(sys.argv[3])
X0 = np.array([float(v) for v in sys.argv[4].split(",")], dtype=float)
BURN = int(sys.argv[5]); ORC = float(sys.argv[6]); OUT = sys.argv[7]
LEARN = ["value_gradient", "policy_gradient", "actor_critic"]
LAB = {"value_gradient": "value gradient", "policy_gradient": "policy gradient", "actor_critic": "actor–critic"}
KIND = ["markovian", "raw_history", "signature"]
KLAB = {"markovian": "markovian", "raw_history": "raw history", "signature": "signature"}
CELL = {"harmonic_oscillator": "harmonic oscillator (ODE)", "linear_dde_scalar": "linear DDE", "MG_1D_limit_cycle": "Mackey–Glass limit cycle"}
GOOD, BAD = "#1f77b4", "#c1352b"

fig, ax = plt.subplots(3, 3, figsize=(13.5, 10.2), sharex=True)
fig.suptitle(f"{CELL.get(ENV, ENV)}  ·  held-out seed {SEED}  ·  clean fixed initial condition  ·  controlled state per learner × representation",
             fontsize=12.5, y=0.995)
for r, learner in enumerate(LEARN):
    for c, kind in enumerate(KIND):
        a = ax[r, c]
        ds = sorted(glob.glob(f"{D}/final_newengine_{ENV}_*/*_{learner}_{kind}_*_seed{SEED}"))
        if not ds:
            a.text(0.5, 0.5, "missing", ha="center", va="center", color="0.6"); continue
        cfg = OmegaConf.load(f"{ds[0]}/config.yaml")
        env = hydra.utils.instantiate(cfg.env.environment_params)
        agent = build_agent(cfg, env)
        agent.load(f"{ds[0]}/checkpoint_agent.pkl")
        ed = collect_evaluation_data(agent, X0, float(cfg.eval.T_sim), burning_steps=BURN)
        t = np.asarray(ed["times"]).ravel()
        xa = np.asarray(ed["states_agent"]).reshape(len(t), -1)[:, 0]
        xn = np.asarray(ed["states_no_ctrl"]).reshape(len(t), -1)[:, 0]
        cr = ed["eval_metrics"]["eval/cost_reduction_pct"]
        eta = cr / ORC * 100 if np.isfinite(cr) else float("nan")
        col = GOOD if (np.isfinite(eta) and eta > 0) else BAD
        xref = ed.get("x_ref"); xref = np.ravel(np.asarray(xref)) if xref is not None else None
        yt = float(xref[0]) if (xref is not None and xref.size) else 0.0
        lo = min(float(np.min(xn)), yt); hi = max(float(np.max(xn)), yt)
        span = (hi - lo) if hi > lo else 1.0; pad = 0.35 * span + 0.15
        ylo, yhi = lo - pad, hi + pad
        xa_plot = np.where((xa >= ylo) & (xa <= yhi), xa, np.nan)
        a.plot(t, xa_plot, "-", lw=1.5, color=col)
        a.plot(t, xn, "--", lw=1.0, color="0.6")
        a.axhline(yt, ls=":", lw=1, color="k")
        a.set_ylim(ylo, yhi)
        shown = f"η={eta:.0f}%" if (np.isfinite(eta) and eta > -1000) else "η≪0"
        a.set_title(shown, color=col, fontsize=12, fontweight="bold", pad=3)
        if r == 0:
            a.annotate(KLAB[kind], xy=(0.5, 1.28), xycoords="axes fraction", ha="center", fontsize=12, color="0.25")
        if c == 0:
            a.set_ylabel(LAB[learner], fontsize=12, color="0.25")
        if r == 2:
            a.set_xlabel("time")
        print(f"{ENV} {learner} {kind} seed{SEED}: eta={eta:.1f}%")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=125, bbox_inches="tight")
print(f"saved {OUT}")
