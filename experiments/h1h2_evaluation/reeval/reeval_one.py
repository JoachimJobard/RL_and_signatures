"""Re-evaluate a trained run from its checkpoint under a chosen initial condition + burn-in, WITHOUT
retraining. Reuses the tested build_agent + collect_evaluation_data. Prints the stored (original)
cost reduction and the re-evaluated one, so the reload can be validated by reproducing the stored
number with the stored IC (burn 0), then a clean fixed IC + burn can be applied.

Usage: reeval_one.py <run_dir> <x0 comma-sep | 'stored'> <burn_steps>
"""
import sys, os, pickle
import numpy as np
from omegaconf import OmegaConf
import hydra

sys.path.insert(0, os.getcwd())
from src.training.train import build_agent
from src.training.evaluate import collect_evaluation_data

run_dir = sys.argv[1].rstrip("/")
x0_arg = sys.argv[2]
burn = int(sys.argv[3])

cfg = OmegaConf.load(f"{run_dir}/config.yaml")
stored = pickle.load(open(f"{run_dir}/eval.pkl", "rb"))
T_sim = float(cfg.eval.T_sim)
stored_ic = np.asarray(stored["states_no_ctrl"]).reshape(len(np.asarray(stored["times"]).ravel()), -1)[0]
stored_j0 = float(np.asarray(stored["cum_cost_no_ctrl"])[-1])
stored_ja = float(np.asarray(stored["cum_cost_agent"])[-1])
stored_cr = 100 * (stored_j0 - stored_ja) / stored_j0 if stored_j0 > 0 else float("nan")

env = hydra.utils.instantiate(cfg.env.environment_params)
agent = build_agent(cfg, env)
agent.load(f"{run_dir}/checkpoint_agent.pkl")

if x0_arg == "stored":
    x0 = np.asarray(stored_ic, dtype=float)
else:
    x0 = np.array([float(v) for v in x0_arg.split(",")], dtype=float)

ed = collect_evaluation_data(agent, x0, T_sim, burning_steps=burn)
j0 = float(np.asarray(ed["cum_cost_no_ctrl"])[-1])
ja = float(np.asarray(ed["cum_cost_agent"])[-1])
cr = ed["eval_metrics"]["eval/cost_reduction_pct"]
ic_used = np.asarray(ed["states_no_ctrl"]).reshape(len(np.asarray(ed["times"]).ravel()), -1)[0]

print(f"RUN {os.path.basename(run_dir)}")
print(f"  STORED : IC={stored_ic}  J0={stored_j0:.4f}  Jagent={stored_ja:.4e}  cost_red={stored_cr:.2f}%")
print(f"  REEVAL : IC={ic_used}  burn={burn}  J0={j0:.4f}  Jagent={ja:.4e}  cost_red={cr:.2f}%")
if x0_arg == "stored" and burn == 0:
    dcr = abs(cr - stored_cr)
    print(f"  VALIDATION (stored IC, burn 0): |Δcost_red| = {dcr:.4f}  -> {'PASS' if dcr < 0.5 else 'FAIL'}")
