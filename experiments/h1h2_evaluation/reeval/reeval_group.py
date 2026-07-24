"""Re-evaluate every run in a campaign group under a clean fixed initial condition + burn-in, from
checkpoints, without retraining. Writes <group>/reeval_clean/summary.tsv with the clean cost
reduction and %opt (denominator = the group's analytic oracle re-evaluated on the SAME clean IC).

Usage: reeval_group.py <group_dir> <x0 comma-sep> <burn_steps>
Reuses the tested build_agent + collect_evaluation_data; the eval IC is passed explicitly, so the
stored (buggy) config.eval.x0_test is bypassed and the trained params are untouched."""
import sys, os, glob, re, pickle
import numpy as np
from omegaconf import OmegaConf
import hydra

sys.path.insert(0, os.getcwd())
from src.training.train import build_agent
from src.training.evaluate import collect_evaluation_data

GROUP = sys.argv[1].rstrip("/")
X0 = np.array([float(v) for v in sys.argv[2].split(",")], dtype=float)
BURN = int(sys.argv[3])
OUT = f"{GROUP}/reeval_clean"
os.makedirs(OUT, exist_ok=True)

rx = re.compile(r"_(value_gradient|policy_gradient|actor_critic)_(markovian|raw_history|signature)_lr([0-9.]+)(?:_c([0-9.]+))?_seed(\d+)$")


def reeval(run_dir, load_ckpt=True):
    cfg = OmegaConf.load(f"{run_dir}/config.yaml")
    env = hydra.utils.instantiate(cfg.env.environment_params)
    agent = build_agent(cfg, env)
    if load_ckpt and os.path.exists(f"{run_dir}/checkpoint_agent.pkl"):
        agent.load(f"{run_dir}/checkpoint_agent.pkl")
    ed = collect_evaluation_data(agent, X0, float(cfg.eval.T_sim), burning_steps=BURN)
    j0 = float(np.asarray(ed["cum_cost_no_ctrl"])[-1])
    ja = float(np.asarray(ed["cum_cost_agent"])[-1])
    cr = ed["eval_metrics"]["eval/cost_reduction_pct"]
    return cr, j0, ja


# 1) oracle denominator on the clean IC (analytic controller from its own config; no checkpoint needed)
oracle_dirs = glob.glob(f"{GROUP}/*oracle_reference*")
orc_cr = None
if oracle_dirs:
    try:
        orc_cr, oj0, oja = reeval(oracle_dirs[0], load_ckpt=False)
        print(f"ORACLE clean cost_red = {orc_cr:.3f}%  (J0={oj0:.4f})")
    except Exception as e:
        print(f"ORACLE re-eval FAILED: {type(e).__name__}: {e}")

# 2) every combo run
rows = ["learner\tkind\tlr\tseed\tcost_red\tpct_opt\tj0\tdiverged"]
run_dirs = [d for d in glob.glob(f"{GROUP}/2026*_seed*") if "oracle" not in d]
for d in sorted(run_dirs):
    m = rx.search(os.path.basename(d.rstrip("/")))
    if not m:
        continue
    learner, kind, lr, clr, seed = m.groups()
    lrlab = f"{lr}/{clr}" if clr else lr
    try:
        cr, j0, ja = reeval(d)
        pct = (cr / orc_cr * 100) if orc_cr else float("nan")
        div = (not np.isfinite(cr)) or (cr < -1e6) or (ja > 1e3 * max(j0, 1e-9))
        rows.append(f"{learner}\t{kind}\t{lrlab}\t{seed}\t{cr:.3f}\t{pct:.2f}\t{j0:.4f}\t{int(bool(div))}")
    except Exception as e:
        rows.append(f"{learner}\t{kind}\t{lrlab}\t{seed}\tERR\tERR\tERR\t1")
        print(f"  {os.path.basename(d)} FAILED: {type(e).__name__}: {e}")

open(f"{OUT}/summary.tsv", "w").write("\n".join(rows) + "\n")
print(f"wrote {OUT}/summary.tsv  ({len(rows)-1} runs, oracle={orc_cr})")
