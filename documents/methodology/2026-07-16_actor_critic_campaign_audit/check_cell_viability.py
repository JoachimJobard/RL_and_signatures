"""Viability gate: does at least one learned controller exist on each campaign cell?

The question this answers, and why it precedes the campaign. Each cell costs 30 SLURM tasks
(3 representations x 5 seeds x 2 learners). If no learner produces a working controller on a cell,
that cell's H1 verdict is not a scientific finding but an artefact of a plant nobody can control,
and its 30 tasks are wasted. This runs ONE seed per cell and asks only the weakest useful question:
is the plant controllable AT ALL by this stack?

METRIC. The evaluation reports `eval/cost_reduction_pct`, the closed-loop cost of the learned
controller against the SAME plant left uncontrolled, at the deployment initial condition and
horizon. A positive value means the controller helps; a negative value means it is worse than doing
nothing. The gate is deliberately weak: > 0 is "a controller exists", not "the controller is good".

WHAT A NEGATIVE RESULT DOES AND DOES NOT MEAN. This runs a REDUCED episode budget against the
campaign's 1000, so a cell that fails here may still succeed at full budget: a negative is
INCONCLUSIVE and is reported as such, never as "the cell is unlearnable". A positive is conclusive
in the direction that matters -- it proves a controller exists.

Every run carries debug=true, so its output directory is prefixed `_debug_` and can be wiped with
    find data -type d -name '_debug_*' -prune -exec rm -rf {} +
without touching real data. This is an exploratory code-path/viability check, which is exactly the
category the project rule reserves the flag for.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path("/home/rho/Documents/work/phd/git_repositories/RL_and_signatures")
PY = str(REPO / ".venv" / "bin" / "python")

# The array's own per-cell table (experiments/h1h2_evaluation/jeanzay/rl_array.slurm:73-81),
# reproduced here so the gate runs the campaign's plants and not something adjacent.
CELLS = {
    "markovian":          dict(env="double_integrator",  win=10, maxt=15.0, tsim=15.0, x0="[1.0,0.5]"),
    "linear_dde":         dict(env="linear_dde_scalar",  win=8,  maxt=20.0, tsim=20.0, x0="[0.8]"),
    "hopfield_linear":    dict(env="hopfield_linear",    win=5,  maxt=15.0, tsim=15.0, x0="[0.5,0.0]"),
    "hopfield_nonlinear": dict(env="hopfield_nonlinear", win=5,  maxt=15.0, tsim=15.0, x0="[0.5,0.0]"),
    "mg_chaotic":         dict(env="MG_1D_chaotic",      win=71, maxt=60.0, tsim=60.0, x0="[0.8]"),
}

REP_ARG = {
    "markovian":   "agent.signature.kind=markovian agent.signature.degree=2",
    "raw_history": "agent.signature.kind=raw_history agent.signature.degree=2",
    "signature":   "agent.signature.kind=signature agent.signature.depth=2",
}


def run_one(cell: str, rep: str, agent: str, n_episodes: int, seed: int = 0) -> dict:
    spec = CELLS[cell]
    cmd = [
        PY, "main_unified.py",
        f"agent={agent}", f"env={spec['env']}",
        *REP_ARG[rep].split(),
        f"agent.signature.window_size={spec['win']}",
        "agent.signature.force_signature_window=true",
        "agent.training.divergence_threshold=100.0",
        f"agent.training.n_episodes={n_episodes}",
        f"agent.training.max_time={spec['maxt']}",
        f"eval.x0_test={spec['x0']}", f"eval.T_sim={spec['tsim']}",
        f"seed={seed}",
        "experiment_group=_debug_viability_gate",
        f"run_tag=viability_{agent}_{cell}_{rep}",
        "debug=true", "wandb.mode=offline",
    ]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=5400)
    reduction = None
    for line in proc.stdout.splitlines():
        if "Cost reduction:" in line:
            try:
                reduction = float(line.split("Cost reduction:")[1].strip().rstrip("%"))
            except (ValueError, IndexError):
                pass
    return {
        "cell": cell, "rep": rep, "agent": agent, "n_episodes": n_episodes,
        "cost_reduction_pct": reduction, "returncode": proc.returncode,
        "stderr_tail": proc.stderr.strip().splitlines()[-3:] if proc.returncode else [],
    }


if __name__ == "__main__":
    agent = sys.argv[1] if len(sys.argv) > 1 else "value_gradient"
    rep = sys.argv[2] if len(sys.argv) > 2 else "signature"
    n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    only = sys.argv[4] if len(sys.argv) > 4 else None

    cells = [only] if only else list(CELLS)
    print(f"VIABILITY GATE | agent={agent} rep={rep} n_episodes={n_episodes} seed=0")
    print(f"gate: cost_reduction_pct > 0 means a controller EXISTS (weak, deliberately).")
    print(f"NOTE: {n_episodes} episodes against the campaign's 1000 -- a NEGATIVE here is "
          f"INCONCLUSIVE, a positive is conclusive.\n")
    results = []
    for cell in cells:
        r = run_one(cell, rep, agent, n_episodes)
        results.append(r)
        red = r["cost_reduction_pct"]
        if r["returncode"] != 0:
            verdict = f"CRASHED (rc={r['returncode']}) {r['stderr_tail']}"
        elif red is None:
            verdict = "NO METRIC PARSED -- not measured"
        elif red > 0:
            verdict = f"controller EXISTS (cost reduction {red:+.2f}%)"
        else:
            verdict = f"NO controller at this budget (cost reduction {red:+.2f}%) -- INCONCLUSIVE"
        print(f"  {cell:<20} {verdict}", flush=True)

    print("\nSUMMARY")
    for r in results:
        print(f"  RESULT {r['agent']}/{r['cell']}/{r['rep']} "
              f"cost_reduction_pct={r['cost_reduction_pct']} rc={r['returncode']}")
