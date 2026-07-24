"""Central configuration for the paper figures — the ONE file to edit for a new campaign.

To regenerate every figure (A1/A2/A3, the comparison grid, the F2 data) for a NEW experiment campaign,
change ONLY this file: point each cell at its data group, set its clean evaluation initial condition
(x0, burn_steps), its oracle cost reduction (the %opt denominator), and the representative seed. Then
re-run generate_all.slurm. Every downstream script reads the cell table from here — no other edit is
needed, and no group name, oracle value, or IC recipe is hard-coded anywhere else.

DATA_ROOT and each group_glob point into the byte-verified $WORK snapshot; see the snapshot README for
the directory layout. group_glob is relative to DATA_ROOT and may span several run groups (e.g. the
two seed-half groups of a 10-seed final).
"""
import os

# Root of the campaign data snapshot. Override with PAPER_FIGURES_DATA_ROOT (the launcher passes it,
# resolving $WORK from a login shell).
DATA_ROOT = os.environ.get(
    "PAPER_FIGURES_DATA_ROOT",
    "/lustre/fswork/projects/rech/oym/ucd32aq/paper_data/aaai2027_h1_clean_20260724",
)

# Held-out test seed used for the single-run illustrations (A3, the comparison grid). Seed-independent
# figures (A1, and the F2 distribution) do not use it.
REPRESENTATIVE_SEED = 3

# The cells of the study, in order of increasing delay. Each entry is a complete, self-contained recipe.
#   title                  : human-readable cell name for figure titles
#   group_glob             : run-group glob relative to DATA_ROOT (may match several groups)
#   oracle_cost_reduction  : the analytic delayed-LQR cost reduction (%), i.e. the denominator of
#                            eta = cost_reduction / oracle_cost_reduction
#   x0                     : the fixed evaluation initial condition (env-dimensional)
#   burn_steps             : uncontrolled burn-in before the cost window opens (0 = start at x0;
#                            >0 places the delayed cell on its developed attractor)
CELLS = {
    "harmonic": dict(
        title="Harmonic oscillator (ODE, delay 0)",
        group_glob="final_dt0.25/final_newengine_harmonic_oscillator_*",
        oracle_cost_reduction=94.275, x0=[1.0, 1.0], burn_steps=0,
    ),
    "linear_dde": dict(
        title="Linear DDE (delay 1)",
        group_glob="final_dt0.25/final_newengine_linear_dde_scalar_*",
        oracle_cost_reduction=94.057, x0=[1.0], burn_steps=0,
    ),
    "MG": dict(
        title="Mackey-Glass limit cycle (delay 6)",
        group_glob="final_dt0.05/final_dt0p05_clean_*",
        oracle_cost_reduction=99.198, x0=[0.8], burn_steps=2000,
    ),
}

LEARNERS = ["value_gradient", "policy_gradient", "actor_critic"]
LEARNER_LABEL = {"value_gradient": "value gradient", "policy_gradient": "policy gradient",
                 "actor_critic": "actor-critic"}
KINDS = ["markovian", "raw_history", "signature"]
KIND_LABEL = {"markovian": "markovian", "raw_history": "raw history", "signature": "signature"}

# Representation colours (categorical, consistent across every figure). Grey / orange / blue.
KIND_COLOUR = {"markovian": "#6E6E6E", "raw_history": "#D95F02", "signature": "#1B66B4"}
# Stroke convention (see the repo plotting convention): solid = trained agent, dashed = analytic
# reference (the delayed-LQR oracle), dotted = auxiliary (no-control baseline, target line).
AGENT_COLOUR, ORACLE_COLOUR, NOCTRL_COLOUR, TARGET_COLOUR = "#1f77b4", "#2E7D5B", "#8c8c8c", "#333333"
