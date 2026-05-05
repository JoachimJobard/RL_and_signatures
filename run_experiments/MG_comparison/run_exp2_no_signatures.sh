#!/bin/bash
# =============================================================================
# Experiment 2: No Signatures (Standard CTAC, state-based)
# =============================================================================

set -e  # Exit on error

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "Experiment 2: No Signatures (state-based)"
echo "=============================================="

# Seed range
START_SEED=1
END_SEED=10

for (( SEED=$START_SEED; SEED<=$END_SEED; SEED++ ))
do
    echo ">>> [Exp2 - Seed $SEED] CTAC without signatures"
    
    uv run python main_unified.py -m \
        seed=${SEED} \
        agent=CTAC_jax \
        agent.algorithm.fix_initial_state=true \
        agent.training.max_time=500 \
        agent.training.n_episodes=200 \
        agent.training.actor_lr=1e-4 \
        agent.training.critic_lr=1e-3 \
        agent.training.eval_interval=20 \
        agent.noise.schedule=constant \
        agent.algorithm.delayed_state=false \
        eval.snapshot_interval=100 \
        env=MG_1D \
        env.environment_params.x_target=0. \
        eval.x0_test=[0.1] \
        eval.T_sim=1000 \
        wandb.group=comparison_signature_mg_26_02 \
        wandb.name="no_signature_seed_${SEED}" \
        wandb.mode=offline \
        eval_data_dir=outputs/comparison_signatures_MG_26_02 \
        eval_data_name="no_signature_seed_${SEED}"
done

echo "Experiment 2 complete for seeds $START_SEED to $END_SEED"
