#!/bin/bash
# =============================================================================
# Experiment 1: Signature-based CTAC (depth sweep)
# =============================================================================

set -e  # Exit on error

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "Experiment 1: Signature-based CTAC"
echo "=============================================="

# Seed range
START_SEED=1
END_SEED=10

for (( SEED=$START_SEED; SEED<=$END_SEED; SEED++ ))
do
    echo ">>> [Exp1 - Seed $SEED] Signature depth sweep"
    
    uv run python main_unified.py -m \
        seed=${SEED} \
        agent=CTAC_sig_MG_1D \
        agent.training.actor_lr=1e-4 \
        agent.training.critic_lr=1e-3 \
        agent.training.max_time=1000 \
        agent.training.n_episodes=500 \
        agent.training.eval_interval=20 \
        agent.noise.schedule=constant \
        agent.signature.depth=2,3,4 \
        agent.signature.time_augmentation=false \
        agent.signature.origin_augmentation=true \
        agent.algorithm.fix_initial_state=true \
        env=MG_1D \
        env.environment_params.x_target=0. \
        eval.x0_test=[0.1] \
        eval.T_sim=1000 \
        eval.snapshot_interval=100 \
        wandb.group=comparison_signature_mg_26_02 \
        wandb.name="signature_depth_\${agent.signature.depth}_seed_${SEED}" \
        wandb.mode=offline \
        eval_data_dir=outputs/comparison_signatures_MG_26_02 \
        eval_data_name="signature_depth_\${agent.signature.depth}_seed_${SEED}"
done

echo "Experiment 1 complete for seeds $START_SEED to $END_SEED"
