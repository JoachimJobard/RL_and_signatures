#!/bin/bash
# =============================================================================
# Experiment 4: Value Gradient with Signatures
# =============================================================================

set -e  # Exit on error

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "Experiment 4: Value Gradient"
echo "=============================================="

# Seed range
START_SEED=1
END_SEED=10

for (( SEED=$START_SEED; SEED<=$END_SEED; SEED++ ))
do
    echo ">>> [Exp4 - Seed $SEED] Value Gradient with signatures"
    
    uv run python main_unified.py -m \
        seed=${SEED} \
        agent=value_gradient \
        agent.training.n_episodes=200 \
        agent.training.critic_lr=1e-4 \
        agent.training.max_time=500 \
        agent.training.eval_interval=20 \
        agent.noise.schedule=decaying \
        agent.signature.depth=2,3,4 \
        agent.signature.time_augmentation=false \
        agent.signature.origin_augmentation=true \
        agent.algorithm.fix_initial_state=true \
        eval.snapshot_interval=100 \
        env=MG_1D \
        env.environment_params.x_target=0. \
        eval.x0_test=[0.1] \
        eval.T_sim=1000 \
        wandb.group=comparison_signature_mg_26_02 \
        wandb.name="vg_signature_depth_\${agent.signature.depth}_seed_${SEED}" \
        wandb.mode=offline \
        eval_data_dir=outputs/comparison_signatures_MG_26_02 \
        eval_data_name="value_gradient_depth_\${agent.signature.depth}_seed_${SEED}"
done

echo "Experiment 4 complete for seeds $START_SEED to $END_SEED"
