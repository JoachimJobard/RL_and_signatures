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
START_SEED=100
END_SEED=150

for (( SEED=$START_SEED; SEED<=$END_SEED; SEED++ ))
do
    echo ">>> [Exp4 - Seed $SEED] Value Gradient with signatures"
    
    uv run python main_unified.py -m \
        seed=${SEED} \
        eval.x0_test='[0.15,-0.03,0.1,0.0]' \
        agent=value_gradient \
        agent.training.n_episodes=1300 \
        agent.training.max_time=3 \
        agent.training.critic_lr=1e-3 \
        agent.noise.schedule=adaptative \
        agent.discount.discounted=false \
        agent.noise.sigma=0.4 \
        agent.signature.depth=4 \
        agent.signature.time_augmentation=false \
        agent.signature.origin_augmentation=true \
        agent.algorithm.fix_initial_state=true \
        agent.algorithm.preheat=true \
        agent.algorithm.burning_steps=5\
        agent.discount.V_bad='-0.1' \
        agent.discount.V_target='1'\
        agent.network.std_init=0.1 \
        agent.training.eval_interval=10 \
        agent.noise.schedule=linear_decay \
        env=chemical_reaction_4D \
        env.environment_params.x_target=0. \
        eval.T_sim=2 \
        wandb.group=comparison_signature_chemical_process \
        wandb.name="vg_signature_depth_\${agent.signature.depth}_seed_${SEED}" \
        wandb.mode=online \
        eval_data_dir=outputs/comparison_signatures_chemical_process_16_02 \
        eval_data_name="value_gradient_signature_depth_\${agent.signature.depth}_seed_${SEED}" \
        eval.snapshot_interval=100
done

echo "Experiment 4 complete for seeds $START_SEED to $END_SEED"
