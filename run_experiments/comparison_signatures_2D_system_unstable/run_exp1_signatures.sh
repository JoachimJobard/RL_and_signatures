#!/bin/bash
# =============================================================================
# Experiment 1: Signature with P. ARRIBAS augmentation (Depth 2)
# =============================================================================

set -e  # Exit on error

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "Experiment 1: Signature with augmentation"
echo "=============================================="

# Seed range
START_SEED=1
END_SEED=10

for (( SEED=$START_SEED; SEED<=$END_SEED; SEED++ ))
do
    echo ">>> [Exp1 - Seed $SEED] Signature depth 2"
    
    uv run python main_unified.py -m \
        seed=${SEED} \
        agent=signatures \
        agent.training.critic_lr=1e-4 \
        agent.training.actor_lr=5e-3 \
        agent.signature.depth=2,3,4 \
        agent.training.max_time=25 \h
        agent.training.n_episodes=1000 \
        agent.network.std_init=0.1 \
        agent.training.eval_interval=20 \
        agent.algorithm.actor_update_frequency=2 \
        agent.noise.sigma=0.1 \
        agent.noise.schedule=constant \
        agent.discount.discounted=true \
        agent.discount.tau=3 \
        eval.snapshot_interval=100 \
        agent.algorithm.burning_steps=5 \
        agent.signature.time_augmentation=false \
        agent.algorithm.preheat=true \
        agent.signature.origin_augmentation=true \
        agent.algorithm.fix_initial_state=true \
        eval.T_sim=20 \
        env=delay_jax_unstable \
        wandb.group=comparison_signature_2D_unstable_discounted \
        wandb.name="signature_depth_\${agent.signature.depth}_seed_${SEED}" \
        wandb.mode=online \
        eval_data_dir=outputs/comparison_signatures_2D_system_unstable_discounted \
        eval_data_name="signature_depth_\${agent.signature.depth}_seed_${SEED}"

done

echo "Experiment 1 completed for all seeds."
