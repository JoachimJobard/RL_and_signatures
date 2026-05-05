#!/bin/bash
# =============================================================================
# Experiment 2: No Signatures (Standard CTAC)
# =============================================================================

set -e  # Exit on error

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "Experiment 2: No Signatures (Standard CTAC)"
echo "=============================================="

# Seed range
START_SEED=1
END_SEED=10

for (( SEED=$START_SEED; SEED<=$END_SEED; SEED++ ))
do
    echo ">>> [Exp2 - Seed $SEED] No signatures"
    
    uv run python main_unified.py -m \
        seed=${SEED} \
        agent=CTAC_jax \
        agent.algorithm.fix_initial_state=true \
        agent.training.max_time=25 \
        agent.training.critic_lr=1e-3 \
        agent.training.actor_lr=1e-3 \
        agent.training.n_episodes=1500 \
        agent.noise.sigma=0.2 \
        agent.discount.V_bad='-0.1' \
        agent.discount.V_target='0.' \
        agent.discount.discounted=false \
        agent.algorithm.actor_update_frequency=2 \
        agent.network.std_init=0.1 \
        agent.training.eval_interval=10 \
        agent.noise.schedule=decaying \
        eval.snapshot_interval=100 \
        agent.algorithm.burning_steps=5 \
        agent.algorithm.delayed_state=false \
        agent.algorithm.preheat=true \
        eval.T_sim=20 \
        env=delay_jax_unstable \
        wandb.group=comparison_signature_2D_unstable_discounted \
        wandb.name="no_signature_seed_${SEED}" \
        wandb.mode=online \
        eval_data_dir=outputs/comparison_signatures_2D_system_unstable \
        eval_data_name="base_no_delay_seed_${SEED}"

done

echo "Experiment 2 completed for all seeds."
