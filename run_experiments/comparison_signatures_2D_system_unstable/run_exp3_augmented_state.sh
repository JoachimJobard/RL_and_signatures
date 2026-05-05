#!/bin/bash
# =============================================================================
# Experiment 3: Augmented State without Signatures
# =============================================================================

set -e  # Exit on error

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "Experiment 3: Augmented State (no signatures)"
echo "=============================================="

# Seed range
START_SEED=1
END_SEED=10

for (( SEED=$START_SEED; SEED<=$END_SEED; SEED++ ))
do
    echo ">>> [Exp3 - Seed $SEED] Augmented state without signatures"
    
    uv run python main_unified.py -m \
        seed=${SEED} \
        agent=CTAC_jax \
        agent.algorithm.fix_initial_state=true \
        agent.training.max_time=25 \
        agent.training.critic_lr=1e-3 \
        agent.training.actor_lr=1e-2 \
        agent.training.n_episodes=1500 \
        agent.noise.sigma=0.2 \
        agent.discount.V_bad='-0.1' \
        agent.discount.V_target='0.' \
        agent.network.std_init=0.1 \
        agent.training.eval_interval=20 \
        agent.noise.schedule=decaying \
        eval.snapshot_interval=100 \
        agent.algorithm.actor_update_frequency=2 \
        agent.algorithm.burning_steps=5 \
        agent.algorithm.delayed_state=true \
        agent.algorithm.preheat=true \
        eval.T_sim=20 \
        env=delay_jax_unstable \
        wandb.group=comparison_signature_2D_unstable \
        wandb.name="no_signature_augmented_state_seed_${SEED}" \
        wandb.mode=online \
        eval_data_dir=outputs/comparison_signatures_2D_system_unstable \
        eval_data_name="base_augmented_state_seed_${SEED}"

done

echo "Experiment 3 completed for all seeds."
