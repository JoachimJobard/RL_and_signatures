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
START_SEED=100
END_SEED=110

for (( SEED=$START_SEED; SEED<=$END_SEED; SEED++ ))
do
    echo ">>> [Exp2 - Seed $SEED] CTAC without signatures"
    
    uv run python main_unified.py -m \
        seed=${SEED} \
        agent=CTAC_jax \
        agent.algorithm.fix_initial_state=true \
        agent.training.max_time=3 \
        agent.training.critic_lr=1e-3 \
        agent.training.actor_lr=1e-2 \
        agent.training.n_episodes=1500 \
        agent.noise.sigma=0.2 \
        agent.discount.V_bad='-0.1' \
        agent.discount.V_target='0.' \
        agent.algorithm.actor_update_frequency=2 \
        agent.network.std_init=0.2 \
        agent.training.eval_interval=20 \
        agent.noise.schedule=decaying \
        eval.snapshot_interval=100 \
        agent.algorithm.burning_steps=5 \
        agent.algorithm.delayed_state=false \
        agent.algorithm.preheat=true \
        env=chemical_reaction_4D \
        env.environment_params.x_target=0. \
        eval.x0_test='[0.15,-0.03,0.1,0.0]' \
        eval.T_sim=2 \
        wandb.group=comparison_signature_chemical_process \
        wandb.name="no_signature_seed_${SEED}" \
        wandb.mode=online \
        eval_data_dir=outputs/comparison_signatures_chemical_process_26_02_test_grad \
        eval_data_name="base_no_delay_seed_${SEED}"
done

echo "Experiment 2 complete for seeds $START_SEED to $END_SEED"
