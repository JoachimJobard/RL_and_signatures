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
START_SEED=100
END_SEED=150

for (( SEED=$START_SEED; SEED<=$END_SEED; SEED++ ))
do
    echo ">>> [Exp1 - Seed $SEED] Signature depth sweep"
    
    uv run python main_unified.py -m \
        seed=${SEED} \
        agent=CTAC_sig_chemical \
        agent.training.critic_lr=1e-3 \
        agent.training.actor_lr=5e-2 \
        agent.signature.depth=2,3,4 \
        agent.training.max_time=3 \
        agent.training.n_episodes=1000 \
        agent.discount.V_bad='-0.1' \
        agent.discount.V_target='0.' \
        agent.network.std_init=0.2 \
        agent.training.eval_interval=20 \
        agent.algorithm.actor_update_frequency=2 \
        agent.noise.sigma=0.1 \
        agent.noise.schedule=decaying \
        agent.noise.schedule=constant \
        eval.snapshot_interval=100 \
        agent.algorithm.burning_steps=5 \
        agent.signature.time_augmentation=false \
        agent.algorithm.preheat=true \
        agent.signature.origin_augmentation=true \
        agent.algorithm.fix_initial_state=true \
        env=chemical_reaction_4D \
        env.environment_params.x_target=0. \
        eval.x0_test='[0.15,-0.03,0.1,0.0]' \
        eval.T_sim=2 \
        wandb.group=comparison_signature_chemical_process \
        wandb.name="signature_depth_\${agent.signature.depth}_seed_${SEED}" \
        wandb.mode=online \
        eval_data_dir=outputs/comparison_signatures_chemical_process_14_02 \
        eval_data_name="signature_depth_\${agent.signature.depth}_seed_${SEED}"
done

echo "Experiment 1 complete for seeds $START_SEED to $END_SEED"
