# Mackey-Glass (MG) Comparison Experiments

This directory contains scripts to run parallel comparison experiments for the Mackey-Glass 1D environment.

## Experiments

1. **Signature-based CTAC** (`run_exp1_signatures.sh`)
   - Agent: `CTAC_sig_MG_1D`
   - Depth sweep: 2, 3, 4
   - Uses path signatures with origin augmentation
   - actor_lr=1e-4, critic_lr=1e-3

2. **No Signatures (State-based)** (`run_exp2_no_signatures.sh`)
   - Agent: `CTAC_jax`
   - Standard actor-critic on raw state
   - No delay augmentation (`delayed_state=false`)
   - actor_lr=1e-4, critic_lr=1e-3

3. **Augmented State (Delayed)** (`run_exp3_augmented_state.sh`)
   - Agent: `CTAC_jax`
   - State concatenated with delayed state (`delayed_state=true`)
   - actor_lr=1e-4, critic_lr=1e-3

4. **Value Gradient** (`run_exp4_value_gradient.sh`)
   - Agent: `value_gradient`
   - Gradient-based policy (no explicit actor network)
   - critic_lr=1e-3, depth=2

## Usage

### Run individual experiments

```bash
cd run_experiments/MG_comparison

./run_exp1_signatures.sh
./run_exp2_no_signatures.sh
./run_exp3_augmented_state.sh
./run_exp4_value_gradient.sh
```

### Run all experiments in parallel

```bash
# Launch all 4 experiments as background jobs
./run_all_parallel.sh

# Launch specific experiments only (e.g., 1 and 4)
./run_all_parallel.sh 1 4
```

### Monitor progress

```bash
tail -f logs/exp1_*.log
tail -f logs/exp2_*.log
tail -f logs/exp3_*.log
tail -f logs/exp4_*.log

ps aux | grep main_unified
```

## Configuration

- Seeds: 1–10 (10 runs per configuration)
- WandB: **offline** mode
- Noise schedule: constant
- Evaluation: every 20 episodes, snapshot every 100
- Target: x_target=0, x0_test=[0.1], T_sim=1000
- Group: `comparison_signature_mg_26_02`

## Output

Results are saved to:
```
outputs/comparison_signatures_MG_26_02/
├── signature_depth_2_seed_1_eval.pkl
├── signature_depth_3_seed_1_eval.pkl
├── no_signature_seed_1_eval.pkl
├── augmented_state_seed_1_eval.pkl
├── value_gradient_depth_2_seed_1_eval.pkl
└── ...
```
