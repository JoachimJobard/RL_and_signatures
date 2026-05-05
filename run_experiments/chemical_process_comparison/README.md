# Chemical Process Comparison Experiments

This directory contains scripts to run parallel comparison experiments for the chemical reaction 4D environment.

## Experiments

1. **Signature-based CTAC** (`run_exp1_signatures.sh`)
   - Agent: `CTAC_sig_chemical`
   - Depth sweep: 2, 4
   - Uses path signatures with origin augmentation

2. **No Signatures (State-based)** (`run_exp2_no_signatures.sh`)
   - Agent: `CTAC_jax`
   - Standard actor-critic on raw state
   - No delay augmentation (`delayed_state=false`)

3. **Augmented State (Delayed)** (`run_exp3_augmented_state.sh`)
   - Agent: `CTAC_jax`
   - State concatenated with delayed state
   - Uses `delayed_state=true`

4. **Value Gradient** (`run_exp4_value_gradient.sh`)
   - Agent: `value_gradient`
   - Gradient-based policy (no explicit actor network)
   - Depth sweep: 2, 4

## Usage

### Run individual experiments

```bash
# Run each experiment separately
cd run_experiments/chemical_process_comparison

./run_exp1_signatures.sh
./run_exp2_no_signatures.sh
./run_exp3_augmented_state.sh
./run_exp4_value_gradient.sh
```

### Run all experiments in parallel

```bash
# Launch all 4 experiments as background jobs
./run_all_parallel.sh

# Launch specific experiments only (e.g., 1 and 3)
./run_all_parallel.sh 1 3
```

### Monitor progress

```bash
# View live logs for each experiment
tail -f logs/exp1_*.log
tail -f logs/exp2_*.log
tail -f logs/exp3_*.log
tail -f logs/exp4_*.log

# Check if processes are still running
ps aux | grep main_unified
```

### Stop experiments

```bash
# Find PIDs
ps aux | grep main_unified

# Kill specific experiment
kill <PID>

# Kill all Python processes (use with caution!)
pkill -f main_unified
```

## Configuration

All experiments use the same hyperparameters for fair comparison:
- Seeds: 42-47 (6 runs per configuration)
- Episodes: 1000
- Max time: 3.0
- Learning rates: 1e-3 (both actor and critic)
- Noise: σ=0.2, `noise_schedule=constant`
- Evaluation: every 20 episodes, with best checkpoint restoration
- Initial condition: `[0.15, -0.03, 0.1, 0.0]`

## Output

Results are saved to:
```
outputs/comparison_signatures_chemical_process_v_target_zero/
├── signature_depth_2_seed_42_eval.pkl
├── signature_depth_2_seed_42_multi.pkl
├── signature_depth_2_seed_42_training_metrics.pkl
├── base_no_delay_seed_42_eval.pkl
├── base_no_delay_seed_42_multi.pkl
├── base_no_delay_seed_42_training_metrics.pkl
└── ...
```

Each run produces:
- `*_eval.pkl`: Single trajectory evaluation data
- `*_multi.pkl`: Multiple trajectories evaluation data
- `*_training_metrics.pkl`: Training curves (cost, loss, gradients, weights)

## WandB Tracking

All experiments log to WandB under:
- Project: (configured in your config)
- Group: `comparison_signature_chemical_process`
- Run names: descriptive with seed information
