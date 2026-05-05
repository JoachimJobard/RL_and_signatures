# 2D System Comparison Experiments (delay_jax_unstable)

This directory contains scripts to run comparison experiments for the 2D delayed system with unstable environment.

## Experiments

1. **Signature-based CTAC** (`run_exp1_signatures.sh`)
   - Agent: `signatures`
   - Depth: 2
   - Time augmentation: enabled
   - Origin augmentation: enabled

2. **No Signatures (Standard CTAC)** (`run_exp2_no_signatures.sh`)
   - Agent: `CTAC_jax`
   - State-based policy without signatures
   - No delay augmentation (`delayed_state=false`)

3. **Augmented State (Delayed)** (`run_exp3_augmented_state.sh`)
   - Agent: `CTAC_jax`
   - State concatenated with delayed state history
   - Uses `delayed_state=true`

4. **Value Gradient** (`run_exp4_value_gradient.sh`)
   - Agent: `value_gradient`
   - Gradient-based policy with signatures
   - Depth: 2

## Usage

### Run individual experiments

```bash
# Run each experiment separately
cd run_experiments/comparison_signatures_2D_system_unstable

./run_exp1_signatures.sh
./run_exp2_no_signatures.sh
./run_exp3_augmented_state.sh
./run_exp4_value_gradient.sh
```

### Run all experiments sequentially

```bash
# Launch all 4 experiments one after another
./run_all_sequential.sh

# Launch specific experiments only (e.g., 1 and 3)
./run_all_sequential.sh 1 3
```

### Monitor progress

```bash
# View live logs
tail -f logs/exp1_*.log
tail -f logs/exp2_*.log
tail -f logs/exp3_*.log
tail -f logs/exp4_*.log
```

## Configuration

All experiments use:
- **Environment**: `delay_jax_unstable` (instead of `delay_jax`)
- **Seeds**: 1-10 (10 runs per configuration)
- **Max time**: 25s
- **Eval T_sim**: 20s
- **Preheat**: enabled
- **Burning steps**: 5
- **Eval interval**: 20 (10 for Value Gradient)
- **Snapshot interval**: 100
- **V_bad**: -0.1
- **fix_initial_state**: true

| Experiment | Agent | Episodes | Actor LR | Critic LR | Sigma | Noise Schedule | std_network |
|---|---|---|---|---|---|---|---|
| Exp1 (Signatures) | `signatures` | 1000 | 5e-2 | 1e-3 | 0.1 | constant | 0.2 |
| Exp2 (No Sig) | `CTAC_jax` | 1500 | 1e-2 | 1e-3 | 0.2 | decaying | 0.2 |
| Exp3 (Aug State) | `CTAC_jax` | 1500 | 1e-2 | 1e-3 | 0.2 | decaying | 0.1 |
| Exp4 (Value Grad) | `value_gradient` | 1300 | — | 1e-3 | 0.4 | linear_decay | 0.1 |

- **Output directory**: `outputs/comparison_signatures_2D_system_unstable/`
- **Weights & Biases group**: `comparison_signature_2D_unstable`

## Output

Results are saved to:
```
outputs/comparison_signatures_2D_system_unstable/
├── signature_depth_2_seed_1_eval.pkl
├── signature_depth_2_seed_1_multi.pkl
├── signature_depth_2_seed_1_training_metrics.pkl
├── ...
└── value_gradient_signature_depth_2_seed_10_training_metrics.pkl
```

## Notes

- The `delay_jax_unstable` environment provides a more challenging task than `delay_jax`
- Results can be compared directly with the `comparison_signatures_2D_system` variant
- Logs are saved in the `logs/` subdirectory for debugging
