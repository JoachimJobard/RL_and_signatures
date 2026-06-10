# Jean Zay (IDRIS) — launchers for RL_and_signatures

Standalone `sbatch` launchers for running `main_unified.py` on Jean Zay, with
job-array parallelisation for ablations/sweeps. They replace the Inria-`tau`
Hydra submitit launcher (`conf/launcher/slurm.yaml`).

## Layout

```
bash_scripts/cluster/jeanzay/
  _environment.sh                       # shared: locate project, activate .venv, JAX diagnostics
  python/
    python_script_launcher.sh           # single run, CPU  (cpu_p1, akz@cpu)
    python_script_launcher_gpu.sh       # single run, GPU  (gpu_p13/V100, akz@v100)
    experiment_array_launcher.sh        # job array over a variants file (+ optional finalize)
    run_python_script.slurm             # single-run worker
    job_array_batch_xp.slurm            # array worker (one task per variant)
    run_interactive_job.sh              # interactive compute shell (CPU prepost, or --gpu)
    variants_example.txt                # example variants file (depth sweep + baselines)
```

## Partition / QoS routing

Choices follow the routing table in `~/.claude/CLAUDE.md`; each launcher's header
documents the partition + QoS + why.

| Use | Partition | Account | QoS | Billed |
|-----|-----------|---------|-----|--------|
| Array / single, **CPU** (default for these small RL models) | `cpu_p1` | `akz@cpu` | `qos_cpu-t3` (≤20 h) | yes |
| Array / single, **GPU** (only if needed) | `gpu_p13` (V100) | `akz@v100` | `qos_gpu-t3` (≤20 h) | yes |
| **Finalize** / replot / aggregation | `prepost` | `akz@cpu` | — | **no** |
| Interactive CPU check | `prepost` | `akz@cpu` | — | **no** |
| Interactive GPU check | `gpu_p13` | `akz@v100` | `qos_gpu-dev` (≤2 h) | yes |

Smoke tests: add `--qos qos_cpu-dev` / `qos_gpu-dev` (≤2 h, schedule fast) and put
`debug=true` in the overrides so run directories are flagged `_debug_`.

## One-time setup (on Jean Zay)

```bash
# 1. Clone into $WORK (the launchers expect $WORK/git_repositories/RL_and_signatures).
mkdir -p "$WORK"/git_repositories && cd "$WORK"/git_repositories
git clone git@github.com:JoachimJobard/RL_and_signatures.git
cd RL_and_signatures

# 2. Install uv (login node has outbound HTTPS via the IDRIS proxy).
#    export HTTPS_PROXY=http://prodprox.idris.fr:3128 if curl cannot reach the net.
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create the venv. Default resolution gives a CPU JAX, which is what the CPU
#    launchers use. Build/installs that need network should run on the 'compil'
#    partition (non-billed, has PyPI egress) rather than the login node:
sbatch --partition=compil --account=akz@cpu --time=00:30:00 \
       --wrap="cd $PWD && ~/.local/bin/uv sync"
#    For the GPU path, install a CUDA-enabled JAX wheel matching Jean Zay's CUDA
#    module into the same venv before using python_script_launcher_gpu.sh.

# 4. Smoke-test on a compute node (never on the login node):
bash bash_scripts/cluster/jeanzay/python/run_interactive_job.sh
python -c "import jax; print(jax.__version__, jax.devices())"
```

## Running

Single run (CPU):
```bash
bash bash_scripts/cluster/jeanzay/python/python_script_launcher.sh \
  --args "agent=signatures env=MG_1D agent.training.n_episodes=2000 wandb.mode=offline"
```

Ablation / sweep as a job array (one task per variant line):
```bash
bash bash_scripts/cluster/jeanzay/python/experiment_array_launcher.sh \
  --variants-file bash_scripts/cluster/jeanzay/python/variants_example.txt \
  --experiment-group mg_depth_ablation \
  --common "agent.training.n_episodes=2000" \
  --seed 0
```
All tasks write under `data/main_unified/mg_depth_ablation/`, one folder per
`run_tag`; SLURM logs go to `data/main_unified/mg_depth_ablation/slurm/`.

Multi-seed: submit the same array once per seed (each seed is an explicit axis):
```bash
for s in 0 1 2; do
  bash .../experiment_array_launcher.sh --variants-file ... \
       --experiment-group mg_depth_ablation --seed "$s"
done
```

Optional aggregation after all tasks succeed (runs on non-billed `prepost`); the
`{GROUP_DIR}` placeholder is substituted with the experiment-group directory:
```bash
  --finalize "python <aggregation_script>.py {GROUP_DIR}"
```
(An aggregation/comparison script will be added with the experiment set; the
`main_unified.py replot=<run_dir>` mode already rebuilds per-run figures.)

## After the run

`wandb.mode=offline` writes `wandb/offline-run-*`; sync from the **login** node
(which has outbound network) with `wandb sync wandb/offline-run-*`. Pull results
to the laptop with, e.g.:
```bash
rsync -avzP jeanzay-any:'$WORK/git_repositories/RL_and_signatures/data/main_unified/mg_depth_ablation/' \
      data/main_unified/mg_depth_ablation/
```
(SSH host alias `jeanzay-any` per the laptop's `~/.ssh/config`.)
