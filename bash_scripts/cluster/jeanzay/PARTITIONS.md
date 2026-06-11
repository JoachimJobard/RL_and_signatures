# Jean Zay — partitions & QoS (live-extracted reference)

Extracted from the cluster on 2026-06-11 via `sinfo -a` + `sacctmgr show qos` +
`sacctmgr show assoc where user=$USER`. Authoritative over any second-hand table;
re-run the commands at the bottom to refresh. See also the routing rules in the
user's global `~/.claude/CLAUDE.md`.

## Partitions

| partition      | max wall | nodes | cores/node | mem/node | hardware                | billed? |
|----------------|----------|-------|------------|----------|-------------------------|---------|
| `cpu_p1`*      | 100 h    | 720   | 80         | 191 GB   | CPU (Cascade Lake)      | **yes** |
| `gpu_p13`      | 100 h    | 396   | 80         | 191 GB   | V100 quadri (16/32 GB)  | **yes** |
| `gpu_v116`     | 100 h    | 126   | 80         | 191 GB   | V100 16 GB (constrained)| **yes** |
| `gpu_v132`     | 100 h    | 270   | 80         | 191 GB   | V100 32 GB (constrained)| **yes** |
| `gpu_p2`       | 100 h    | 31    | 48         | 384 GB   | V100 octo               | **yes** |
| `gpu_p2l`      | 100 h    | 11    | 48         | 772 GB   | V100 octo (large RAM)   | **yes** |
| `gpu_p2s`      | 100 h    | 20    | 48         | 384 GB   | V100 octo (small)       | **yes** |
| `gpu_p5`       | 100 h    | 52    | 128        | 515 GB   | A100 octo 80 GB         | **yes** |
| `gpu_p6`       | 100 h    | 364   | 192        | 514 GB   | H100 quadri 80 GB       | **yes** |
| `prepost`      | 20 h     | 4     | 96         | **3 TB** | + 1 V100                | **NO**  |
| `visu`         | 4 h      | 5     | 40         | 192 GB   | Quadro P6000            | **NO**  |
| `archive`      | 20 h     | 6     | —          | —        | data movement (rsync)   | **NO**  |
| `compil`       | 20 h     | 10    | —          | —        | compilation / pip       | **NO**  |
| `compil_h100`  | 20 h     | 1     | —          | —        | H100-matched compile    | **NO**  |

`100 h` = `4-04:00:00`. The 5 **non-billed** partitions (`prepost`, `visu`, `archive`,
`compil`, `compil_h100`) deduct **no** allocation hours and are reachable without
CPU/GPU hours — use them for anything that does not strictly need a billed node.

## QoS (the `-dev` / `-t3` / `-t4` ladder)

| qos            | max wall | max submitted jobs / user | priority |
|----------------|----------|---------------------------|----------|
| `qos_cpu-dev`  | 2 h      | **10**                    | 80 (high)|
| `qos_cpu-t3`   | 20 h     | 10000                     | 50       |
| `qos_cpu-t4`   | 100 h    | 1000                      | 45       |
| `qos_gpu-dev`  | 2 h      | **10**                    | 80 (high)|
| `qos_gpu-t3`   | 20 h     | 10000                     | 50       |
| `qos_gpu-t4`   | 100 h    | 1000                      | 45       |

A100/H100 have their own QoS (`qos_gpu_a100-{dev,t3}`, `qos_gpu_h100-{dev,t3,t4}`).
**The `-dev` QoS schedules fast (priority 80) but caps a user at 10 submitted jobs** —
a job-array launch with >10 tasks (e.g. 7 variants × 2 seeds) gets the *excess
rejected* (`AssocMaxSubmitJobLimit`). Split across `-dev` submissions, or use a
non-billed partition, or `qos_cpu-t3` (no practical submit cap).

## My accounts (associations)

| resource | accounts (`<proj>@<hw>`) | QoS available |
|----------|--------------------------|----------------|
| CPU      | `akz@cpu`, `oym@cpu`     | `qos_cpu-{dev,t3,t4}` |
| V100     | `akz@v100`, `oym@v100`   | `qos_gpu-{dev,t3,t4}` |
| A100     | `akz@a100`, `oym@a100`   | `qos_gpu_a100-{dev,t3}` |
| H100     | `akz@h100`               | `qos_gpu_h100-{dev,t3,t4}` |

The non-billed partitions are entered with `--partition=<name> --account=akz@cpu` and
**no `--qos`** (e.g. the launcher's finalize uses `--partition=prepost --account=akz@cpu`).

## Routing for THIS repo

- **Real training (H1/H2 grids, MG, linear):** `--mode cpu` ⇒ `cpu_p1` / `qos_cpu-t3`
  (billed). JAX runs on CPU here (small linear-in-Φ critics).
- **Smoke / quick checks:** `--qos qos_cpu-dev` (2 h, fast) — but keep the launch ≤10
  tasks total (the dev cap).
- **Light diagnostics that are training but tiny** (e.g. the markovian convergence /
  trick ablation, a few hundred episodes on a 2-D plant): prefer **`prepost`** — non-
  billed, 96 cores, 3 TB, 20 h, far less contended than `-dev`, and **no 10-job cap**.
  The launcher does not expose `--partition` for TRAIN yet (it routes TRAIN to
  `cpu_p1`/`gpu_p13` only; see `experiment_array_launcher.sh` `--mode`), so to use
  `prepost`/`visu` for training, submit directly (`sbatch --partition=prepost
  --account=akz@cpu ...`) or extend the launcher with a `--partition` override.
- **Replot / aggregation / finalize:** `prepost` (non-billed), as the launcher's
  `--finalize` already does.

## Refresh commands

```bash
sinfo -a -h -o "%P %.12l %.5D %.4c %.7m %.22f"          # partitions
sacctmgr -nP show qos format=Name,MaxWallDurationPerJob,MaxSubmitJobsPerUser,Priority
sacctmgr -nP show assoc where user=$USER format=account,partition,qos
```
