# MG Comparison Experiments - Slurm via Hydra

Simple script Python qui utilise Hydra pour lancer tous les jobs en parallèle sur Slurm.

## 🚀 Utilisation

### Sur le cluster

```bash
# Transférer les fichiers
scp -r run_experiments/MG_comparison_tau_slurm user@cluster:~/path/to/Reinforcement_Learning/run_experiments/

# Se connecter
ssh user@cluster
cd ~/path/to/Reinforcement_Learning
source .venv/bin/activate

# Lancer TOUT (180 jobs)
python run_experiments/MG_comparison_tau_slurm/run_comparison.py

# Ou spécifier certains seeds
python run_experiments/MG_comparison_tau_slurm/run_comparison.py seed=1,2,3
```

Hydra se charge d'envoyer tous les jobs à Slurm automatiquement via Submitit.

## 📊 Sweeps

Le script lance :

| Paramètre | Valeurs | Jobs/combo |
|-----------|---------|-----------|
| **Agent** | CTAC_sig_MG_1D, CTAC_jax (3 configs) | 5 |
| **Tau** | 8, 30 | 2X |
| **Depth** | 2, 3, 4 | 3X (Sig + VG only) |
| **Seeds** | 1-10 | 10X |

**Total**: ~180 jobs

## 📁 Outputs

```
outputs/comparison_signatures_MG_26_02/
├── signature_tau_8_depth_2_seed_1.pkl
├── signature_tau_30_depth_4_seed_5.pkl
├── no_signature_tau_8_seed_3.pkl
└── ...
```

## 📊 Monitoring

```bash
# Voir vos jobs
squeue -u $USER

# Temps réel
watch -n 5 'squeue -u $USER'

# Annuler tous les jobs
scancel -u $USER
```
