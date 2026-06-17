"""Why does the platoon LSTD blow up while MG works? Compare the feature-data statistics.

For (cell, rep) over a WORKING cell (mg_limit_cycle) and the FAILING one (platoon), collect the LSTD
transitions under the oracle policy (+ exploration) and report: feature dim D, n_transitions, the
EFFECTIVE RANK of the centred feature Gram (participation ratio (sum lambda)^2 / sum lambda^2), its
condition number, the LSTD matrix M = Phi_c^T Psi condition number, and the variance kept by the top
modes. The blow-up is the ker-G pathology: thin on-policy data -> low effective rank -> M ill-posed.

Usage:
    uv run python run/study/mwe_lspi_data_stats.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h1h2_evaluation import make_cell, REPS                                       # noqa: E402
from mwe_lspi_test import make_feat, reference_control, collect_dataset           # noqa: E402

CELLS = [("mg_limit_cycle", "WORKS"), ("platoon", "FAILS")]
REPSEL = ["raw_history", "signature"]
TAU = 2.0


def stats(cell, feat, P, PN, dt):
    n, D = P.shape
    Pc = P - P.mean(0)
    C = (Pc.T @ Pc) / n
    lam = np.linalg.eigvalsh((C + C.T) / 2.0)
    lam = np.clip(lam[::-1], 0, None)                       # descending, nonneg
    eff = float(lam.sum() ** 2 / (np.sum(lam ** 2) + 1e-30))   # participation ratio
    cond_C = float(lam[0] / (lam[lam > 1e-14][-1] if np.any(lam > 1e-14) else lam[0]))
    psi = (PN - P) / dt - Pc / TAU
    M = Pc.T @ psi
    sv = np.linalg.svd(M, compute_uv=False)
    cond_M = float(sv[0] / (sv[sv > 1e-12][-1] if np.any(sv > 1e-12) else sv[0]))
    cum = np.cumsum(lam) / (lam.sum() + 1e-30)
    var10 = float(cum[min(9, D - 1)])                      # variance in top-10 modes
    return dict(D=D, n=n, nD=n / D, eff=eff, effD=eff / D, condC=cond_C, condM=cond_M, var10=var10)


def main():
    print(f"\n{'cell':>16}{'rep':>12}{'D':>6}{'n':>7}{'n/D':>6}{'effRank':>9}"
          f"{'eff/D':>7}{'cond(C)':>10}{'cond(M)':>10}{'var@10':>8}")
    for cellname, _ in CELLS:
        cell = make_cell(cellname)
        u_star = reference_control(cell)
        tf, dt = cell["tf"], cell["dt"]
        t_collect = tf if cell.get("nonlinear") else tf * 0.3
        for rep in REPSEL:
            feat, _ = make_feat(cell, REPS[rep])
            P, PN, _ = collect_dataset(cell, feat, u_star, seed=0, t_collect=t_collect, sigma=0.3)
            s = stats(cell, feat, np.asarray(P), np.asarray(PN), dt)
            print(f"{cellname:>16}{rep:>12}{s['D']:>6}{s['n']:>7}{s['nD']:>6.1f}{s['eff']:>9.1f}"
                  f"{s['effD']:>7.3f}{s['condC']:>10.1e}{s['condM']:>10.1e}{s['var10']:>8.3f}")


if __name__ == "__main__":
    main()
