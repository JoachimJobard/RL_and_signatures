"""Mechanism measurement: on-manifold effective rank of the raw-history feature Gram
versus the delay tau (the Hopfield nonlinear cell).

The H2 controls show that raw-history's degree-2 feature dimension grows as O(tau^2)
(window ~ tau/dt) while the controlled trajectory occupies a low-dimensional attractor.
This script MEASURES that gap: for each delay tau it builds the on-sheet (optimal-
trajectory) windows exactly as the benchmark does (run/study/h1h2_evaluation.py ->
on_sheet_windows, the Pontryagin optimum), forms the centred degree-2 raw-history
feature covariance C = E[(Phi-mean)(Phi-mean)^T], and reports

  * the nominal feature dimension dim(Phi)  (grows as O(tau^2)),
  * the effective rank erank(C) = (tr C)^2 / ||C||_F^2  (participation ratio),

computed via the small-side Gram K = Phi_c Phi_c^T (n_win x n_win), since
(tr C)^2/||C||_F^2 = (tr K)^2/||K||_F^2 and n_win << dim. The signature depth-2 dimension
(fixed in tau) is reported alongside for contrast.

This is the empirical face of the Veronese / effective-rank bound of the note
(latex_documents/notes/2026_06_17_signature_density_vs_conditioning): the covariance rank
of degree-2 features of a window affine in a low-dimensional initial condition is bounded
by binom(k+2,2), independent of the ambient dimension dim(Phi).

Usage:
    H1H2_N_IC=8 uv run python run/study/h1h2_effective_rank_vs_delay.py --taus 1 2 3 4 5 6
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def effective_rank_centered(Phi):
    """erank of the centred covariance via the n_win x n_win sample Gram (dim-agnostic)."""
    Phi = np.asarray(Phi, dtype=np.float64)
    Phic = Phi - Phi.mean(axis=0, keepdims=True)
    K = Phic @ Phic.T                                 # (n_win, n_win); tr/Frob match the dim-side cov
    tr = float(np.trace(K))
    fro2 = float(np.sum(K * K))
    return tr * tr / (fro2 + 1e-300)


def measure_one(tau, seed=0):
    import jax, jax.numpy as jnp
    os.environ["HOPFIELD_TAU"] = str(tau)
    from h1h2_evaluation import make_cell, generate_data, REPS
    from src.representations.factory import make_representation
    cell = make_cell("hopfield_nonlinear")
    # on-sheet (optimal-trajectory) windows only: the on-manifold data the benchmark fits.
    W, _ = generate_data(cell, seed, off_sheet=False)
    out = dict(tau=float(tau), window_length=int(cell["window_length"]),
               n_state=int(cell["n"]), n_windows=int(len(W)))
    for name in ("raw_history", "signature"):
        cfg = REPS[name]
        kw = {k: v for k, v in cfg.items() if k != "kind"}
        rep = make_representation(cfg["kind"], window_length=cell["window_length"],
                                  n_state=cell["n"], **kw)
        feat = jax.jit(rep.feature_fn)
        Phi = np.stack([np.asarray(feat(jnp.asarray(w))) for w in W])
        out[f"{name}_dim"] = int(Phi.shape[1])
        out[f"{name}_effrank"] = float(effective_rank_centered(Phi))
    print(f"  tau={tau:>3} win={out['window_length']:>3} | "
          f"raw_history dim={out['raw_history_dim']:>5} erank={out['raw_history_effrank']:>5.2f} | "
          f"signature dim={out['signature_dim']:>3} erank={out['signature_effrank']:>5.2f} | "
          f"n_win={out['n_windows']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", type=float, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    n_ic = os.environ.get("H1H2_N_IC", "(default)")
    print(f"effective-rank vs delay (Hopfield nonlinear), H1H2_N_IC={n_ic}, seed={args.seed}")
    rows = []
    for tau in args.taus:
        rows.append(measure_one(tau, seed=args.seed))

    from src.utils.run_context import script_data_dir
    out = Path(args.out_dir) if args.out_dir else (
        script_data_dir(__file__) / f"{datetime.now():%Y%m%d_%H%M%S}_effrank_vs_tau")
    out.mkdir(parents=True, exist_ok=True)
    (out / "effrank_vs_tau.json").write_text(json.dumps(dict(seed=args.seed, n_ic=str(n_ic), rows=rows), indent=2))
    print(f"\nwrote {out / 'effrank_vs_tau.json'}")


if __name__ == "__main__":
    main()
