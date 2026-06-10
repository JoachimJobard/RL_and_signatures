"""Compare path-signature computation in float32 versus float64.

Side-quest diagnostic for code_review finding F-E1: the signature buffers used to
hard-cast to float32 even though ``main_unified.py`` enables JAX x64. This script
quantifies how much precision that lost, by computing the signature of the same
windowed state history at both precisions (via the ``dtype`` parameter added to
``SlidingSignatureJAX``) and reporting their relative discrepancy as a function of
the truncation depth and the window length.

The discrepancy is reported as the relative Euclidean error
    e = || S_{f32} - S_{f64} ||_2 / || S_{f64} ||_2 ,
treating the float64 signature as the reference. Because a depth-``m`` signature of
a ``d``-channel path is a sum of ``d^m`` products of windowed increments, the error
is expected to grow with both ``m`` and the window length.

Run (inside the project environment):
    python run/diagnostics/compare_signature_float32_vs_float64.py
    python run/diagnostics/compare_signature_float32_vs_float64.py --debug   # quick smoke run

Outputs (figure + machine-readable results + a self-contained log) are written to
    data/compare_signature_float32_vs_float64/<timestamp>[ _debug ]/
so the figure can be regenerated from the saved results without recomputation.
"""

import argparse
import datetime as _datetime
import json
import platform
import sys
from pathlib import Path

import numpy as np

# JAX x64 must be enabled so that float64 is genuinely 64-bit (matches main_unified.py).
import jax

jax.config.update("jax_enable_x64", True)

from src.utils.dynamic_signature import SlidingSignatureJAX
from src.utils.run_context import resolve_run_dir


def make_representative_path(window_length: int, n_state: int, rng: np.random.Generator) -> np.ndarray:
    """Build a representative windowed state history of shape (window_length, n_state).

    A smooth (low-frequency sinusoidal) component plus a small random walk mimics the
    delayed state trajectories the signature consumes, in float64.
    """
    t = np.linspace(0.0, 1.0, window_length)
    smooth = np.stack(
        [np.sin(2.0 * np.pi * (k + 1) * t + rng.uniform(0, np.pi)) for k in range(n_state)],
        axis=1,
    )
    random_walk = np.cumsum(rng.standard_normal((window_length, n_state)) * 0.05, axis=0)
    return (smooth + random_walk).astype(np.float64)


def signature_at_dtype(path: np.ndarray, depth: int, dtype) -> np.ndarray:
    """Compute the signature of ``path`` (window, d) at the requested buffer dtype."""
    window_length, n_state = path.shape
    sig = SlidingSignatureJAX(
        depth=depth,
        window_size=window_length - 1,  # buffer capacity is window_size + 1
        d=n_state,
        time_augmentation=True,
        origin_augmentation=True,
        bias=False,
        use_jax_buffer=False,
        dtype=dtype,
    )
    sig.reset(prefill_zeros=False)
    for row in path:
        sig.append(row)
    return np.asarray(sig.current_signature, dtype=np.float64)


def relative_error(reference: np.ndarray, other: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference))
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm(other - reference) / denom)


# Smoke-test guard: a real run sweeps several depths/windows; refuse a degenerate run
# unless --debug is passed (mirrors the project's smoke-test guard convention).
SMOKE_TEST_MIN_DEPTHS = 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Master seed for the synthetic paths.")
    parser.add_argument("--n-state", type=int, default=1, help="State dimension N of the path.")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Signature truncation depths to sweep.")
    parser.add_argument("--window-lengths", type=int, nargs="+", default=[10, 20, 40, 80],
                        help="Window lengths (number of samples) to sweep.")
    parser.add_argument("--n-repeats", type=int, default=16,
                        help="Number of random paths averaged per (depth, window) point.")
    parser.add_argument("--debug", action="store_true",
                        help="Flag this as an exploratory run (prepends _debug_ to the output dir).")
    args = parser.parse_args()

    if not args.debug and len(args.depths) < SMOKE_TEST_MIN_DEPTHS:
        parser.error(
            f"Refusing an under-sized real run with {len(args.depths)} depth(s) "
            f"(< {SMOKE_TEST_MIN_DEPTHS}). Pass --debug to flag it as exploratory."
        )

    timestamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = resolve_run_dir(__file__, "f32_vs_f64", seed=args.seed,
                              debug=args.debug, timestamp=timestamp)

    # Self-contained run log: command line, environment, and resolved parameters.
    log_lines = [
        "=== compare_signature_float32_vs_float64 ===",
        f"command_line     : {' '.join(sys.argv)}",
        f"timestamp        : {timestamp}",
        f"python           : {platform.python_version()}",
        f"jax              : {jax.__version__}",
        f"numpy            : {np.__version__}",
        f"jax_x64_enabled  : {jax.config.jax_enable_x64}",
        f"seed             : {args.seed}",
        f"n_state          : {args.n_state}",
        f"depths           : {args.depths}",
        f"window_lengths   : {args.window_lengths}",
        f"n_repeats        : {args.n_repeats}",
        f"output_dir       : {out_dir}",
    ]
    print("\n".join(log_lines))

    rng = np.random.default_rng(args.seed)
    results = []  # one record per (depth, window_length)
    for depth in args.depths:
        for window_length in args.window_lengths:
            errors = []
            for _ in range(args.n_repeats):
                path = make_representative_path(window_length, args.n_state, rng)
                sig64 = signature_at_dtype(path, depth, np.float64)
                sig32 = signature_at_dtype(path, depth, np.float32)
                errors.append(relative_error(sig64, sig32))
            errors = np.asarray(errors, dtype=np.float64)
            record = {
                "depth": depth,
                "window_length": window_length,
                "relative_error_mean": float(np.nanmean(errors)),
                "relative_error_std": float(np.nanstd(errors)),
                "relative_error_max": float(np.nanmax(errors)),
            }
            results.append(record)
            print(f"  depth={depth} window={window_length:>3} "
                  f"rel_err mean={record['relative_error_mean']:.3e} "
                  f"max={record['relative_error_max']:.3e}")

    # Persist machine-readable results so the figure regenerates without recomputation.
    results_payload = {"metadata": {k: v for k, v in vars(args).items()}, "results": results}
    (out_dir / "results.json").write_text(json.dumps(results_payload, indent=2))
    (out_dir / "run.log").write_text("\n".join(log_lines) + "\n")

    _plot(results, args, out_dir)
    print(f"\nDone. Results and figure in: {out_dir}")


def _plot(results, args, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"text.usetex": False, "font.family": "serif", "font.size": 11})

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    cmap = plt.get_cmap("viridis")
    depths = sorted({r["depth"] for r in results})
    for i, depth in enumerate(depths):
        rows = sorted((r for r in results if r["depth"] == depth), key=lambda r: r["window_length"])
        windows = [r["window_length"] for r in rows]
        means = [r["relative_error_mean"] for r in rows]
        color = cmap(i / max(1, len(depths) - 1))
        # Trained/measured quantities use a solid stroke; the depth sweep is encoded in colour.
        ax.plot(windows, means, "-", marker="o", color=color, label=rf"depth $m={depth}$")

    ax.set_yscale("log")
    ax.set_xlabel(r"window length (number of samples) $L$")
    ax.set_ylabel(r"relative error $\;\|S_{f32}-S_{f64}\|_2 / \|S_{f64}\|_2$")
    ax.set_title("Path-signature precision: float32 versus float64")
    ax.grid(True, which="both", alpha=0.3)

    # Explanatory text box (definition of the plotted quantity), below the axes.
    textbox = (
        r"$S$: depth-$m$ truncated path signature of a $d$-channel windowed history "
        r"($d = N + 1 + N$ with time + origin augmentation). "
        r"$S_{f64}$ is the float64 reference; larger $m$ and $L$ accumulate more "
        r"float32 rounding in the $d^{m}$ iterated-integral products."
    )
    fig.text(0.5, -0.02, textbox, ha="center", va="top", wrap=True, fontsize=8)

    # External legend, below the axes, arranged in columns.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=min(4, len(depths)), fontsize=9, frameon=True)
    fig.tight_layout(rect=[0, 0.20, 1, 1])
    fig.savefig(out_dir / "signature_float32_vs_float64.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "signature_float32_vs_float64.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
