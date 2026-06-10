"""Polynomial representations: Markovian (current state) and raw-history.

These are the non-signature representations of the H2 comparison. The readout is
linear, so degree-m polynomial monomials of the input span the degree-m functionals
of (the current state / the discretised history). Degree 2 of the raw history spans
the quadratic functionals of the path — exactly the delayed-LQR value class
(Kolmanovskii eq. 2.4) and the fair match to a depth-2 signature. Sweeping the
degree m (raw history) against the depth m (signature) is the matched-capacity
fairness sweep.
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np


def _monomial_index_tuples(input_dim: int, degree: int) -> list[tuple[int, ...]]:
    """All monomial index multisets of total degree 1..``degree`` over ``input_dim``
    variables (e.g. ``(i,)`` -> x_i, ``(i, i)`` -> x_i^2, ``(i, j)`` -> x_i x_j)."""
    tuples: list[tuple[int, ...]] = []
    for k in range(1, degree + 1):
        tuples.extend(itertools.combinations_with_replacement(range(input_dim), k))
    return tuples


class _PolynomialFeatureMap:
    """Polynomial monomials (degrees 1..m) of a flattened input vector."""

    def __init__(self, input_dim: int, degree: int) -> None:
        if degree < 1:
            raise ValueError(f"degree must be >= 1 (got {degree})")
        self.input_dim = input_dim
        self.degree = degree
        self._index_tuples = _monomial_index_tuples(input_dim, degree)
        self.feature_dim = len(self._index_tuples)
        # Vectorised monomial evaluation: prepend a 1.0 to the input (index 0), shift
        # the real variable indices by +1, and right-pad each monomial to `degree`
        # with 0 (which multiplies by the prepended 1.0). Then a single gather + a
        # single product over the padded matrix computes ALL monomials at once — a
        # small, fast XLA graph under autodiff (a per-monomial list of jnp.prod
        # explodes the graph and is intractable to differentiate at high degree).
        padded = np.zeros((self.feature_dim, degree), dtype=int)
        for i, idx in enumerate(self._index_tuples):
            shifted = [j + 1 for j in idx]
            padded[i, : len(shifted)] = shifted
        self._padded_index = jnp.asarray(padded)

    def _features_from_vector(self, v: jnp.ndarray) -> jnp.ndarray:
        v_aug = jnp.concatenate([jnp.ones((1,), dtype=v.dtype), v])  # index 0 -> 1.0
        gathered = v_aug[self._padded_index]  # (feature_dim, degree)
        return jnp.prod(gathered, axis=1)


class MarkovianRepresentation(_PolynomialFeatureMap):
    """Phi(x_t) = polynomial monomials (degrees 1..m) of the CURRENT state x(t).

    The history is ignored: this is the Markovian baseline for H1. Degree 2 gives the
    quadratic LQR feature class for the non-delayed cell.
    """

    def __init__(self, n_state: int, degree: int = 2) -> None:
        self.n_state = n_state
        super().__init__(input_dim=n_state, degree=degree)

    def feature_fn(self, window: jnp.ndarray) -> jnp.ndarray:
        return self._features_from_vector(window[-1])


class RawHistoryRepresentation(_PolynomialFeatureMap):
    """Phi(x_t) = polynomial monomials (degrees 1..m) of the flattened discretised
    history window (x(t), x(t-dt), ..., x(t-(L-1)dt)).

    Degree 1 is the flat window (linear readout = linear functional of the history,
    the delayed-LQR control class); degree 2 adds pairwise products (the delayed-LQR
    value class). This is the non-signature history representation for H2.
    """

    def __init__(self, window_length: int, n_state: int, degree: int = 2) -> None:
        self.window_length = window_length
        self.n_state = n_state
        super().__init__(input_dim=window_length * n_state, degree=degree)

    def feature_fn(self, window: jnp.ndarray) -> jnp.ndarray:
        return self._features_from_vector(window.reshape(-1))
