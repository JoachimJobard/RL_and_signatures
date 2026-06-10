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
        # Precompute a static integer index array per monomial for a jittable gather.
        self._index_arrays = [np.asarray(idx, dtype=int) for idx in self._index_tuples]
        self.feature_dim = len(self._index_tuples)

    def _features_from_vector(self, v: jnp.ndarray) -> jnp.ndarray:
        # Each monomial is the product of the gathered (with repetition) entries.
        terms = [jnp.prod(v[idx]) for idx in self._index_arrays]
        return jnp.stack(terms)


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
