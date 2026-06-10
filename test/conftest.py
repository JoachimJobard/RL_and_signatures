"""Shared pytest configuration.

Enable JAX float64 for the whole test session, matching ``main_unified.py`` so
tests exercise the same numerical precision as real runs.
"""

import jax

jax.config.update("jax_enable_x64", True)
