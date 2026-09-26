"""Shared grids and profiles for the harmonic tests.

The same log-spaced grid and spline fit were rebuilt by hand in most tests
here. Naming them once makes the differences between tests -- which are the
interesting part -- visible instead of buried in identical setup.
"""

from collections.abc import Callable
from jaxtyping import Array, Float

import pytest

import quaxed.numpy as jnp

from galax.potential._src.harmonic.spline import fit_log_spline

R_MIN, R_MAX, N_R = 0.05, 20.0, 128
"""A deliberately small ``r_max`` so 2x, 5x and 10x are all real space."""


@pytest.fixture(scope="session")
def log_r() -> Float[Array, "n_r"]:
    """Log-spaced knots over ``[R_MIN, R_MAX]``."""
    return jnp.log(jnp.geomspace(R_MIN, R_MAX, N_R))


@pytest.fixture(scope="session")
def r(log_r: Float[Array, "n_r"]) -> Float[Array, "n_r"]:
    """Return the same knots as radii."""
    return jnp.exp(log_r)


@pytest.fixture(scope="session")
def splined() -> Callable[..., tuple[Array, Array, Array]]:
    """Build ``(log_r, values, derivs)`` for a radial profile.

    Takes ``phi_of_r`` and optional grid bounds, so a test that needs a
    different bracket says so in one line instead of rebuilding the grid.
    """

    def build(phi_of_r, *, r_min=R_MIN, r_max=R_MAX, n_r=N_R, dtype=None):
        lr = jnp.log(jnp.geomspace(r_min, r_max, n_r))
        if dtype is not None:
            lr = lr.astype(dtype)
        vals = jnp.atleast_2d(phi_of_r(jnp.exp(lr)).T).T
        return lr, vals, fit_log_spline(lr, vals)

    return build


@pytest.fixture(scope="session")
def hernquist() -> Callable[[Float[Array, "n_r"]], Float[Array, "n_r"]]:
    """Hernquist monopole with ``M = a = 1``: ``Phi = -1/(1+r)``."""
    return lambda rr: -1.0 / (1.0 + rr)
