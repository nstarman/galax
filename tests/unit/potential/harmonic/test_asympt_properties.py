import numpy as np
import pytest
from hypothesis import given, strategies as st

import quaxed.numpy as jnp

from galax.potential._src.harmonic.asympt import (
    _pow_diff,
    _series_tol,
    eval_log_spline_asympt,
)
from galax.potential._src.harmonic.spline import fit_log_spline

# Bounded so `exp(e * L)` stays inside float32 range: _pow_diff itself does no
# clamping -- that is `eval_log_spline_asympt`'s job.
expo = st.floats(min_value=-8.0, max_value=8.0, allow_nan=False, allow_infinity=False)
lnx = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)


@given(a=expo, b=expo, L=lnx)
def test_pow_diff_is_symmetric_under_swapping_its_exponents(a, b, L):
    """(x^a - x^b)/(a - b) is unchanged by swapping a and b: both signs flip."""
    lhs = float(_pow_diff(jnp.asarray(a), jnp.asarray(b), jnp.asarray(L)))
    rhs = float(_pow_diff(jnp.asarray(b), jnp.asarray(a), jnp.asarray(L)))
    assert np.isfinite(lhs)
    assert lhs == pytest.approx(rhs, rel=1e-9, abs=1e-12)


@given(b=expo, L=lnx)
def test_pow_diff_is_continuous_across_the_series_seam(b, L):
    """The two branches must agree where they meet.

    That is the whole job of `_series_tol`: pick the crossover so neither
    branch is inaccurate there. A seam that is misplaced shows up as a step.
    """
    seam = _series_tol(jnp.asarray(L))
    if abs(L) < 1e-3:  # z = (a-b) L cannot reach the seam
        return
    d = seam / abs(L)  # |z| == seam exactly at the crossover
    below = float(_pow_diff(jnp.asarray(b + 0.99 * d), jnp.asarray(b), jnp.asarray(L)))
    above = float(_pow_diff(jnp.asarray(b + 1.01 * d), jnp.asarray(b), jnp.asarray(L)))
    assert np.isfinite(below)
    assert np.isfinite(above)
    assert below == pytest.approx(above, rel=2e-2, abs=1e-9)


@given(
    v=st.floats(-13.0, 13.0, allow_nan=False, allow_infinity=False),
    s=st.floats(-8.0, 8.0, allow_nan=False, allow_infinity=False),
    B=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
    lr=st.floats(-45.0, 45.0, allow_nan=False, allow_infinity=False),
)
def test_the_tail_is_finite_for_any_fitted_coefficients(v, s, B, lr):
    """No (v, s, B) and no query radius may produce a nan or an inf.

    The clamp exists to guarantee exactly this, and it is the guarantee that
    was broken in float32: the bound was sized by |v| rather than by the
    exponents that can grow, so it truncated reachable radii. Ranges cover
    l_max = 12 (|v| <= l+1 = 13) and the `_S_INNER`/`_S_OUTER` brackets.
    """
    log_r = jnp.log(jnp.geomspace(1e-2, 1e2, 16))
    values = jnp.ones((16, 1))
    derivs = fit_log_spline(log_r, values)
    coefs = jnp.asarray([[[v], [s], [B]], [[-abs(v) - 1.0], [-abs(s)], [B]]])

    got = eval_log_spline_asympt(log_r, values, derivs, coefs, jnp.asarray([lr]))

    assert jnp.all(jnp.isfinite(got)), f"v={v} s={s} B={B} log_rq={lr} -> {got}"
