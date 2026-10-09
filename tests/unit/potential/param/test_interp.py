"""Test :mod:`galax.potential._src.params.interp`."""

import jax
import pytest

import quaxed.numpy as jnp
import unxt as u

from galax.potential.params import time_interpolated_parameter

TS = u.Q(jnp.asarray([0.0, 1.0, 2.0, 3.0]), "Gyr")


def _linear(slope: float = 2.0):
    """Build a parameter exactly linear in ``t``, so interpolation is exact."""
    values = u.Q(1.0 + slope * u.ustrip(u.unit("Gyr"), TS), "Msun")
    return time_interpolated_parameter(TS, values), slope


def test_returns_the_tabulated_value_at_a_knot() -> None:
    """On a knot the interpolant must return that knot's value exactly."""
    p, slope = _linear()
    for i, tv in enumerate((0.0, 1.0, 2.0, 3.0)):
        got = u.ustrip(u.unit("Msun"), p(u.Q(tv, "Gyr")))
        assert float(got) == pytest.approx(1.0 + slope * tv, rel=1e-12), i


def test_interpolates_between_knots() -> None:
    """A cubic through linear data must reproduce the line, not the nearest knot.

    Chosen so that "return the nearest knot" and "return the mean of the
    bracketing knots" both give visibly wrong answers.
    """
    p, slope = _linear()
    for tv in (0.25, 1.4, 2.9):
        got = float(u.ustrip(u.unit("Msun"), p(u.Q(tv, "Gyr"))))
        assert got == pytest.approx(1.0 + slope * tv, rel=1e-10), tv


def test_clamps_outside_the_grid() -> None:
    """Outside the tabulated range the value saturates at the boundary.

    Not extrapolated: continuing the edge cubic past the grid invents
    structure the tabulation knows nothing about. A linear ramp makes the
    distinction unmistakable -- extrapolation would keep climbing.
    """
    p, slope = _linear()
    lo = float(u.ustrip(u.unit("Msun"), p(u.Q(-50.0, "Gyr"))))
    hi = float(u.ustrip(u.unit("Msun"), p(u.Q(+50.0, "Gyr"))))
    assert lo == pytest.approx(1.0, rel=1e-12)
    assert hi == pytest.approx(1.0 + slope * 3.0, rel=1e-12)


def test_interpolates_array_valued_parameters() -> None:
    """Time is the leading axis; every trailing axis is carried along."""
    values = u.Q(jnp.stack([jnp.full((2, 3), float(i)) for i in range(4)]), "Msun")
    p = time_interpolated_parameter(TS, values)
    got = u.ustrip(u.unit("Msun"), p(u.Q(1.5, "Gyr")))
    assert got.shape == (2, 3)
    assert jnp.allclose(got, 1.5)


def test_accepts_a_time_in_other_units() -> None:
    """``t`` is converted to the grid's unit, not assumed to share it."""
    p, slope = _linear()
    got = float(u.ustrip(u.unit("Msun"), p(u.Q(1500.0, "Myr"))))
    assert got == pytest.approx(1.0 + slope * 1.5, rel=1e-10)


def test_is_jittable_and_differentiable() -> None:
    """It is evaluated inside an integrator's scan body and differentiated."""
    p, slope = _linear()

    def f(tv):
        return u.ustrip(u.unit("Msun"), p(u.Q(tv, "Gyr")))

    assert float(jax.jit(f)(jnp.asarray(1.25))) == pytest.approx(
        1.0 + slope * 1.25, rel=1e-10
    )
    # d/dt of a line is its slope, and the clamp makes it zero outside.
    assert float(jax.grad(f)(jnp.asarray(1.25))) == pytest.approx(slope, rel=1e-8)
    assert float(jax.grad(f)(jnp.asarray(99.0))) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("ts", "values", "match"),
    [
        (
            u.Q(jnp.zeros((2, 2)), "Gyr"),
            u.Q(jnp.zeros((2, 2)), "Msun"),
            "must be 1-D",
        ),
        (
            u.Q(jnp.asarray([0.0]), "Gyr"),
            u.Q(jnp.asarray([1.0]), "Msun"),
            "at least 2 entries",
        ),
        (
            u.Q(jnp.asarray([0.0, 1.0]), "Gyr"),
            u.Q(jnp.asarray([1.0, 2.0, 3.0]), "Msun"),
            "time on the leading axis",
        ),
    ],
)
def test_the_factory_rejects_mismatched_inputs(ts, values, match: str) -> None:
    """The grid and the table must agree, and there must be something to interpolate."""
    with pytest.raises(ValueError, match=match):
        time_interpolated_parameter(ts, values)


@pytest.mark.parametrize("time_unit", ["Gyr", "Myr"])
def test_the_knot_derivatives_carry_the_right_dimension(time_unit: str) -> None:
    """``derivs`` is d(values)/d(ts), so it is ``values.unit / ts.unit``.

    REGRESSION: it was labelled ``values.unit``. The array was right and the
    interpolated answer was right -- `_interpolate` reads raw ``.value`` for
    the grid, the values and the derivatives alike, so the label never
    reached it -- but the stored quantity was dimensionally a lie, and
    exactly backwards about it: converting to the correct ``Msun/Gyr``
    raised, while converting to the wrong ``Msun`` silently succeeded.

    Parametrized over the time unit because the bug is invisible unless the
    label is compared against ``ts``: with a single unit in play, any wrong
    label is a fixed wrong label.
    """
    slope = 3.0  # Msun per `time_unit`
    ts = u.Q(jnp.asarray([0.0, 1.0, 2.0, 3.0]), time_unit)
    values = u.Q(1.0 + slope * u.ustrip(u.unit(time_unit), ts), "Msun")

    p = time_interpolated_parameter(ts, values)
    derivs = p.args[2]

    assert derivs.unit == values.unit / ts.unit
    assert jnp.allclose(u.ustrip(u.unit(f"Msun/{time_unit}"), derivs), slope)
    # The answer is unchanged by the relabelling.
    assert jnp.allclose(
        u.ustrip(u.unit("Msun"), p(u.Q(1.5, time_unit))), 1.0 + slope * 1.5
    )
