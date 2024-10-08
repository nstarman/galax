"""Tests for `galax.potential._src.frame` package."""

from dataclasses import replace

from plum import convert

import quaxed.numpy as jnp
from unxt import Quantity

import galax.coordinates.operators as gco
import galax.potential as gp


def test_bar_means_of_rotation() -> None:
    """Test the equivalence of hard-coded vs operator means of rotation."""
    base_pot = gp.BarPotential(
        m_tot=Quantity(1e9, "Msun"),
        a=Quantity(5.0, "kpc"),
        b=Quantity(0.1, "kpc"),
        c=Quantity(0.1, "kpc"),
        Omega=Quantity(0.0, "Hz"),
        units="galactic",
    )

    Omega_z_freq = Quantity(220.0, "1/Myr")
    Omega_z_angv = jnp.multiply(Omega_z_freq, Quantity(1.0, "rad"))

    # Hard-coded means of rotation
    hardpot = replace(base_pot, Omega=Omega_z_freq)

    # Operator means of rotation
    op = gco.ConstantRotationZOperator(Omega_z=Omega_z_angv)
    framedpot = gp.PotentialFrame(base_pot, op)

    # quick test of the op
    q = Quantity([5.0, 0.0, 0.0], "kpc")
    t = Quantity(0.0, "Myr")

    newq, newt = op.inverse(q, t)
    assert isinstance(newq, Quantity)
    assert isinstance(newt, Quantity)

    # They should be equivalent at t=0
    assert framedpot.potential(q, t) == hardpot.potential(q, t)
    assert jnp.array_equal(
        convert(framedpot.acceleration(q, t), Quantity),
        convert(hardpot.acceleration(q, t), Quantity),
    )

    # They should be equivalent at t=110 Myr (1/2 period)
    t = Quantity(110, "Myr")
    assert framedpot.potential(q, t) == hardpot.potential(q, t)
    assert jnp.array_equal(
        convert(framedpot.acceleration(q, t), Quantity),
        convert(hardpot.acceleration(q, t), Quantity),
    )

    # They should be equivalent at t=220 Myr (1 period)
    t = Quantity(220, "Myr")
    assert framedpot.potential(q, t) == hardpot.potential(q, t)
    assert jnp.array_equal(
        convert(framedpot.acceleration(q, t), Quantity),
        convert(hardpot.acceleration(q, t), Quantity),
    )

    # They should be equivalent at t=55 Myr (1/4 period)
    t = Quantity(55, "Myr")
    assert framedpot.potential(q, t) == hardpot.potential(q, t)
    assert jnp.array_equal(
        convert(framedpot.acceleration(q, t), Quantity),
        convert(hardpot.acceleration(q, t), Quantity),
    )

    # TODO: move this test to a more appropriate location
    # Test that the frame's constants are the same as the base potential's
    assert framedpot.constants is base_pot.constants
