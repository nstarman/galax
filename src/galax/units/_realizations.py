"""Realizations of unit systems."""

from __future__ import annotations

__all__ = ["dimensionless", "galactic", "solarsystem"]

import astropy.units as u

from ._builtin import DimensionlessUnitSystem, LTMAUnitSystem, LTMAVUnitSystem

# Dimensionless. This is a singleton.
dimensionless = DimensionlessUnitSystem()

# Galactic unit system
galactic = LTMAVUnitSystem(u.kpc, u.Myr, u.Msun, u.radian, u.km / u.s)

# Solar system units
solarsystem = LTMAUnitSystem(u.au, u.yr, u.Msun, u.radian)
