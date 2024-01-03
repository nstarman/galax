"""Built-in unit systems."""

from __future__ import annotations

__all__ = ["dimensionless", "length", "mass", "speed", "time", "angle"]

import astropy.units as u

dimensionless = u.get_physical_type("dimensionless")
length = u.get_physical_type("length")
mass = u.get_physical_type("mass")
time = u.get_physical_type("time")
speed = u.get_physical_type("speed")
angle = u.get_physical_type("angle")
