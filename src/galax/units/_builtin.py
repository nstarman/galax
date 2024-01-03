"""Built-in unit systems."""

from __future__ import annotations

__all__ = ["DimensionlessUnitSystem", "LTMAUnitSystem", "LTMAVUnitSystem"]

from dataclasses import dataclass
from typing import Annotated, TypeAlias, final

import astropy.units as u
from astropy.units import dimensionless_unscaled

from ._base import AbstractUnitSystem
from ._builtin_dimensions import (
    angle as _angle,
    dimensionless as _dimensionless,
    length as _length,
    mass as _mass,
    speed as _speed,
    time as _time,
)

Unit: TypeAlias = u.UnitBase

_dimless_insts: dict[type[DimensionlessUnitSystem], DimensionlessUnitSystem] = {}


@final
@dataclass(frozen=True)
class DimensionlessUnitSystem(AbstractUnitSystem):
    """A unit system with only dimensionless units."""

    dimensionless: Annotated[Unit, _dimensionless] = dimensionless_unscaled

    def __new__(cls) -> DimensionlessUnitSystem:
        # Check if instance already exists
        if cls in _dimless_insts:
            return _dimless_insts[cls]
        # Create new instance and cache it
        self = super().__new__(cls)
        _dimless_insts[cls] = self
        return self

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.dimensionless is not dimensionless_unscaled:
            msg = "DimensionlessUnitSystem must have a dimensionless unit"
            raise ValueError(msg)

    def __str__(self) -> str:
        return "UnitSystem(dimensionless)"


@final
@dataclass(frozen=True)
class LTMAUnitSystem(AbstractUnitSystem):
    """Length, time, mass, angle unit system."""

    length: Annotated[Unit, _length]
    time: Annotated[Unit, _time]
    mass: Annotated[Unit, _mass]
    angle: Annotated[Unit, _angle]


@final
@dataclass(frozen=True)
class LTMAVUnitSystem(AbstractUnitSystem):
    """Length, time, mass, angle, speed unit system."""

    length: Annotated[Unit, _length]
    time: Annotated[Unit, _time]
    mass: Annotated[Unit, _mass]
    angle: Annotated[Unit, _angle]
    speed: Annotated[Unit, _speed]
