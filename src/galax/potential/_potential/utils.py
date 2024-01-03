"""galax: Galactic Dynamix in Jax."""


from functools import singledispatch
from typing import Any

from galax.units import (
    AbstractUnitSystem,
    dimensionless,
    galactic,
    solarsystem,
    unitsystem,
)


@singledispatch
def converter_to_usys(value: Any, /) -> AbstractUnitSystem:
    """Argument to ``eqx.field(converter=...)``."""
    msg = f"cannot convert {value} to a AbstractUnitSystem"
    raise NotImplementedError(msg)


@converter_to_usys.register
def _from_usys(value: AbstractUnitSystem, /) -> AbstractUnitSystem:
    return value


@converter_to_usys.register
def _from_none(value: None, /) -> AbstractUnitSystem:
    return dimensionless


@converter_to_usys.register(tuple)
@converter_to_usys.register(list)
def _from_args(value: tuple[Any, ...] | list[Any], /) -> AbstractUnitSystem:
    return unitsystem(*value)


@converter_to_usys.register
def _from_named(value: str, /) -> AbstractUnitSystem:
    if value == "dimensionless":
        return dimensionless
    if value == "solarsystem":
        return solarsystem
    if value == "galactic":
        return galactic

    msg = f"cannot convert {value} to a AbstractUnitSystem"
    raise NotImplementedError(msg)
