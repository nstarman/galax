"""Base UnitSystem class."""

from __future__ import annotations

__all__ = ["AbstractUnitSystem", "UNITSYSTEMS_REGISTRY"]

from collections.abc import Iterator
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, TypeAlias, get_args, get_type_hints

import astropy.units as u
from astropy.units import PhysicalType as Dimension
from astropy.units.physical import _physical_unit_mapping

from ._utils import get_dimensions_name, is_annotated

UnitT: TypeAlias = u.UnitBase
Unit = u.UnitBase

_UNITSYSTEMS_REGISTRY: dict[tuple[Dimension, ...], type[AbstractUnitSystem]] = {}
UNITSYSTEMS_REGISTRY = MappingProxyType(_UNITSYSTEMS_REGISTRY)


@dataclass(frozen=True)
class AbstractUnitSystem:
    """Represents a system of units.

    At minimum, this consists of a set of length, time, mass, and angle units, but may
    also contain preferred representations for composite units. For example, the base
    unit system could be ``{kpc, Myr, Msun, radian}``, but you can also specify a
    preferred velocity unit, such as ``km/s``.

    This class behaves like a dictionary with keys set by physical types (i.e. "length",
    "velocity", "energy", etc.). If a unit for a particular physical type is not
    specified on creation, a composite unit will be created with the base units. See the
    examples below for some demonstrations.

    Parameters
    ----------
    *units, **units
        The units that define the unit system. At minimum, this must contain length,
        time, mass, and angle units. If passing in keyword arguments, the keys must be
        valid :mod:`astropy.units` physical types.

    Examples
    --------
    If only base units are specified, any physical type specified as a key
    to this object will be composed out of the base units::

        >>> from galax.units import unitsystem
        >>> usys = unitsystem(u.m, u.s, u.kg, u.radian)
        >>> usys["velocity"]
        Unit("m / s")

    However, preferred representations for composite units can also be specified::

        >>> usys = unitsystem(u.m, u.s, u.kg, u.radian, u.erg)
        >>> usys["energy"]
        Unit("m2 kg / s2")
        >>> usys.preferred("energy")
        Unit("erg")

    This is useful for Galactic dynamics where lengths and times are usually given in
    terms of ``kpc`` and ``Myr``, but velocities are often specified in ``km/s``::

        >>> usys = unitsystem(u.kpc, u.Myr, u.Msun, u.radian, u.km/u.s)
        >>> usys["velocity"]
        Unit("kpc / Myr")
        >>> usys.preferred("velocity")
        Unit("km / s")
    """

    _base_field_names: ClassVar[tuple[str, ...]]
    _base_dimensions: ClassVar[tuple[Dimension, ...]]

    def __init_subclass__(cls) -> None:
        # Register class with a tuple of it's dimensions.
        # This requires processeing the type hints, not the dataclass fields
        # since those are made after the original class is defined.
        type_hints = get_type_hints(cls, include_extras=True)

        field_names = []
        dimensions_ = []
        for name, type_hint in type_hints.items():
            # Check it's Annotated
            if not is_annotated(type_hint):
                continue

            # Get the arguments to Annotated
            origin, *f_args = get_args(type_hint)

            # Check that the first argument is a UnitBase
            if not issubclass(origin, Unit):
                continue

            # Need for one of the arguments to be a PhysicalType
            f_dims = [x for x in f_args if isinstance(x, Dimension)]
            if not f_dims:
                msg = f"Field {name} must be an Annotated with a dimension."
                raise TypeError(msg)
            if len(f_dims) > 1:
                msg = f"Field {name} must be an Annotated with only one dimension."
                raise TypeError(msg)

            field_names.append(get_dimensions_name(name))
            dimensions_.append(f_dims[0])

        dimensions = tuple(dimensions_)  # freeze

        # Check the unitsystem is not already registered
        if dimensions in _UNITSYSTEMS_REGISTRY:
            msg = f"Unit system with dimensions {dimensions} already exists."
            raise ValueError(msg)

        # Add attributes to the class
        cls._base_field_names = tuple(field_names)
        cls._base_dimensions = dimensions

        # Register the class
        _UNITSYSTEMS_REGISTRY[dimensions] = cls

    def __post_init__(self) -> None:
        self._registry: dict[Dimension, UnitT]
        object.__setattr__(self, "_registry", {})

        for dim, unit in zip(self.base_dimensions, self.base_units, strict=True):
            pt = unit.physical_type
            if pt != dim:
                msg = f"Unit {unit} must have dimensions {dim}, not {pt}."
                raise ValueError(msg)
            self._registry[pt] = unit

    @property  # TODO: classproperty
    def base_dimensions(self) -> tuple[Dimension, ...]:
        """Dimensions required for the unit system."""
        return self._base_dimensions

    @property
    def base_units(self) -> tuple[UnitT, ...]:
        """List of core units."""
        return tuple(getattr(self, k) for k in self._base_field_names)

    def __getitem__(self, key: str | u.PhysicalType) -> u.UnitBase:
        key = u.get_physical_type(key)
        if key in self.base_dimensions:
            return self._registry[key]

        unit = None
        for k, v in _physical_unit_mapping.items():
            if v == key:
                unit = u.Unit(" ".join([f"{x}**{y}" for x, y in k]))
                break

        if unit is None:
            msg = f"Physical type '{key}' doesn't exist in unit registry."
            raise ValueError(msg)

        unit = unit.decompose(self.base_units)
        unit._scale = 1.0  # noqa: SLF001
        return unit

    def __len__(self) -> int:
        # Note: This is required for q.decompose(usys) to work, where q is a Quantity
        return len(self.base_dimensions)

    def __iter__(self) -> Iterator[UnitT]:
        yield from self.base_units

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractUnitSystem):
            return NotImplemented
        return bool(self._registry == other._registry)

    def __hash__(self) -> int:
        """Hash the unit system."""
        return hash(tuple(zip(self.base_dimensions, self.base_units, strict=True)))

    def __str__(self) -> str:
        fields = ", ".join(
            [f"{k}={getattr(self, k)!s}" for k in self._base_field_names]
        )
        return f"UnitSystem({fields})"

    # -------------------------------------

    def preferred(self, key: str | Dimension) -> UnitT:
        """Return the preferred unit for a given physical type."""
        key = u.get_physical_type(key)
        if key in self._registry:
            return self._registry[key]
        return self[key]

    def as_preferred(self, quantity: u.Quantity) -> u.Quantity:
        """Convert a quantity to the preferred unit for this unit system."""
        return quantity.to(self.preferred(quantity.unit.physical_type))
