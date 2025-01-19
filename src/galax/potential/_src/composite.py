"""Composite Potential."""

__all__ = ["CompositePotential"]


from dataclasses import KW_ONLY
from types import MappingProxyType
from typing import Any, ClassVar, TypeAlias, cast, final

import equinox as eqx

import unxt as u
from xmmutablemap import ImmutableMap
from zeroth import zeroth

from .base import AbstractPotential, default_constants
from .base_multi import AbstractCompositePotential
from .params.attr import CompositeParametersAttribute

ArgPotential: TypeAlias = (
    dict[str, AbstractPotential] | tuple[tuple[str, AbstractPotential], ...]
)


@final
class CompositePotential(
    AbstractCompositePotential,
    ImmutableMap[str, AbstractPotential],  # type: ignore[misc]
    strict=False,
):
    """Composite Potential."""

    parameters: ClassVar = CompositeParametersAttribute(MappingProxyType({}))

    _data: dict[str, AbstractPotential]
    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(
        init=False, static=True, converter=u.unitsystem
    )
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    def __init__(
        self,
        potentials: ArgPotential = (),
        /,
        *,
        units: Any = None,
        constants: Any = default_constants,
        **kwargs: AbstractPotential,
    ) -> None:
        ImmutableMap.__init__(self, potentials, **kwargs)  # <- ImmutableMap.__init__

        # __post_init__ stuff:
        # Check that all potentials have the same unit system
        units_ = units if units is not None else zeroth(self.values()).units
        usys = u.unitsystem(units_)
        if not all(p.units == usys for p in self.values()):
            msg = "all potentials must have the same unit system"
            raise ValueError(msg)
        object.__setattr__(self, "units", usys)  # TODO: not call `object.__setattr__`

        # TODO: some similar check that the same constants are the same, e.g.
        #       `G` is the same for all potentials. Or use `constants` to update
        #       the `constants` of every potential (before `super().__init__`)
        object.__setattr__(self, "constants", ImmutableMap(constants))

        # Apply the unit system to any parameters.
        self._apply_unitsystem()

    def __repr__(self) -> str:  # TODO: not need this hack
        return cast(str, ImmutableMap.__repr__(self))
