"""Unit utils."""

from __future__ import annotations

__all__: list[str] = []

from functools import singledispatch
from typing import (  # type: ignore[attr-defined]
    Annotated,
    Any,
    TypeGuard,
    _AnnotatedAlias,
)

from astropy.units import PhysicalType as Dimensions, UnitBase, get_physical_type

AnnotationType = type(Annotated[int, "_"])
_apy_speed = get_physical_type("speed")


def is_annotated(hint: Any) -> TypeGuard[_AnnotatedAlias]:
    """Check if a type hint is an `Annotated` type."""
    return type(hint) is AnnotationType  # pylint: disable=unidiomatic-typecheck


@singledispatch
def get_dimensions_name(pt: Any, /) -> str:
    """Get the dimension name from the object."""
    msg = f"Cannot get dimension name from {pt!r}."
    raise TypeError(msg)


@get_dimensions_name.register
def _get_dimensions_name_from_str(pt: str, /) -> str:
    return "_".join(pt.split("_"))


@get_dimensions_name.register
def _get_dimensions_name_from_astropy_dimension(pt: Dimensions, /) -> str:
    # TODO: this is not deterministic b/c ``_physical_type`` is a set
    #       that's why the `if` statement is needed.
    if pt == _apy_speed:
        return "speed"
    return _get_dimensions_name_from_str(next(iter(pt._physical_type)))  # noqa: SLF001


@get_dimensions_name.register
def _get_dimensions_name_from_unit(pt: UnitBase, /) -> str:
    return _get_dimensions_name_from_astropy_dimension(pt.physical_type)
