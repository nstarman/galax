"""Compatibility."""

__all__: list[str] = []

from typing import Any

from astropy.coordinates import BaseRepresentation
from astropy.units import Quantity as APYQuantity
from plum import convert, dispatch

import coordinax as cx
import unxt as u

# =============================================================================
# parse_to_quantity


@dispatch
def parse_to_quantity(value: APYQuantity, /, **_: Any) -> u.Quantity:
    return convert(value, u.Quantity)


@dispatch
def parse_to_quantity(rep: BaseRepresentation, /, **_: Any) -> u.Quantity:
    cart = convert(rep, cx.CartesianPos3D)
    return parse_to_quantity(cart)
