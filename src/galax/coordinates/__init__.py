""":mod:`galax.coordinates` --- Coordinate systems and transformations.

Copyright (c) 2023 galax maintainers. All rights reserved.
"""

__all__ = [
    # Modules
    "ops",
    "frames",
    # Base
    "AbstractPhaseSpaceObject",
    # Coordinates
    "AbstractBasicPhaseSpaceCoordinate",
    "AbstractPhaseSpaceCoordinate",
    "PhaseSpaceCoordinate",
    "AbstractCompositePhaseSpaceCoordinate",
    "CompositePhaseSpaceCoordinate",
    "ComponentShapeTuple",
    # PSPs
    "PhaseSpacePosition",
    "PSPComponentShapeTuple",
    # Protocols
    "PhaseSpaceObjectInterpolant",
]

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.coordinates", RUNTIME_TYPECHECKER):
    from . import frames, ops
    from ._src.base import AbstractPhaseSpaceObject
    from ._src.interp import PhaseSpaceObjectInterpolant
    from ._src.pscs import (
        AbstractBasicPhaseSpaceCoordinate,
        AbstractCompositePhaseSpaceCoordinate,
        AbstractPhaseSpaceCoordinate,
        ComponentShapeTuple,
        CompositePhaseSpaceCoordinate,
        PhaseSpaceCoordinate,
    )
    from ._src.psps import (
        ComponentShapeTuple as PSPComponentShapeTuple,
        PhaseSpacePosition,
    )

# Clean up the namespace
del install_import_hook, RUNTIME_TYPECHECKER
