""":mod:`galax.coordinates` --- Coordinate systems and transformations.

Copyright (c) 2023 galax maintainers. All rights reserved.
"""

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.coordinates", RUNTIME_TYPECHECKER):
    from . import frames, ops
    from ._src.psps import (
        AbstractCompositePhaseSpacePosition,
        AbstractOnePhaseSpacePosition,
        AbstractPhaseSpacePosition,
        ComponentShapeTuple,
        CompositePhaseSpacePosition,
        PhaseSpacePosition,
        PhaseSpacePositionInterpolant,
    )

__all__ = [
    # Modules
    "ops",
    "frames",
    # Contents
    "AbstractPhaseSpacePosition",
    "AbstractOnePhaseSpacePosition",
    "PhaseSpacePosition",
    "AbstractCompositePhaseSpacePosition",
    "CompositePhaseSpacePosition",
    # Utils
    "ComponentShapeTuple",
    # Protocols
    "PhaseSpacePositionInterpolant",
]

# Clean up the namespace
del install_import_hook, RUNTIME_TYPECHECKER
