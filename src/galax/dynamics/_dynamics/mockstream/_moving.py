"""Moving object. TEMPORARY CLASS.

This class adds time-dependent translations to a potential.
It is NOT careful about the implied changes to velocity, etc.
"""

__all__: list[str] = []


from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, final

import equinox as eqx

import coordinax as cx
import unxt as u
from coordinax.operators._base import op_call_dispatch

import galax.coordinates as gc
import galax.typing as gt


@final
class TimeDependentSpatialTranslationOperator(cx.operators.AbstractOperator):  # type: ignore[misc]
    r"""Operator for time-dependent translation."""

    translation: Callable[[gt.FloatQScalar], cx.AbstractVector] = eqx.field(static=True)
    """The spatial translation."""

    @op_call_dispatch
    def __call__(
        self: "TimeDependentSpatialTranslationOperator",
        q: cx.AbstractVector,
        t: u.AbstractQuantity,
        /,
    ) -> tuple[cx.AbstractVector, gt.RealQScalar]:
        """Do."""
        return (q + self.translation(t), t)

    @op_call_dispatch
    def __call__(
        self: "TimeDependentSpatialTranslationOperator",
        q: gc.AbstractPhaseSpacePosition,
        /,
    ) -> gc.AbstractPhaseSpacePosition:
        """Apply the operator to a phase-space-time position."""
        return replace(q, q=q.q + self.translation(q.t))

    @property
    def is_inertial(self) -> Literal[False]:
        """Galilean translation is an inertial frame-preserving transformation."""
        return False

    @property
    def inverse(self) -> "TimeDependentSpatialTranslationOperator":
        """The inverse of the operator."""
        func = (
            self.translation.func
            if isinstance(self.translation, _Inv)
            else _Inv(self.translation)
        )
        return TimeDependentSpatialTranslationOperator(translation=func)


@dataclass
class _Inv:
    func: Callable[[gt.FloatQScalar], cx.AbstractVector]

    def __call__(self, t: gt.FloatQScalar) -> cx.AbstractVector:
        return -self.func(t)


# TODO: move to the class in py3.11+
@cx.operators.AbstractOperator.constructor._f.register  # type: ignore[misc]  # noqa: SLF001
def constructor(
    cls: type[TimeDependentSpatialTranslationOperator],
    obj: gc.PhaseSpacePositionInterpolant,
    /,
) -> TimeDependentSpatialTranslationOperator:
    return cls(lambda t: obj(t).q)
