"""Wrapper to add frame operations to a potential."""

__all__ = ["PotentialFrame"]


from dataclasses import replace
from typing import cast, final

import equinox as eqx
from plum import dispatch

import coordinax.ops as cxo
import unxt as u
from xmmutablemap import ImmutableMap

import galax.typing as gt
from .base import AbstractPotential


@final
class PotentialFrame(AbstractPotential):
    """Reference frame of the potential.

    Examples
    --------
    In this example, we create a triaxial Hernquist potential and apply a few
    coordinate transformations.

    First some imports:

    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp

    Now we define a triaxial Hernquist potential with a time-dependent mass:

    >>> mfunc = gp.params.UserParameter(lambda t: u.Quantity(1e12 * (1 + u.ustrip("Gyr", t) / 10), "Msun"))
    >>> pot = gp.TriaxialHernquistPotential(m_tot=mfunc, r_s=u.Quantity(1, "kpc"),
    ...                                     q1=1, q2=0.5, units="galactic")

    Let's see the triaxiality of the potential:

    >>> t = u.Quantity(0, "Gyr")
    >>> w1 = gc.PhaseSpacePosition(q=u.Quantity([1, 0, 0], "kpc"),
    ...                            p=u.Quantity([0, 1, 0], "km/s"),
    ...                            t=t)

    The triaxiality can be seen in the potential energy of the three positions:

    >>> pot.potential(w1)
    Quantity[...](Array(-2.24925108, dtype=float64), unit='kpc2 / Myr2')

    >>> q = u.Quantity([0, 1, 0], "kpc")
    >>> pot.potential(q, t)
    Quantity[...](Array(-2.24925108, dtype=float64), unit='kpc2 / Myr2')

    >>> q = u.Quantity([0, 0, 1], "kpc")
    >>> pot.potential(q, t)
    Quantity[...](Array(-1.49950072, dtype=float64), unit='kpc2 / Myr2')

    Let's apply a spatial translation to the potential:

    >>> op1 = cx.ops.GalileanSpatialTranslation.from_([3, 0, 0], "kpc")
    >>> op1
    GalileanSpatialTranslation(CartesianPos3D( ... ))

    >>> framedpot1 = gp.PotentialFrame(original_potential=pot, operator=op1)
    >>> framedpot1
    PotentialFrame(
      original_potential=TriaxialHernquistPotential( ... ),
      operator=GalileanSpatialTranslation(
        translation=CartesianPos3D( ... )
      )
    )

    Now the potential energy is different because the potential has been
    translated by 3 kpc in the x-direction:

    >>> framedpot1.potential(w1)
    Quantity[...](Array(-1.49950072, dtype=float64), unit='kpc2 / Myr2')

    This is the same as evaluating the untranslated potential at [-2, 0, 0] kpc:

    >>> q = u.Quantity([-2, 0, 0], "kpc")
    >>> pot.potential(q, u.Quantity(0, "Gyr"))
    Quantity[...](Array(-1.49950072, dtype=float64), unit='kpc2 / Myr2')

    We can also apply a time translation to the potential:

    >>> op2 = cx.ops.GalileanTranslation.from_([1_000, 0, 0, 0], "kpc")
    >>> op2.translation.t.uconvert("Myr")  # doctest: +SKIP
    Quantity['time'](Array(3.26156378, dtype=float64), unit='Myr')

    >>> framedpot2 = gp.PotentialFrame(original_potential=pot, operator=op2)

    We can see that the potential energy is the same as before, since we have
    been evaluating the potential at ``w1.t=t=0``:

    >>> framedpot2.potential(w1)
    Quantity[...](Array(-2.24851747, dtype=float64), unit='kpc2 / Myr2')

    But if we evaluate the potential at a different time, the potential energy
    will be different:

    >>> from dataclasses import replace
    >>> w2 = replace(w1, t=u.Quantity(10, "Myr"))
    >>> framedpot2.potential(w2)
    Quantity[...](Array(-2.25076672, dtype=float64), unit='kpc2 / Myr2')

    Now let's boost the potential by 200 km/s in the y-direction:

    >>> op3 = cx.ops.GalileanBoost.from_([0, 200, 0], "km/s")
    >>> op3
    GalileanBoost(CartesianVel3D( ... ))

    >>> framedpot3 = gp.PotentialFrame(original_potential=pot, operator=op3)
    >>> framedpot3.potential(w2)
    Quantity[...](Array(-1.37421204, dtype=float64), unit='kpc2 / Myr2')

    Alternatively we can rotate the potential by 90 degrees about the y-axis:

    >>> import quaxed.numpy as jnp
    >>> op4 = cx.ops.GalileanRotation.from_euler("y", u.Quantity(90, "deg"))
    >>> op4
    GalileanRotation(rotation=f64[3,3])

    >>> framedpot4 = gp.PotentialFrame(original_potential=pot, operator=op4)
    >>> framedpot4.potential(w1)
    Quantity[...](Array(-1.49950072, dtype=float64), unit='kpc2 / Myr2')

    >>> q = u.Quantity([0, 0, 1], "kpc")
    >>> framedpot4.potential(q, t)
    Quantity[...](Array(-2.24925108, dtype=float64), unit='kpc2 / Myr2')

    If you look all the way back to the first examples, you will see that the
    potential energy at [1, 0, 0] and [0, 0, 1] have swapped, as expected for a
    90 degree rotation about the y-axis!

    Lastly, we can apply a sequence of transformations to the potential. There
    are two ways to do this. The first is to create a pre-defined composite
    operator, like a :class:`~galax.coordinates.ops.GalileanOperator`:

    >>> op5 = cx.ops.GalileanOperator(rotation=op4, translation=op2, velocity=op3)
    >>> op5
    GalileanOperator(
      rotation=GalileanRotation(rotation=f64[3,3]),
      translation=GalileanTranslation( ... ),
      velocity=GalileanBoost( ... )
    )

    >>> framedpot5 = gp.PotentialFrame(original_potential=pot, operator=op5)
    >>> framedpot5.potential(w2)
    Quantity[...](Array(-1.16598068, dtype=float64), unit='kpc2 / Myr2')

    The second way is to create a custom sequence of operators. In this case we
    will make a sequence that mimics the previous example:

    >>> op6 = op4 | op2 | op3
    >>> framedpot6 = gp.PotentialFrame(original_potential=pot, operator=op6)
    >>> framedpot6.potential(w2)
    Quantity[...](Array(-1.16598068, dtype=float64), unit='kpc2 / Myr2')

    We've seen that the potential can be time-dependent, but so far the
    operators have been Galilean. Let's fix the time-dependent mass of the
    potential but make the frame operator time-dependent in a more interesting
    way. We will also exaggerate the triaxiality of the potential to make the
    effect of the rotation more obvious:

    >>> pot2 = gp.TriaxialHernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...     r_s=u.Quantity(1, "kpc"), q1=0.1, q2=0.1, units="galactic")

    >>> op7 = gc.ops.ConstantRotationZOperator(Omega_z=u.Quantity(90, "deg/Gyr"))
    >>> framedpot7 = gp.PotentialFrame(original_potential=pot2, operator=op7)

    The potential energy at a given position will change with time:

    >>> framedpot7.potential(w1).value  # t=0 Gyr
    Array(-2.24925108, dtype=float64)
    >>> framedpot7.potential(w2).value  # t=1 Gyr
    Array(-2.23568166, dtype=float64)
    """  # noqa: E501

    original_potential: AbstractPotential

    operator: cxo.AbstractOperator = eqx.field(default_factory=lambda: cxo.Identity())
    """Transformation to reference frame of the potential.

    The default is no transformation, ie the coordinates are specified in the
    'simulation' frame.
    """

    @property
    def units(self) -> u.AbstractUnitSystem:
        """The unit system of the potential."""
        return cast(u.AbstractUnitSystem, self.original_potential.units)

    @property
    def constants(self) -> ImmutableMap[str, u.Quantity]:
        """The constants of the potential."""
        return cast(ImmutableMap[str, u.Quantity], self.original_potential.constants)

    def _potential(
        self, q: gt.BtQuSz3, t: gt.BBtRealQuSz0, /
    ) -> gt.SpecificEnergyBtSz0:
        """Compute the potential energy at the given position(s).

        This method applies the frame operators to the coordinates and then
        evaluates the potential energy at the transformed coordinates.

        Parameters
        ----------
        q : Array[float, (3,)]
            The position(s) at which to compute the potential energy.
        t : float
            The time at which to compute the potential energy.

        Returns
        -------
        Array[float, (...)]
            The potential energy at the given position(s).
        """
        # Make inverse operator  # TODO: pre-compute and cache
        inv = self.operator.inverse
        # Transform the position, time.
        qp, tp = inv(q, t)
        # Evaluate the potential energy at the transformed position, time.
        return self.original_potential._potential(qp, tp)  # noqa: SLF001


#####################################################################


@dispatch
def simplify_op(frame: PotentialFrame, /) -> PotentialFrame:
    """Simplify the operators in an PotentialFrame."""
    return replace(frame, operator=cxo.simplify_op(frame.operator))
