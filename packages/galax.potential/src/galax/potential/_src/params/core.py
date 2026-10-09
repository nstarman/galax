"""Parameters on a Potential."""

__all__ = [
    "LinearParameter",
    "CustomParameter",
]

import functools as ft

from collections.abc import Callable
from typing import Any, final

import equinox as eqx
import jax
import jax.core

import unxt as u
from unxt.quantity import AllowValue

import galax.potential.custom_types as gt
from .base import AbstractParameter

t0 = u.Q(0, "Myr")


class LinearParameter(AbstractParameter):
    """Linear time dependence Parameter.

    This is in point-slope form, where the parameter is given by

    .. math::

        p(t) = m * (t - ti) + p(ti)

    Parameters
    ----------
    slope : Quantity[float, (), "[parameter]/[time]"]
        The slope of the linear parameter.
    point_time : Array[float, (), "time"]
        The time at which the parameter is equal to the intercept.
    point_value : Quantity[float, (), "[parameter]"]
        The value of the parameter at the ``point_time``.

    Examples
    --------
    >>> import galax.potential as gp
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    >>> lp = gp.params.LinearParameter(slope=u.Q(-1e3, "Msun/yr"),
    ...     point_time=u.Q(0, "Myr"), point_value=u.Q(1e12, "Msun"))

    >>> lp(u.Q(0, "Gyr")).uconvert("Msun")
    Q(1.e+12, 'solMass')

    >>> jnp.round(lp(u.Q(1.0, "Gyr")), 3)
    Q(0., 'Gyr solMass / yr')

    The parameter can then be used as a potential's field:

    >>> pot = gp.KeplerPotential(m_tot=lp, units="galactic")

    """

    slope: gt.QuSzAny = eqx.field(converter=u.Q.from_)
    point_time: gt.BBtQuSz0 = eqx.field(converter=u.Quantity["time"].from_)
    point_value: gt.QuSzAny = eqx.field(converter=u.Q.from_)

    def __check_init__(self) -> None:
        """Check the initialization of the class."""
        # TODO: check point_value and slope * point_time have the same dimensions

    @ft.partial(jax.jit, static_argnames=("ustrip",))
    def __call__(
        self, t: gt.BBtQuSz0, *, ustrip: u.AbstractUnit | None = None, **_: Any
    ) -> gt.QuSzAny | gt.SzAny:
        """Return the parameter value.

        .. math::

            p(t) = m * (t - ti) + p(ti)

        Returns
        -------
        Array[float, "*shape"]
            The constant parameter value.

        Examples
        --------
        >>> from galax.potential.params import LinearParameter
        >>> import unxt as u
        >>> import quaxed.numpy as jnp

        >>> lp = LinearParameter(slope=u.Q(-1, "Msun/yr"),
        ...                      point_time=u.Q(0, "Myr"),
        ...                      point_value=u.Q(1e9, "Msun"))

        >>> lp(u.Q(0, "Gyr")).uconvert("Msun")
        Q(1.e+09, 'solMass')

        >>> jnp.round(lp(u.Q(1, "Gyr")), 3)
        Q(0., 'Gyr solMass / yr')

        """
        out = self.slope * (t - self.point_time) + self.point_value
        return out if ustrip is None else u.ustrip(AllowValue, ustrip, out)


#####################################################################
# User-defined Parameter
# For passing a function as a parameter.


@final
class CustomParameter(AbstractParameter):
    """User-defined Parameter.

    Parameters
    ----------
    func : Callable[[BBtRealQuSz0], Array[float, (*shape,)]]
        The function to use to compute the parameter value.
    args : tuple
        Extra arguments passed to ``func`` after ``t``. Put any *data* the
        function needs here rather than closing over it -- see below.

    Examples
    --------
    >>> from galax.potential.params import CustomParameter
    >>> import unxt as u
    >>> from unxts.parametric import ParametricQuantity

    >>> def func(t: u.Quantity["time"]) -> ParametricQuantity["mass"]:
    ...     return u.Q(1e9, "Msun/Gyr") * t

    >>> up = CustomParameter(func=func)
    >>> up(u.Q(1e3, "Myr"))
    Q(1.e+12, 'Myr solMass / Gyr')

    Data the function needs goes in ``args``, not in a closure:

    >>> import quaxed.numpy as jnp
    >>> def scaled(t, m0):
    ...     return m0 * u.ustrip("Gyr", t)

    >>> up = CustomParameter(func=scaled, args=(u.Q(1e9, "Msun"),))
    >>> up(u.Q(2.0, "Gyr"))
    Q(2.e+09, 'solMass')

    ``func`` is a *static* field -- it is hashed, not traced -- so arrays
    captured in a closure are invisible to JAX. They are not pytree leaves,
    which costs two things that are easy to miss. A closure is hashed by
    identity, so rebuilding one over the *same* data is a fresh `jax.jit`
    cache entry and recompiles; and `equinox.tree_serialise_leaves` writes
    nothing for it, so saving a potential silently drops the data. ``args``
    is an ordinary field, so its arrays are leaves and both work.

    """

    # `Callable[..., Any]`, not `ParameterCallable`: with `args` the function
    # takes whatever data it was given after `t`, so its signature is the
    # caller's business. `ParameterCallable` still describes `__call__` below,
    # which is what a *parameter* must look like and is unchanged.
    func: Callable[..., Any] = eqx.field(static=True)
    args: tuple[Any, ...] = eqx.field(default=())

    @ft.partial(jax.jit, static_argnames=("ustrip",))
    def __call__(
        self, t: gt.BBtQuSz0, *, ustrip: u.AbstractUnit | None = None, **kwargs: Any
    ) -> gt.QuSzAny | gt.SzAny:
        out = self.func(t, *self.args, **kwargs)
        return out if ustrip is None else u.ustrip(AllowValue, ustrip, out)
