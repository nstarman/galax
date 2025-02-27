"""Parameters on a Potential."""

__all__ = [
    "ParameterCallable",
    "AbstractParameter",
    "ConstantParameter",
    "LinearParameter",
    "UserParameter",
]

import abc
from functools import partial
from typing import Any, Protocol, final, runtime_checkable

import equinox as eqx
import jax

import unxt as u
from dataclassish.converters import Unless
from unxt._src.units.api import AstropyUnits
from unxt.quantity import AbstractQuantity, AllowValue

import galax._custom_types as gt

t0 = u.Quantity(0, "Myr")


@runtime_checkable
class ParameterCallable(Protocol):
    """Protocol for a Parameter callable."""

    def __call__(
        self, t: gt.BBtQuSz0, *, ustrip: AstropyUnits | None = None, **kwargs: Any
    ) -> gt.QuSzAny | gt.SzAny:
        """Compute the parameter value at the given time(s).

        Parameters
        ----------
        t : `~galax.typing.BBtQuSz0`
            Time(s) at which to compute the parameter value.
        ustrip : Unit | None
            Unit to strip from the parameter value.
            If None, the parameter value is returned with its original unit.
        **kwargs : Any
            Additional parameters to pass to the parameter function.

        Returns
        -------
        Array[float, "*shape"]
            Parameter value(s) at the given time(s).
        """
        ...


# -------------------------------------------


class AbstractParameter(eqx.Module, strict=True):  # type: ignore[call-arg, misc]
    """Abstract base class for Parameters on a Potential.

    Parameters are time-dependent quantities that are used to define a
    Potential. They can be constant (see `ConstantParameter`), or they can be
    functions of time.

    """

    @abc.abstractmethod
    def __call__(
        self, t: gt.BBtQuSz0, *, ustrip: AstropyUnits | None = None, **kwargs: Any
    ) -> gt.QuSzAny:
        """Compute the parameter value at the given time(s).

        Parameters
        ----------
        t : `~galax.typing.BBtQuSz0`
            The time(s) at which to compute the parameter value.
        ustrip : Unit | None
            The unit to strip from the parameter value. If None, the
            parameter value is returned with its original unit.
        **kwargs : Any
            Additional parameters to pass to the parameter function.

        Returns
        -------
        Array[float, "*shape"]
            The parameter value at times ``t``.
        """
        ...


#####################################################################


@final
class ConstantParameter(AbstractParameter):
    """Time-independent potential parameter.

    Examples
    --------
    >>> from galax.potential.params import ConstantParameter
    >>> import unxt as u

    >>> p = ConstantParameter(value=u.Quantity(1., "Msun"))
    >>> p
    ConstantParameter(Quantity['mass'](Array(1., dtype=float64, ...), unit='solMass'))

    The parameter value is constant:

    >>> p(u.Quantity(0, "Gyr"))
    Quantity['mass'](Array(1., dtype=float64, ...), unit='solMass')

    >>> p(u.Quantity(1, "Gyr")) - p(u.Quantity(2, "Gyr"))
    Quantity['mass'](Array(0., dtype=float64, ...), unit='solMass')

    We can do some arithmetic with the parameter, which degrades it
    back to a `unxt.Quantity`:

    >>> p + u.Quantity(2, "Msun")
    Quantity['mass'](Array(3., dtype=float64, ...), unit='solMass')

    >>> u.Quantity(2, "Msun") + p
    Quantity['mass'](Array(3., dtype=float64, ...), unit='solMass')

    >>> p - u.Quantity(2, "Msun")
    Quantity['mass'](Array(-1., dtype=float64, ...), unit='solMass')

    >>> u.Quantity(2, "Msun") - p
    Quantity['mass'](Array(1., dtype=float64, ...), unit='solMass')

    >>> p * 2
    Quantity['mass'](Array(2., dtype=float64, ...), unit='solMass')

    >>> 2 * p
    Quantity['mass'](Array(2., dtype=float64, ...), unit='solMass')

    >>> p / 2
    Quantity['mass'](Array(0.5, dtype=float64, ...), unit='solMass')

    >>> 2 / p
    Quantity['kg-1'](Array(2., dtype=float64, ...), unit='1 / solMass')

    """

    # TODO: link this shape to the return shape from __call__
    value: gt.QuSzAny = eqx.field(converter=Unless(AbstractQuantity, u.Quantity.from_))
    """The time-independent value of the parameter."""

    @partial(jax.jit, static_argnames=("ustrip",))
    def __call__(
        self,
        t: gt.BBtQuSz0 = t0,  # noqa: ARG002
        *,
        ustrip: AstropyUnits | None = None,
        **__: Any,
    ) -> gt.QuSzAny:
        """Return the constant parameter value.

        Parameters
        ----------
        t : `~galax.typing.BBtQuSz0`, optional
            This is ignored and is thus optional. Note that for most
            :class:`~galax.potential.AbstractParameter` the time is required.
        ustrip : Unit | None
            The unit to strip from the parameter value. If None, the
            parameter value is returned with its original unit.
        **kwargs : Any
            This is ignored.

        """
        return (
            self.value if ustrip is None else u.ustrip(AllowValue, ustrip, self.value)
        )

    # -------------------------------------------
    # String representation

    def __repr__(self) -> str:
        """Return string representation.

        Examples
        --------
        >>> from galax.potential.params import ConstantParameter
        >>> import unxt as u

        >>> p = ConstantParameter(value=u.Quantity(1, "Msun"))
        >>> p
        ConstantParameter(Quantity['mass'](Array(1, dtype=int64, ...), unit='solMass'))

        """
        return f"{self.__class__.__name__}({self.value!r})"

    # -------------------------------------------
    # Arithmetic operations
    # TODO: do better than the lambda functions
    __add__ = lambda self, other: self.value + other  # noqa: E731
    __radd__ = lambda self, other: other + self.value  # noqa: E731
    __sub__ = lambda self, other: self.value - other  # noqa: E731
    __rsub__ = lambda self, other: other - self.value  # noqa: E731
    __mul__ = lambda self, other: self.value * other  # noqa: E731
    __rmul__ = lambda self, other: other * self.value  # noqa: E731
    __truediv__ = lambda self, other: self.value / other  # noqa: E731
    __rtruediv__ = lambda self, other: other / self.value  # noqa: E731


#####################################################################
# Linear time dependence Parameter


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
    >>> from galax.potential.params import LinearParameter
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    >>> lp = LinearParameter(slope=u.Quantity(-1, "Msun/yr"),
    ...                      point_time=u.Quantity(0, "Myr"),
    ...                      point_value=u.Quantity(1e9, "Msun"))

    >>> lp(u.Quantity(0, "Gyr")).uconvert("Msun")
    Quantity['mass'](Array(1.e+09, dtype=float64), unit='solMass')

    >>> jnp.round(lp(u.Quantity(1.0, "Gyr")), 3)
    Quantity['mass'](Array(0., dtype=float64, ...), unit='Gyr solMass / yr')

    """

    slope: gt.QuSzAny = eqx.field(converter=u.Quantity.from_)
    point_time: gt.BBtQuSz0 = eqx.field(converter=u.Quantity["time"].from_)
    point_value: gt.QuSzAny = eqx.field(converter=u.Quantity.from_)

    def __check_init__(self) -> None:
        """Check the initialization of the class."""
        # TODO: check point_value and slope * point_time have the same dimensions

    @partial(jax.jit, static_argnames=("ustrip",))
    def __call__(
        self, t: gt.BBtQuSz0, *, ustrip: AstropyUnits | None = None, **_: Any
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

        >>> lp = LinearParameter(slope=u.Quantity(-1, "Msun/yr"),
        ...                      point_time=u.Quantity(0, "Myr"),
        ...                      point_value=u.Quantity(1e9, "Msun"))

        >>> lp(u.Quantity(0, "Gyr")).uconvert("Msun")
        Quantity['mass'](Array(1.e+09, dtype=float64), unit='solMass')

        >>> jnp.round(lp(u.Quantity(1, "Gyr")), 3)
        Quantity['mass'](Array(0., dtype=float64), unit='Gyr solMass / yr')

        """
        out = self.slope * (t - self.point_time) + self.point_value
        return out if ustrip is None else u.ustrip(AllowValue, ustrip, out)


#####################################################################
# User-defined Parameter
# For passing a function as a parameter.


@final
class UserParameter(AbstractParameter):
    """User-defined Parameter.

    Parameters
    ----------
    func : Callable[[BBtRealQuSz0], Array[float, (*shape,)]]
        The function to use to compute the parameter value.

    Examples
    --------
    >>> from galax.potential.params import UserParameter
    >>> import unxt as u

    >>> def func(t: u.Quantity["time"]) -> u.Quantity["mass"]:
    ...     return u.Quantity(1e9, "Msun/Gyr") * t

    >>> up = UserParameter(func=func)
    >>> up(u.Quantity(1e3, "Myr"))
    Quantity['mass'](Array(1.e+12, dtype=float64, ...), unit='Myr solMass / Gyr')

    """

    func: ParameterCallable = eqx.field(static=True)

    @partial(jax.jit, static_argnames=("ustrip",))
    def __call__(
        self, t: gt.BBtQuSz0, *, ustrip: AstropyUnits | None = None, **kwargs: Any
    ) -> gt.QuSzAny | gt.SzAny:
        out = self.func(t, **kwargs)
        return out if ustrip is None else u.ustrip(AllowValue, ustrip, out)
