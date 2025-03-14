"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "BurkertPotential",
    "HernquistPotential",
    "IsochronePotential",
    "JaffePotential",
    "KeplerPotential",
    "PlummerPotential",
    "PowerLawCutoffPotential",
    "StoneOstriker15Potential",
    "TriaxialHernquistPotential",
]

from dataclasses import KW_ONLY
from functools import partial
from typing import Any, final

import equinox as eqx
import jax

import quaxed.lax as qlax
import quaxed.numpy as jnp
import quaxed.scipy.special as qsp
import unxt as u
from unxt.quantity import AllowValue
from xmmutablemap import ImmutableMap

import galax._custom_types as gt
from .const import BURKERT_CONST
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField

# -------------------------------------------------------------------


@final
class BurkertPotential(AbstractSinglePotential):
    """Burkert Potential.

    https://ui.adsabs.harvard.edu/abs/1995ApJ...447L..25B/abstract,
    https://iopscience.iop.org/article/10.1086/309140/fulltext/50172.text.html.

    """

    m: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass",
        doc=r"""Characteristic mass of the potential.

    $$ m0 = \pi \rho_0 r_s^3 (3 \log(2) - \pi / 2) $$

    """,
    )

    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale radius")  # type: ignore[assignment]

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Quantity.from_(t, self.units["time"])

        # Compute parameters
        m = self.m(t, ustrip=self.units["mass"])
        r_s = self.r_s(t, ustrip=self.units["length"])

        # Compute potential
        x = jnp.linalg.vector_norm(xyz, axis=-1) / r_s
        xinv = 1 / x
        prefactor = self.constants["G"].value * m / (r_s * BURKERT_CONST)
        return -prefactor * (
            jnp.pi
            - 2 * (1 + xinv) * jnp.atan(x)
            + 2 * (1 + xinv) * jnp.log(1 + x)
            - (1 - xinv) * jnp.log(1 + x**2)
        )

    @partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Quantity.from_(t, self.units["time"])

        # Compute parameters
        m = self.m(t, ustrip=self.units["mass"])
        r_s = self.r_s(t, ustrip=self.units["length"])

        r = jnp.linalg.vector_norm(xyz, axis=-1)
        return m / (jnp.pi * BURKERT_CONST) / ((r + r_s) * (r**2 + r_s**2))

    @partial(jax.jit)
    def _mass(self, q: gt.BBtQuSz3, /, t: gt.BtQuSz0 | gt.QuSz0) -> gt.BtFloatQuSz0:
        t = u.Quantity.from_(t, self.units["time"])
        x = jnp.linalg.vector_norm(q, axis=-1) / self.r_s(t)
        return (
            self.m(t)
            / BURKERT_CONST
            * (-2 * jnp.atan(x) + 2 * jnp.log(1 + x) + jnp.log(1 + x**2))
        )

    # -------------------------------------------------------------------

    def rho0(self, t: gt.BtQuSz0 | gt.QuSz0) -> gt.BtFloatQuSz0:
        r"""Central density of the potential.

        .. math::

            m0 = \pi \rho_0 r_s^3 (3 \log(2) - \pi / 2)
        """
        return self.m(t) / (jnp.pi * self.r_s(t) ** 3 * BURKERT_CONST)

    # -------------------------------------------------------------------
    # Constructors

    @classmethod
    def from_central_density(
        cls, rho_0: u.Quantity, r_s: u.Quantity, **kwargs: Any
    ) -> "BurkertPotential":
        r"""Create a Burkert potential from the central density.

        Parameters
        ----------
        rho_0 : :class:`~unxt.Quantity`[mass density]
            Central density.
        r_s : :class:`~unxt.Quantity`[length]
            Scale radius.

        Returns
        -------
        :class:`~galax.potential.BurkertPotential`
            Burkert potential.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.potential as gp

        >>> rho_0 = u.Quantity(1e6, "Msun / kpc3")
        >>> r_s = u.Quantity(1, "kpc")
        >>> pot = gp.BurkertPotential.from_central_density(rho_0, r_s, units="galactic")
        >>> pot
        BurkertPotential(
            units=LTMAUnitSystem( length=Unit("kpc"), ...),
            constants=ImmutableMap({'G': ...}),
            m=ConstantParameter( ... ),
            r_s=ConstantParameter( ... )
        )

        """
        m = jnp.pi * rho_0 * r_s**3 * BURKERT_CONST
        return cls(m=m, r_s=r_s, **kwargs)


# -------------------------------------------------------------------


@final
class HernquistPotential(AbstractSinglePotential):
    """Hernquist Potential."""

    m_tot: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Total mass of the potential."
    )

    r_s: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Scale radius."
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Quantity.from_(t, self.units["time"])

        # Compute parameters
        m_tot = self.m_tot(t, ustrip=self.units["mass"])
        r_s = self.r_s(t, ustrip=self.units["length"])

        r = jnp.linalg.vector_norm(xyz, axis=-1)
        return -self.constants["G"].value * m_tot / (r + r_s)

    @partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Quantity.from_(t, self.units["time"])

        # Compute parameters
        m_tot = self.m_tot(t, ustrip=self.units["mass"])
        r_s = self.r_s(t, ustrip=self.units["length"])

        s = jnp.linalg.vector_norm(xyz, axis=-1) / r_s
        rho0 = m_tot / (2 * jnp.pi * r_s**3)
        return rho0 / (s * (1 + s) ** 3)


# -------------------------------------------------------------------


@final
class IsochronePotential(AbstractSinglePotential):
    r"""Isochrone Potential.

    .. math::

        \Phi = -\frac{G M(t)}{r_s + \sqrt{r^2 + r_s^2}}
    """

    m_tot: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Total mass of the potential."
    )

    r_s: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length",
        doc=r"""Scale radius of the potential.

    The value of :math:`r_s` defines the transition between the inner, more
    harmonic oscillator-like behavior of the potential, and the outer, :math:`1
    / r` Keplerian falloff.
    """,
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit)
    def _potential(  # TODO: inputs w/ units
        self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /
    ) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Quantity.from_(t, self.units["time"])

        # Compute parameters
        m_tot = self.m_tot(t, ustrip=self.units["mass"])
        r_s = self.r_s(t, ustrip=self.units["length"])

        r = jnp.linalg.vector_norm(xyz, axis=-1)
        return -self.constants["G"].value * m_tot / (r_s + jnp.sqrt(r**2 + r_s**2))


# -------------------------------------------------------------------


@final
class JaffePotential(AbstractSinglePotential):
    """Jaffe Potential."""

    m: AbstractParameter = ParameterField(dimensions="mass", doc="Characteristic mass.")  # type: ignore[assignment]
    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale length.")  # type: ignore[assignment]

    @partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Quantity.from_(t, self.units["time"])

        # Compute parameters
        m = self.m(t, ustrip=self.units["mass"])
        r_s = self.r_s(t, ustrip=self.units["length"])

        r = jnp.linalg.vector_norm(xyz, axis=-1)
        return -self.constants["G"].value * m / r_s * jnp.log(1 + r_s / r)


# -------------------------------------------------------------------


@final
class KeplerPotential(AbstractSinglePotential):
    r"""The Kepler potential for a point mass.

    .. math::

        \Phi = -\frac{G M(t)}{r}
    """

    m_tot: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Total mass of the potential."
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit)
    def _potential(  # TODO: inputs w/ units
        self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /
    ) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Quantity.from_(t, self.units["time"])

        # Compute parameters
        m_tot = self.m_tot(t, ustrip=self.units["mass"])

        r = jnp.linalg.vector_norm(xyz, axis=-1)
        return -self.constants["G"].value * m_tot / r

    @partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Quantity.from_(t, self.units["time"])

        # Compute parameters
        m = self.m_tot(t, ustrip=self.units["mass"])

        r = jnp.linalg.vector_norm(xyz, axis=-1)
        pred = jnp.logical_or(  # are we at the origin with non-zero mass?
            jnp.greater(r, jnp.zeros_like(r)), jnp.equal(m, jnp.zeros_like(m))
        )
        return qlax.select(
            pred, jnp.zeros_like(r), jnp.full_like(r, fill_value=jnp.inf)
        )


# -------------------------------------------------------------------


@final
class PlummerPotential(AbstractSinglePotential):
    """Plummer Potential."""

    m_tot: AbstractParameter = ParameterField(dimensions="mass", doc="Total mass.")  # type: ignore[assignment]
    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale length.")  # type: ignore[assignment]

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        ul = self.units["length"]
        xyz = u.ustrip(AllowValue, ul, xyz)
        t = u.Quantity.from_(t, self.units["time"])

        # Compute parameters
        m_tot = self.m_tot(t, ustrip=self.units["mass"])
        r_s = self.r_s(t, ustrip=ul)

        r2 = jnp.linalg.vector_norm(xyz, axis=-1) ** 2
        return -self.constants["G"].value * m_tot / jnp.sqrt(r2 + r_s**2)


# -------------------------------------------------------------------


@partial(jax.jit)
def _safe_gamma_inc(a: u.Quantity, x: u.Quantity) -> u.Quantity:  # TODO: types
    return qsp.gammainc(a, x) * qsp.gamma(a)


@final
class PowerLawCutoffPotential(AbstractSinglePotential):
    r"""A spherical power-law density profile with an exponential cutoff.

    .. math::

        \rho(r) = \frac{G M}{2\pi \Gamma((3-\alpha)/2) r_c^3} \left(\frac{r_c}{r}\right)^\alpha \exp{-(r / r_c)^2}

    Parameters
    ----------
    m_tot : :class:`~unxt.Quantity`[mass]
        Total mass.
    alpha : :class:`~unxt.Quantity`[dimensionless]
        Power law index. Must satisfy: ``0 <= alpha < 3``.
    r_c : :class:`~unxt.Quantity`[length]
        Cutoff radius.
    """  # noqa: E501

    m_tot: AbstractParameter = ParameterField(dimensions="mass", doc="Total mass.")  # type: ignore[assignment]
    """Total mass of the potential."""

    alpha: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc="Power law index. Must satisfy: ``0 <= alpha < 3``",
    )

    r_c: AbstractParameter = ParameterField(dimensions="length", doc="Cutoff radius.")  # type: ignore[assignment]

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit)
    def _potential(self, xyz: gt.BBtQuSz3, t: gt.BBtQuSz0, /) -> gt.BtSz0:
        # Parse inputs
        ul = self.units["length"]
        xyz = u.ustrip(AllowValue, ul, xyz)
        t = u.Quantity.from_(t, self.units["time"])

        # Compute parameters
        m_tot = self.m_tot(t, ustrip=self.units["mass"])
        alpha = self.alpha(t, ustrip=self.units["dimensionless"])
        r_c = self.r_c(t, ustrip=ul)

        a = alpha / 2
        r = jnp.linalg.vector_norm(xyz, axis=-1)
        s2 = (r / r_c) ** 2
        GM = self.constants["G"].value * m_tot

        return GM * (
            (a - 1.5) * _safe_gamma_inc(1.5 - a, s2) / (r * qsp.gamma(2.5 - a))
            + _safe_gamma_inc(1 - a, s2) / (r_c * qsp.gamma(1.5 - a))
        )


# -------------------------------------------------------------------


class StoneOstriker15Potential(AbstractSinglePotential):
    r"""StoneOstriker15Potential(m, r_c, r_h, units=None, origin=None, R=None).

    Stone potential from `Stone & Ostriker (2015)
    <http://dx.doi.org/10.1088/2041-8205/806/2/L28>`_.

    .. math::

        \Phi = -\frac{2 G m}{\pi (r_h - r_c)} \left(
            \frac{r_h}{r} \tan^{-1}(\frac{r}{r_h})
            - \frac{r_c}{r} \tan^{-1}(\frac{r}{r_c})
            + \frac{1}{2} \log(\frac{r^2 + r_h^2}{r^2 + r_c^2})
            \right)

    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass", doc="Total mass")  # type: ignore[assignment]

    r_c: AbstractParameter = ParameterField(dimensions="length", doc="Core radius.")  # type: ignore[assignment]

    r_h: AbstractParameter = ParameterField(dimensions="length", doc="Halo radius.")  # type: ignore[assignment]

    # def __check_init__(self) -> None:
    #     _ = eqx.error_if(self.r_c, self.r_c.value >= self.r_h.value, "Core radius must be less than halo radius")   # noqa: E501, ERA001

    @partial(jax.jit)
    def _potential(self, xyz: gt.BBtQuSz3, t: gt.BBtQuSz0, /) -> gt.BtSz0:
        # Parse inputs
        ul = self.units["length"]
        xyz = u.ustrip(AllowValue, ul, xyz)
        t = u.Quantity.from_(t, self.units["time"])

        # Compute parameters
        m_tot = self.m_tot(t, ustrip=self.units["mass"])
        r_h = self.r_h(t, ustrip=ul)
        r_c = self.r_c(t, ustrip=ul)

        r = jnp.linalg.vector_norm(xyz, axis=-1)
        A = -2 * self.constants["G"].value * m_tot / (jnp.pi * (r_h - r_c))
        return A * (
            (r_h / r) * jnp.atan2(r, r_h)
            - (r_c / r) * jnp.atan2(r, r_c)
            + 0.5 * jnp.log((r**2 + r_h**2) / (r**2 + r_c**2))
        )


# -------------------------------------------------------------------


@final
class TriaxialHernquistPotential(AbstractSinglePotential):
    """Triaxial Hernquist Potential.

    Parameters
    ----------
    m_tot : :class:`~galax.potential.AbstractParameter`['mass']
        Mass parameter. This can be a
        :class:`~galax.potential.AbstractParameter` or an appropriate callable
        or constant, like a Quantity. See
        :class:`~galax.potential.ParameterField` for details.
    r_s : :class:`~galax.potential.AbstractParameter`['length']
        A scale length that determines the concentration of the system.  This
        can be a :class:`~galax.potential.AbstractParameter` or an appropriate
        callable or constant, like a Quantity. See
        :class:`~galax.potential.ParameterField` for details.
    q1 : :class:`~galax.potential.AbstractParameter`['length']
        Scale length in the y direction. This can be a
        :class:`~galax.potential.AbstractParameter` or an appropriate callable
        or constant, like a Quantity. See
        :class:`~galax.potential.ParameterField` for details.
    a2 : :class:`~galax.potential.AbstractParameter`['length']
        Scale length in the z direction. This can be a
        :class:`~galax.potential.AbstractParameter` or an appropriate callable
        or constant, like a Quantity. See
        :class:`~galax.potential.ParameterField` for details.

    units : :class:`~unxt.AbstractUnitSystem`, keyword-only
        The unit system to use for the potential.  This parameter accepts a
        :class:`~unxt.AbstractUnitSystem` or anything that can be converted to a
        :class:`~unxt.AbstractUnitSystem` using :func:`~unxt.unitsystem`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.TriaxialHernquistPotential(m_tot=1e12, r_s=8, q1=1, q2=0.5,
    ...                                     units="galactic")

    >>> q = u.Quantity([1, 0, 0], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.potential(q, t)
    Quantity[...](Array(-0.49983357, dtype=float64), unit='kpc2 / Myr2')
    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass", doc="Total mass.")  # type: ignore[assignment]

    r_s: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length",
        doc="Scale a scale length that determines the concentration of the system.",
    )

    # TODO: move to a triaxial wrapper
    q1: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.Quantity(1.0, ""),
        dimensions="dimensionless",
        doc="Scale length in the y direction divided by ``c``.",
    )

    q2: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.Quantity(1.0, ""),
        dimensions="dimensionless",
        doc="Scale length in the z direction divided by ``c``.",
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        converter=ImmutableMap, default=default_constants
    )

    @partial(jax.jit)
    def _potential(  # TODO: inputs w/ units
        self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /
    ) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Quantity.from_(t, self.units["time"])

        # Compute parameters
        u1 = self.units["dimensionless"]
        m_tot = self.m_tot(t, ustrip=self.units["mass"])
        r_s = self.r_s(t, ustrip=self.units["length"])
        q1, q2 = self.q1(t, ustrip=u1), self.q2(t, ustrip=u1)

        r_s = eqx.error_if(r_s, r_s <= 0, "r_s must be positive")
        rprime = jnp.sqrt(
            xyz[..., 0] ** 2 + (xyz[..., 1] / q1) ** 2 + (xyz[..., 2] / q2) ** 2
        )
        return -self.constants["G"].value * m_tot / (rprime + r_s)
