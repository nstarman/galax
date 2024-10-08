from typing import Any

import pytest
from plum import convert

import quaxed.numpy as jnp
from unxt import Quantity

import galax.potential as gp
from ...test_core import AbstractPotential_Test
from ..test_common import ParameterMTotMixin
from galax.potential import AbstractPotentialBase, KeplerPotential
from galax.typing import QVec3


class TestKeplerPotential(
    AbstractPotential_Test,
    # Parameters
    ParameterMTotMixin,
):
    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.KeplerPotential]:
        return gp.KeplerPotential

    @pytest.fixture(scope="class")
    def fields_(self, field_m_tot, field_units) -> dict[str, Any]:
        return {"m_tot": field_m_tot, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: KeplerPotential, x: QVec3) -> None:
        expect = Quantity(-1.20227527, pot.units["specific energy"])
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: KeplerPotential, x: QVec3) -> None:
        expect = Quantity(
            [0.08587681, 0.17175361, 0.25763042], pot.units["acceleration"]
        )
        got = convert(pot.gradient(x, t=0), Quantity)
        assert jnp.allclose(got, expect, atol=Quantity(1e-8, expect.unit))

    def test_density(self, pot: KeplerPotential, x: QVec3) -> None:
        expect = Quantity(0.0, pot.units["mass density"])
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: KeplerPotential, x: QVec3) -> None:
        expect = Quantity(
            [
                [0.06747463, -0.03680435, -0.05520652],
                [-0.03680435, 0.01226812, -0.11041304],
                [-0.05520652, -0.11041304, -0.07974275],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.hessian(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractPotentialBase, x: QVec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = Quantity(
            [
                [0.06747463, -0.03680435, -0.05520652],
                [-0.03680435, 0.01226812, -0.11041304],
                [-0.05520652, -0.11041304, -0.07974275],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )
