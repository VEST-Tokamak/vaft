"""What the per-product split actually protects (#527 step 7, #137).

The combined `mhd_linear` folded DCON, RDCON and STRIDE into one dense
`(time, n_tor)` grid, and it is tempting to say the three overwrote each other.
They did not, and stating it that way would leave the next reader unable to tell
which of the three pairs is the dangerous one.

Only `dcon-peeling` and `dcon-kink` destroy data: they are the same executable
run two ways, so both go through `_write_dcon_entry` and write the same fields
at the same position. That is the case #527 exists for, and the combined product
had nowhere to record which of the two it held.

These tests pin the distinction, so a future change that makes DCON and RDCON
start sharing a field -- or that makes the two edge treatments stop colliding --
is caught rather than assumed.
"""

import numpy as np
import pytest
from omas import ODS

from vaft.code.gpec._dcon_output import DconOutput
from vaft.machine_mapping.mhd_linear import (
    _write_dcon_entry,
    _write_resistive_entry,
    ensure_toroidal_mode_grid,
)


pytestmark = pytest.mark.core


GRID = [1, 2]


def _dcon(energy: float) -> DconOutput:
    """The minimum a DCON entry needs. No eigenfunction, so `m_pol_dominant` is
    absent -- which keeps these tests about which *fields* each writer owns."""
    return DconOutput(
        n_tor=1,
        mlow=-2,
        mhigh=2,
        mpert=5,
        mband=0,
        mode=np.array([1]),
        W_t_eigenvalue=np.array([complex(energy, 0.0)]),
    )


class _Resistive:
    """The subset of `Pest3MatchingOutput` the entry writer touches."""

    def __init__(self, solver: str) -> None:
        self.solver = solver
        self.n_tor = 1
        self.mlow, self.mhigh, self.mpert, self.mband, self.msing = -2, 2, 5, 0, 1


def _ods() -> ODS:
    ods = ODS(consistency_check=False)
    ensure_toroidal_mode_grid(ods, 0, GRID)
    return ods


def _entry(ods: ODS):
    return ods["mhd_linear"]["time_slice"][0]["toroidal_mode"][0]


def test_dcon_and_a_resistive_solver_co_populate_rather_than_collide():
    """Disjoint fields, so the combined product lost nothing between them."""
    ods = _ods()

    _write_dcon_entry(ods, 0, 0, _dcon(-0.3))
    _write_resistive_entry(ods, 0, 0, _Resistive("rdcon"), [])

    entry = _entry(ods)
    # DCON's number survived the resistive write.
    assert entry["energy_perturbed"] == pytest.approx(-0.3)
    # And the resistive write landed, in a field DCON never touches.
    assert entry["ballooning_type"]["name"] == "Tearing"


def test_two_resistive_solvers_write_the_same_value_to_the_one_field_they_share():
    """RDCON and STRIDE overwrite, but with an identical value.

    Their per-surface results go to `ntms`, which appends rather than
    overwrites, and each surface carries a `<solver name=...>` fragment -- so
    nothing is lost between them either.
    """
    ods = _ods()
    surfaces = [{"n": 1, "m": 2, "delta_prime_real": 1.5}]

    _write_resistive_entry(ods, 0, 0, _Resistive("rdcon"), surfaces)
    before = len(ods["ntms"]["time_slice"][0]["mode"])
    _write_resistive_entry(ods, 0, 0, _Resistive("stride"), surfaces)

    assert _entry(ods)["ballooning_type"]["name"] == "Tearing"
    assert len(ods["ntms"]["time_slice"][0]["mode"]) == before + len(surfaces), (
        "ntms surfaces must append; overwriting would lose one solver's result"
    )
    parameters = ods["ntms"]["code"]["parameters"]
    assert 'name="rdcon"' in parameters and 'name="stride"' in parameters


def test_the_two_dcon_edge_treatments_destroy_each_other():
    """The collision the per-product split exists for.

    Both are `_write_dcon_entry` at the same `(time_slice, position)`, writing
    the same fields. One silently replaces the other, and the combined product
    had nowhere to record which of the two it was holding -- which is why the
    edge treatment is identity rather than a parameter.
    """
    ods = _ods()

    _write_dcon_entry(ods, 0, 0, _dcon(-0.3))   # stands for dcon-peeling
    peeling = _entry(ods)["energy_perturbed"]
    _write_dcon_entry(ods, 0, 0, _dcon(+7.9))   # stands for dcon-kink

    kink = _entry(ods)["energy_perturbed"]
    assert peeling == pytest.approx(-0.3)
    assert kink == pytest.approx(7.9), "the second write replaced the first"
    # Nothing in the entry says which edge treatment produced it, which is the
    # whole reason the two need separate products.
    assert "edge" not in str(sorted(_entry(ods).keys())).lower()
