"""Resistance conventions for pf_active and pf_passive (issue #117).

`pf_active.coil[:].resistance` is the coil's *terminal* resistance, per the DD's
"Coil resistance". For N turns in series, each occupying `A_pack / N` of the
winding pack:

    R = N * rho * (2*pi*r) / (A_pack / N) = N^2 * rho * 2*pi*r / A_pack

Before #117 the N^2 factor was missing, which is the resistance of one turn
filling the entire pack -- 1.4e-7 Ohm for PF1, five orders of magnitude below
anything physical.

`pf_passive.loop[:].resistance` is deliberately NOT touched here: for the
inboard and outboard wall regions it holds experimentally fitted effective
resistances, not a geometric formula, and it feeds the eddy-current solve.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from omas import ODS

from vaft.machine_mapping.pf_active import (
    COPPER_RESISTIVITY,
    PF_COIL_COUNT,
    PF_RADIUS_BY_COIL,
    PF_WIDTH_BY_COIL,
    _geometry_profile_for_shot,
    vfit_pf_active_static,
)
from vaft.machine_mapping.pf_passive import pf_passive

SHOT = 41672


@pytest.fixture(scope="module")
def pf_active_ods():
    ods = ODS(consistency_check=False)
    vfit_pf_active_static(ods, SHOT)
    return ods


def _turns(ods, coil_index: int) -> float:
    elements = ods[f"pf_active.coil.{coil_index}.element"]
    return float(
        sum(abs(float(elements[j]["turns_with_sign"])) for j in range(len(elements)))
    )


def test_resistance_is_the_series_turn_formula_over_the_mapped_turns(pf_active_ods):
    """Ties the stored resistance to the stored turns so the two cannot drift."""
    _, height_by_coil = _geometry_profile_for_shot(SHOT)
    for coil_index in range(PF_COIL_COUNT):
        turns = _turns(pf_active_ods, coil_index)
        area = PF_WIDTH_BY_COIL[coil_index] * height_by_coil[coil_index]
        expected = (
            turns**2
            * 2.0
            * math.pi
            * COPPER_RESISTIVITY
            * PF_RADIUS_BY_COIL[coil_index]
            / area
        )
        stored = float(pf_active_ods[f"pf_active.coil.{coil_index}.resistance"])
        assert stored == pytest.approx(expected, rel=1e-12), coil_index


def test_terminal_resistances_are_physically_plausible(pf_active_ods):
    """A copper PF coil is milliohms, not microohms.

    The pre-#117 value for PF1 was 1.4e-7 Ohm -- less than a short length of
    busbar for a 632-turn coil, which is what made the missing factor visible.
    """
    for coil_index in range(PF_COIL_COUNT):
        stored = float(pf_active_ods[f"pf_active.coil.{coil_index}.resistance"])
        assert 1e-3 < stored < 1.0, coil_index


def test_multi_turn_coils_are_the_ones_that_moved(pf_active_ods):
    """PF1/PF2 bundle several turns per element; PF3-10 carry one each."""
    assert _turns(pf_active_ods, 0) == pytest.approx(632.0)
    assert _turns(pf_active_ods, 1) == pytest.approx(500.0)
    for coil_index in range(2, PF_COIL_COUNT):
        elements = pf_active_ods[f"pf_active.coil.{coil_index}.element"]
        # One turn per element, so the turn count is the element count.
        assert _turns(pf_active_ods, coil_index) == pytest.approx(float(len(elements)))


def test_pf_passive_resistance_is_left_alone():
    """#117 changes nothing here: these values are fitted, and they are live.

    The inboard and outboard wall regions carry effective resistances fitted
    against magnetics in a plasma-free window, not `rho * 2*pi*R / A`, and
    `vaft/omas/process_wrapper.py` feeds them straight into the eddy solve.
    Nine of eleven regions do match the geometric formula exactly; this pins
    that split so a future "cleanup" cannot quietly regularise the two that
    do not.
    """
    ods = ODS(consistency_check=False)
    pf_passive(ods)
    loops = ods["pf_passive.loop"]
    assert len(loops) == 950

    ratios: dict[str, set[float]] = {}
    for index in range(len(loops)):
        loop = ods[f"pf_passive.loop.{index}"]
        element = loop["element"][0]
        radius = float(
            np.mean(np.asarray(element["geometry"]["outline"]["r"], dtype=float))
        )
        predicted = (
            float(loop["resistivity"]) * 2.0 * math.pi * radius / float(element["area"])
        )
        ratios.setdefault(str(loop["name"]), set()).add(
            round(float(loop["resistance"]) / predicted, 6)
        )

    fitted = {name for name, values in ratios.items() if values != {1.0}}
    assert fitted == {"W1", "W11"}
    # 12 outboard and 19 inboard factors, matching the legacy
    # `Wall_Factor_Outboard` / `Wall_Factor_Inboard` arrays.
    assert len(ratios["W1"]) == 12
    assert len(ratios["W11"]) == 19


def test_geometry_fingerprint_ignores_the_derived_resistance():
    """The coupling guard must not reject a valid ODS over a formula change.

    `em_coupling` fingerprints each coil to check that a stored ODS matches the
    geometry version its coupling matrices were built for. Resistance is a
    derived scalar, not geometry -- a pure function of the width/radius
    constants, the era height profile and `turns_with_sign`, all already
    covered -- so including it gave the guard no discriminating power while
    making every packaged ODS fail the moment the #117 formula changed.
    """
    from vaft.machine_mapping.em_coupling import _static_signature

    baseline = ODS(consistency_check=False)
    vfit_pf_active_static(baseline, SHOT)
    altered = ODS(consistency_check=False)
    vfit_pf_active_static(altered, SHOT)
    altered["pf_active.coil.0.resistance"] = (
        float(baseline["pf_active.coil.0.resistance"]) * 1e5
    )

    assert _static_signature(altered["pf_active.coil.0"]) == _static_signature(
        baseline["pf_active.coil.0"]
    )

    # Geometry itself must still be discriminating.
    moved = ODS(consistency_check=False)
    vfit_pf_active_static(moved, SHOT)
    moved["pf_active.coil.0.element.0.geometry.rectangle.r"] = 99.0
    assert _static_signature(moved["pf_active.coil.0"]) != _static_signature(
        baseline["pf_active.coil.0"]
    )
