"""Unit tests for the ODS -> TokaMaker geometry builder (OFT-free)."""

import numpy as np
import pytest
from omas import ODS

from vaft.code.tokamaker import (
    TokaMakerConfig,
    geometry_signature,
    tokamaker_geometry_from_ods,
)


def _add_rectangle_element(ods, coil, elem, r, z, w, h, turns):
    base = f"pf_active.coil.{coil}.element.{elem}"
    ods[f"{base}.geometry.rectangle.r"] = r
    ods[f"{base}.geometry.rectangle.z"] = z
    ods[f"{base}.geometry.rectangle.width"] = w
    ods[f"{base}.geometry.rectangle.height"] = h
    ods[f"{base}.turns_with_sign"] = turns


def _build_ods():
    ods = ODS(consistency_check=False)
    ods["wall.description_2d.0.limiter.unit.0.outline.r"] = np.array([0.2, 0.6, 0.6, 0.2])
    ods["wall.description_2d.0.limiter.unit.0.outline.z"] = np.array([-0.4, -0.4, 0.4, 0.4])
    # PF1: up/down-mirrored pair of elements -> two rectangles, one coil set
    ods["pf_active.coil.0.name"] = "PF1"
    _add_rectangle_element(ods, 0, 0, r=0.10, z=+0.50, w=0.04, h=0.10, turns=+8.0)
    _add_rectangle_element(ods, 0, 1, r=0.10, z=-0.50, w=0.04, h=0.10, turns=+8.0)
    # PF2: single-sided coil with anti-series windings -> one rectangle, net turns
    ods["pf_active.coil.1.name"] = "PF2"
    _add_rectangle_element(ods, 1, 0, r=0.70, z=+0.30, w=0.06, h=0.06, turns=+5.0)
    _add_rectangle_element(ods, 1, 1, r=0.70, z=+0.42, w=0.06, h=0.06, turns=-2.0)
    return ods


def test_mirrored_coil_splits_into_upper_and_lower_rectangles():
    geometry = tokamaker_geometry_from_ods(_build_ods(), TokaMakerConfig())

    assert set(geometry["coils"]) == {"PF1_U", "PF1_L", "PF2"}
    upper = geometry["coils"]["PF1_U"]
    lower = geometry["coils"]["PF1_L"]
    assert upper["coil_set"] == lower["coil_set"] == "PF1"
    assert upper["zc"] == pytest.approx(+0.50)
    assert lower["zc"] == pytest.approx(-0.50)
    assert upper["rc"] == pytest.approx(0.10)
    assert upper["w"] == pytest.approx(0.04)
    assert upper["h"] == pytest.approx(0.10)
    assert upper["nturns"] == pytest.approx(8.0)


def test_single_sided_coil_keeps_signed_turn_sum_and_bounding_box():
    geometry = tokamaker_geometry_from_ods(_build_ods(), TokaMakerConfig())

    pf2 = geometry["coils"]["PF2"]
    assert pf2["coil_set"] == "PF2"
    assert pf2["nturns"] == pytest.approx(3.0)  # +5 - 2 anti-series
    # bounding box spans both stacked elements
    assert pf2["zc"] == pytest.approx(0.36)
    assert pf2["h"] == pytest.approx(0.18)


def test_limiter_override_and_exclude_coils():
    config = TokaMakerConfig(
        limiter=([0.25, 0.55, 0.55, 0.25], [-0.3, -0.3, 0.3, 0.3]),
        exclude_coils=("PF2",),
    )
    geometry = tokamaker_geometry_from_ods(_build_ods(), config)

    assert geometry["limiter"] == [[0.25, -0.3], [0.55, -0.3], [0.55, 0.3], [0.25, 0.3]]
    assert set(geometry["coils"]) == {"PF1_U", "PF1_L"}


def test_zero_net_turns_is_rejected():
    ods = _build_ods()
    ods["pf_active.coil.1.element.1.turns_with_sign"] = -5.0  # PF2 nets to zero
    with pytest.raises(ValueError, match="zero net turns_with_sign"):
        tokamaker_geometry_from_ods(ods, TokaMakerConfig())


def test_limiter_with_nonpositive_radius_is_rejected():
    ods = _build_ods()
    ods["wall.description_2d.0.limiter.unit.0.outline.r"] = np.array([0.0, 0.6, 0.6, 0.2])
    with pytest.raises(ValueError, match="R > 0"):
        tokamaker_geometry_from_ods(ods, TokaMakerConfig())


def test_geometry_signature_is_stable_and_sensitive():
    ods = _build_ods()
    config = TokaMakerConfig()
    geometry = tokamaker_geometry_from_ods(ods, config)

    # stable across recomputation and dict insertion order
    again = tokamaker_geometry_from_ods(ods, config)
    reordered = {"coils": dict(reversed(list(again["coils"].items()))), "limiter": again["limiter"]}
    signature = geometry_signature(geometry, config)
    assert geometry_signature(reordered, config) == signature

    # sensitive to mesh resolution and to geometry changes
    assert geometry_signature(geometry, TokaMakerConfig(dx_plasma=0.02)) != signature
    moved = tokamaker_geometry_from_ods(ods, TokaMakerConfig(exclude_coils=("PF2",)))
    assert geometry_signature(moved, config) != signature


def test_vsc_coil_splits_into_independent_coil_sets():
    config = TokaMakerConfig(vsc_coil="PF1")
    geometry = tokamaker_geometry_from_ods(_build_ods(), config)

    assert geometry["coils"]["PF1_U"]["coil_set"] == "PF1_U"
    assert geometry["coils"]["PF1_L"]["coil_set"] == "PF1_L"
    # other coils keep their shared set
    assert geometry["coils"]["PF2"]["coil_set"] == "PF2"


def test_vsc_coil_must_be_mirrored():
    with pytest.raises(ValueError, match="up/down-mirrored"):
        tokamaker_geometry_from_ods(_build_ods(), TokaMakerConfig(vsc_coil="PF2"))


def test_vsc_coil_matching_no_coil_is_rejected_early():
    with pytest.raises(ValueError, match="matches no pf_active coil"):
        tokamaker_geometry_from_ods(_build_ods(), TokaMakerConfig(vsc_coil="PF99"))
