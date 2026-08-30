"""Unit tests for the pf_passive -> vessel conductor-region synthesis (OFT-free)."""

import numpy as np
import pytest
from omas import ODS

from vaft.code.tokamaker import TokaMakerConfig, geometry_signature, vessel_segments_from_ods
from vaft.code.tokamaker.vessel import _trace_strip


def _add_loop(ods, index, name, rc, zc, w, h, resistivity=None):
    base = f"pf_passive.loop.{index}"
    ods[f"{base}.name"] = name
    ods[f"{base}.element.0.geometry.outline.r"] = np.array([rc - w / 2, rc + w / 2, rc + w / 2, rc - w / 2])
    ods[f"{base}.element.0.geometry.outline.z"] = np.array([zc - h / 2, zc - h / 2, zc + h / 2, zc + h / 2])
    if resistivity is not None:
        ods[f"{base}.resistivity"] = resistivity


def _build_ods():
    """Two segments interleaved in index order + a W11-style sliver.

    WA: vertical column contiguous through Z=0 (like the outer cylinder).
    WB: mirrored top/bottom lid pair (like the W2 lids).
    W11: 0.1 mm sliver (excluded by default).
    """
    wa = [("WA", dict(rc=0.803, zc=float(zc), w=0.006, h=0.05, resistivity=7.8e-7))
          for zc in np.arange(-0.175, 0.176, 0.05)]      # 8 contiguous 50 mm loops
    wb = []
    for rc in (0.20, 0.25, 0.30):
        wb.append(("WB", dict(rc=rc, zc=+0.5, w=0.05, h=0.02, resistivity=7.8e-7)))
        wb.append(("WB", dict(rc=rc, zc=-0.5, w=0.05, h=0.02, resistivity=7.8e-7)))

    # interleave the segments so index-contiguity cannot be relied upon
    entries = []
    for pair in zip(wa, wb):
        entries.extend(pair)
    entries.extend(wa[len(wb):])
    entries.append(("W11", dict(rc=0.1, zc=0.0, w=0.002, h=0.05, resistivity=5.6e-8)))

    ods = ODS(consistency_check=False)
    for index, (name, params) in enumerate(entries):
        _add_loop(ods, index, name, **params)
    return ods


def test_segments_group_by_name_and_split_mirrored_pairs():
    regions = vessel_segments_from_ods(_build_ods(), TokaMakerConfig(include_vessel=True))

    assert "WA" in regions                       # contiguous through Z=0 -> single
    assert {"WB_U", "WB_L"} <= set(regions)      # mirrored -> split
    assert "W11" not in regions                  # excluded by default
    assert regions["WA"]["segment"] == "WA"
    assert regions["WB_U"]["n_loops"] >= 3
    contour = np.asarray(regions["WA"]["contour"])
    assert contour[:, 1].min() < 0 < contour[:, 1].max()


def test_w11_can_be_included_explicitly():
    config = TokaMakerConfig(include_vessel=True, exclude_vessel_segments=())
    regions = vessel_segments_from_ods(_build_ods(), config)
    assert "W11" in regions


def test_eta_precedence_region_over_segment_over_default():
    config = TokaMakerConfig(
        include_vessel=True,
        eta_vessel=7.8e-7,
        vessel_eta={"WB": 2.0e-6, "WB_U": 1.0e-6},
    )
    regions = vessel_segments_from_ods(_build_ods(), config)
    assert regions["WB_U"]["eta"] == pytest.approx(1.0e-6)   # region wins
    assert regions["WB_L"]["eta"] == pytest.approx(2.0e-6)   # segment fallback
    assert regions["WA"]["eta"] == pytest.approx(7.8e-7)     # default
    assert regions["WA"]["eta_ods_median"] == pytest.approx(7.8e-7)  # provenance only


def test_dx_clamped_between_floor_and_cap():
    config = TokaMakerConfig(include_vessel=True, dx_conductor=0.01, dx_conductor_min=0.004)
    regions = vessel_segments_from_ods(_build_ods(), config)
    # WA thickness ~6 mm (minus the gap shrink) -> between floor and cap
    assert 0.004 <= regions["WA"]["dx"] <= 0.006
    # WB thickness 20 mm -> capped at dx_conductor
    assert regions["WB_U"]["dx"] == pytest.approx(0.01)


def test_vertical_run_is_clamped_out_of_horizontal_bands():
    ods = _build_ods()
    # horizontal band crossing WA's top (r spans the WA column)
    index = len(ods["pf_passive.loop"])
    for rc in (0.75, 0.80, 0.85):
        _add_loop(ods, index, "WD", rc=rc, zc=0.19, w=0.05, h=0.03, resistivity=7.8e-7)
        index += 1
    regions = vessel_segments_from_ods(ods, TokaMakerConfig(include_vessel=True))

    wa = np.asarray(regions["WA"]["contour"])
    wd = np.asarray(regions["WD"]["contour"])
    assert wa[:, 1].max() < wd[:, 1].min()       # WA yielded to the WD band
    # exact disjointness is asserted inside vessel_segments_from_ods; reaching
    # here means the clamp resolved the overlap


def test_staircase_trace_for_stepped_strip():
    rects = [
        {"r_lo": 0.10, "r_hi": 0.12, "z_lo": 0.00, "z_hi": 0.05, "area": 0.001},
        {"r_lo": 0.11, "r_hi": 0.13, "z_lo": 0.05, "z_hi": 0.10, "area": 0.001},
        {"r_lo": 0.12, "r_hi": 0.14, "z_lo": 0.10, "z_hi": 0.15, "area": 0.001},
    ]
    contour = _trace_strip(rects, "CONE")
    assert len(contour) > 4                      # true staircase, not a bbox
    x, y = contour[:, 0], contour[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    assert area == pytest.approx(0.003, rel=1e-6)


def test_non_monotone_cluster_falls_back_to_bounding_box(caplog):
    # two side-by-side columns in one cluster: the strip trace double-counts,
    # the area check fails, and the bounding box is used with a warning
    rects = []
    for r_lo in (0.10, 0.30):
        for z_lo in (0.0, 0.05):
            rects.append({"r_lo": r_lo, "r_hi": r_lo + 0.02,
                          "z_lo": z_lo, "z_hi": z_lo + 0.05, "area": 0.001})
    with caplog.at_level("WARNING"):
        contour = _trace_strip(rects, "BAD")
    assert len(contour) == 4
    assert "bounding box" in caplog.text


def test_signature_stable_without_vessel_and_sensitive_with():
    from vaft.code.tokamaker import tokamaker_geometry_from_ods

    ods = _build_ods()
    ods["wall.description_2d.0.limiter.unit.0.outline.r"] = np.array([0.2, 0.6, 0.6, 0.2])
    ods["wall.description_2d.0.limiter.unit.0.outline.z"] = np.array([-0.4, -0.4, 0.4, 0.4])
    ods["pf_active.coil.0.name"] = "PF1"
    base = "pf_active.coil.0.element.0"
    ods[f"{base}.geometry.rectangle.r"] = 0.1
    ods[f"{base}.geometry.rectangle.z"] = 0.5
    ods[f"{base}.geometry.rectangle.width"] = 0.04
    ods[f"{base}.geometry.rectangle.height"] = 0.1
    ods[f"{base}.turns_with_sign"] = 8.0

    plain = TokaMakerConfig()
    with_vessel = TokaMakerConfig(include_vessel=True)
    geom_plain = tokamaker_geometry_from_ods(ods, plain)
    geom_vessel = tokamaker_geometry_from_ods(ods, with_vessel)

    assert "vessel" not in geom_plain            # v1-compatible geometry & hash
    assert "vessel" in geom_vessel
    assert geometry_signature(geom_plain, plain) != geometry_signature(geom_vessel, with_vessel)

    # eta participates in the mesh-cache key (save_gs_mesh bakes it in)
    other_eta = TokaMakerConfig(include_vessel=True, eta_vessel=1.0e-6)
    geom_other = tokamaker_geometry_from_ods(ods, other_eta)
    assert geometry_signature(geom_other, other_eta) != geometry_signature(geom_vessel, with_vessel)


def test_missing_pf_passive_is_actionable():
    with pytest.raises(ValueError, match="pf_passive"):
        vessel_segments_from_ods(ODS(consistency_check=False), TokaMakerConfig(include_vessel=True))
