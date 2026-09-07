"""The canonical-to-EFUND projection (issue #194).

Invariant under test: canonical static geometry -> EFUND input, in the
em_coupling order, with the F-coil groups the k-file already selects.
"""

from __future__ import annotations

import math

import f90nml
import numpy as np
import pytest

from vaft.code.efit.kfile import efit16_group_indices
from vaft.code.efit.legacy import vfit_pf_active_efit26
from vaft.data.resources import data_path
from vaft.machine_mapping.efund_geometry import (
    EFIT16_GROUP_NAMES,
    efund_geometry_from_static,
    efund_probe_angle_deg,
    equilibrium_probe_count,
    rectangle_from_outline,
)

LEGACY_ERA = "vest-pre-43017-pf1906"


@pytest.fixture(scope="module")
def static():
    from vaft.omas.vest_upstream import build_static_ods

    return build_static_ods(LEGACY_ERA)


@pytest.fixture(scope="module")
def geometry(static):
    ods, manifest = static
    return efund_geometry_from_static(ods, manifest=manifest)


@pytest.fixture(scope="module")
def packaged_mhdin():
    return f90nml.read(str(data_path("efit/mhdin.dat")))


def _legacy_26_names() -> list[str]:
    # The name list is a local of the legacy mapper; read it off the source
    # rather than duplicating it, so the two cannot drift apart silently.
    import inspect

    source = inspect.getsource(vfit_pf_active_efit26)
    start = source.index("PFname = [")
    end = source.index("]", start)
    return [item.strip().strip('"') for item in source[start + len("PFname = [") : end].split(",") if item.strip()]


def test_counts_are_the_legacy_table_counts_with_the_canonical_pf1(geometry):
    assert geometry.counts() == {
        "nfcoil": 302,
        "nfsum": 16,
        "nsilop": 11,
        "magpri": 64,
        "necoil": 0,
        "nesum": 0,
        "nvesel": 950,
        "nvsum": 950,
        "nacoil": 0,
    }


def test_fcoil_groups_are_the_k_file_selection(geometry):
    names = _legacy_26_names()
    selected = [names[index] for index in efit16_group_indices(26, 16)]
    assert selected == list(EFIT16_GROUP_NAMES)
    assert geometry.group_names == EFIT16_GROUP_NAMES
    assert set(np.unique(geometry.fcoil_group)) == set(range(1, 17))
    # Elements are grouped contiguously in group order.
    assert np.all(np.diff(geometry.fcoil_group) >= 0)
    assert np.all(geometry.group_turns == 1.0)


def test_pf1_segments_carry_all_632_turns(geometry):
    rows = geometry.group_summary()
    pf1 = [row for row in rows if row["name"].startswith("PF1-")]
    assert len(pf1) == 8
    assert sum(row["turns"] for row in pf1) == 632.0
    # 158 elements of 4 turns cannot split 8 ways as exactly 79 each; the
    # legacy single-rectangle table rounds to 79.  Each segment stays within
    # one element of that.
    assert all(abs(row["turns"] - 79.0) <= 4.0 for row in pf1)
    upper = pf1[:4]
    lower = pf1[4:]
    assert all(row["z_min"] > 0 for row in upper)
    assert all(row["z_max"] < 0 for row in lower)
    assert [row["z"] < nxt["z"] for row, nxt in zip(upper, upper[1:])] == [True] * 3
    assert [row["z"] > nxt["z"] for row, nxt in zip(lower, lower[1:])] == [True] * 3


@pytest.mark.parametrize(("coil", "turns"), [("PF5", 12.0), ("PF6", 12.0), ("PF9", 24.0), ("PF10", 24.0)])
def test_outer_coils_split_upper_and_lower(geometry, coil, turns):
    rows = {row["name"]: row for row in geometry.group_summary()}
    assert rows[f"{coil}U"]["turns"] == turns
    assert rows[f"{coil}L"]["turns"] == turns
    assert rows[f"{coil}U"]["z"] == pytest.approx(-rows[f"{coil}L"]["z"])
    assert rows[f"{coil}U"]["z_min"] > 0 > rows[f"{coil}L"]["z_max"]


def test_vessel_is_pf_passive_in_em_coupling_order(static, geometry):
    ods, _ = static
    count = len(ods["pf_passive.loop"])
    assert geometry.nvesel == count == len(ods["em_coupling.passive_loops"])
    for index in (0, 239, 240, 500, 949):
        outline = ods[f"pf_passive.loop.{index}.element.0.geometry.outline"]
        rc, zc, width, height = rectangle_from_outline(outline["r"], outline["z"])
        assert geometry.vessel_r[index] == rc
        assert geometry.vessel_z[index] == zc
        assert geometry.vessel_w[index] == width
        assert geometry.vessel_h[index] == height
        assert geometry.vessel_resistance[index] == float(ods[f"pf_passive.loop.{index}.resistance"])
    assert list(geometry.vessel_group) == list(range(1, count + 1))
    assert geometry.vessel_names[0] == str(ods["pf_passive.loop.0.name"])


def test_vessel_order_must_match_the_coupling_asset(static):
    import copy

    ods, _ = static
    broken = copy.deepcopy(ods)
    uris = list(broken["em_coupling.passive_loops"])
    uris[0], uris[1] = uris[1], uris[0]
    broken["em_coupling.passive_loops"] = np.asarray(uris)
    with pytest.raises(ValueError, match="em_coupling.passive_loops"):
        efund_geometry_from_static(broken)


def test_flux_loops_and_probes_follow_ods_order(static, geometry):
    ods, _ = static
    for index in range(geometry.nsilop):
        assert geometry.loop_r[index] == float(ods[f"magnetics.flux_loop.{index}.position.0.r"])
        assert geometry.loop_z[index] == float(ods[f"magnetics.flux_loop.{index}.position.0.z"])
    assert geometry.magpri == equilibrium_probe_count(ods) == 64
    assert len(ods["magnetics.b_field_pol_probe"]) > geometry.magpri
    for index in range(geometry.magpri):
        assert geometry.probe_r[index] == float(ods[f"magnetics.b_field_pol_probe.{index}.position.r"])
        assert geometry.probe_z[index] == float(ods[f"magnetics.b_field_pol_probe.{index}.position.z"])
    assert np.all(geometry.probe_angle_deg == 90.0)
    assert np.all(geometry.probe_length == 0.01)


def test_machine_block_records_the_era_and_input_hashes(geometry):
    assert geometry.machine["era"] == LEGACY_ERA
    assert geometry.machine["pf_geometry"] == "1906"
    assert "static_geometry" in geometry.machine["static_inputs"]
    assert "coupling" in geometry.machine["static_inputs"]
    assert "sha256" in geometry.machine["static_inputs"]["coupling"]


@pytest.mark.parametrize(
    ("theta", "expected"),
    [
        (3 * math.pi / 2, 90.0),
        (0.0, 0.0),
        (math.pi / 2, 270.0),
        (math.pi, 180.0),
        (2 * math.pi, 0.0),
        (-math.pi / 2, 90.0),
    ],
)
def test_probe_angle_is_minus_theta_reduced_to_a_turn(theta, expected):
    assert efund_probe_angle_deg(theta) == pytest.approx(expected)


def test_probe_angle_reproduces_the_dd_axis():
    # EFUND direction (cos amp2, sin amp2) must equal the DD axis (cos theta, -sin theta).
    for theta in (0.3, 1.9, 3 * math.pi / 2, 5.0):
        amp2 = math.radians(efund_probe_angle_deg(theta))
        assert math.cos(amp2) == pytest.approx(math.cos(theta))
        assert math.sin(amp2) == pytest.approx(-math.sin(theta))


def test_rectangle_from_outline_accepts_any_corner_order_and_refuses_the_rest():
    assert rectangle_from_outline([0.8, 0.806, 0.806, 0.8], [-0.6, -0.6, -0.595, -0.595]) == pytest.approx(
        (0.803, -0.5975, 0.006, 0.005)
    )
    assert rectangle_from_outline([0.806, 0.8, 0.8, 0.806], [-0.595, -0.595, -0.6, -0.6]) == pytest.approx(
        (0.803, -0.5975, 0.006, 0.005)
    )
    with pytest.raises(ValueError, match="axis-aligned"):
        rectangle_from_outline([0.0, 1.0, 1.5, 0.5], [0.0, 0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="4-point"):
        rectangle_from_outline([0.0, 1.0, 1.0], [0.0, 0.0, 1.0])


# --- compatibility with the bundled legacy table --------------------------
#
# These fail if either side drifts: the canonical assets or the packaged
# mhdin.dat.  Agreement is expected on everything but the PF discretization
# (single rectangles per group there, canonical elements here), the spelling
# of the probe angle, and rsisvs (resistivity there, resistance here).


def test_packaged_vessel_loops_and_probes_match_the_projection(geometry, packaged_mhdin):
    in3 = packaged_mhdin["in3"]
    machine = packaged_mhdin["machinein"]
    assert (machine["nvesel"], machine["nsilop"], machine["magpri"], machine["nfsum"]) == (950, 11, 64, 16)
    np.testing.assert_allclose(in3["rvs"], geometry.vessel_r, atol=1e-6, rtol=0)
    np.testing.assert_allclose(in3["zvs"], geometry.vessel_z, atol=1e-6, rtol=0)
    np.testing.assert_allclose(in3["wvs"], geometry.vessel_w, atol=1e-6, rtol=0)
    np.testing.assert_allclose(in3["hvs"], geometry.vessel_h, atol=1e-6, rtol=0)
    assert list(in3["vsid"]) == list(geometry.vessel_group)
    np.testing.assert_allclose(in3["rsi"], geometry.loop_r, atol=1e-6, rtol=0)
    np.testing.assert_allclose(in3["zsi"], geometry.loop_z, atol=1e-6, rtol=0)
    np.testing.assert_allclose(in3["xmp2"], geometry.probe_r, atol=1e-6, rtol=0)
    np.testing.assert_allclose(in3["ymp2"], geometry.probe_z, atol=1e-6, rtol=0)
    np.testing.assert_allclose(in3["smp2"], geometry.probe_length, atol=1e-9, rtol=0)
    packaged_angle = np.mod(np.asarray(in3["amp2"], dtype=float), 360.0)
    np.testing.assert_allclose(packaged_angle, geometry.probe_angle_deg, atol=1e-9, rtol=0)
    # Known, documented disagreement: the packaged rsisvs is a resistivity.
    assert not np.allclose(in3["rsisvs"], geometry.vessel_resistance)


def test_packaged_pf_groups_agree_on_centroids_and_turns_but_not_discretization(geometry, packaged_mhdin):
    in3 = packaged_mhdin["in3"]
    assert packaged_mhdin["machinein"]["nfcoil"] == 16 != geometry.nfcoil
    rows = geometry.group_summary()
    for index, row in enumerate(rows):
        assert row["r"] == pytest.approx(in3["rf"][index], abs=0.02), row["name"]
        assert row["z"] == pytest.approx(in3["zf"][index], abs=0.02), row["name"]
        assert row["turns"] == pytest.approx(in3["fcturn"][index], abs=4.0), row["name"]
    assert list(in3["fcid"]) == list(range(1, 17))
    assert list(in3["turnfc"]) == [1.0] * 16
