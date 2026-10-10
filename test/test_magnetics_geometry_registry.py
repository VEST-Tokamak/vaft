"""Magnetics in the shared machine-geometry registry, and the flux-loop top view (#1829).

The audit behind these records (issue #1829):

* VEST probe and flux-loop R-Z are identical across every donor geometry
  version (``VEST_MagneticsGeometry_21193`` and ``_Full_ver_2302/2305/2306/2310/
  2409``); what changes by era is the DAQ wiring (field codes) and per-probe
  calibration, which the mapper applies per shot, not the positions.
* A flux loop is a toroidal loop: VEST stores its (R, Z) and no phi, and the
  registry keeps it as an axisymmetric ring, never a point at some angle.
* The equilibrium B-pol probes' phi is VAFT's family-to-port-clock convention
  (#718), not a surveyed angle; a probe without a stored phi stays unknown.
* IMPA is an insertable array: its R is fitted per shot and its phi is the
  insertion port's.

Coordinates are checked numerically against the geometry table and the
mapper conventions, not just that a renderer returns something.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from omas import ODS

import vaft
from vaft.machine_mapping.conventions import port_toroidal_angle
from vaft.machine_mapping.magnetics import (
    EQUILIBRIUM_PROBE_CLOCK,
    INBOARD_PROBE_MAX_R,
    OUTBOARD_PROBE_MIN_R,
    SIDE_PROBE_MIN_ABS_Z,
)
from vaft.plot.machine_geometry import (
    MACHINE_GEOMETRY_FAMILIES,
    MachineGeometry,
    machine_geometry_registry,
    project_machine_geometry,
)

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ROOT / "vaft" / "data" / "geometry" / "VEST_MagneticsGeometry_Full_ver_2302.yaml"
IMPA_SAMPLE = str(ROOT / "vaft" / "data" / "sample" / "legacy" / "shot_{shot}_impa.json.gz")
MAGNETICS = ("flux_loop", "b_field_pol_probe", "b_field_tor_probe")


@pytest.fixture(scope="module")
def sample():
    return vaft.omas.load(vaft.data.sample(39915, representation="omas"))


@pytest.fixture(scope="module")
def table():
    return yaml.safe_load(GEOMETRY.read_text(encoding="utf-8"))["channels"]


def _family_phi(r: float, z: float) -> float:
    family = "side" if abs(z) > SIDE_PROBE_MIN_ABS_Z else "inboard" if r < INBOARD_PROBE_MAX_R else "outboard"
    assert family == "side" or r < INBOARD_PROBE_MAX_R or r > OUTBOARD_PROBE_MIN_R
    return port_toroidal_angle(EQUILIBRIUM_PROBE_CLOCK[family])


def test_magnetics_families_are_registered():
    assert set(MAGNETICS) <= set(MACHINE_GEOMETRY_FAMILIES)


def test_flux_loops_are_axisymmetric_rings_at_the_table_coordinates(sample, table):
    records = machine_geometry_registry(sample, families=("flux_loop",))
    expected = [(c["r"], c["z"]) for c in table if c["kind"] == "flux_loop"]
    assert len(records) == len(expected) == 11
    assert {r.family for r in records} == {"flux_loop"} and {r.semantic for r in records} == {"toroidal_ring"}
    np.testing.assert_allclose([(float(r.r[0]), float(r.z[0])) for r in records], expected, rtol=0, atol=1e-12)
    assert all(r.phi is None for r in records)
    assert {json.loads(r.provenance_json)["phi_source"] for r in records} == {"axisymmetric loop (no toroidal angle)"}


def test_bpol_probes_keep_table_rz_and_the_family_clock_phi(sample, table):
    records = machine_geometry_registry(sample, families=("b_field_pol_probe",))
    expected = [(c["r"], c["z"]) for c in table if c["kind"] == "b_field_pol_probe"]
    assert len(records) == len(expected) == 64
    np.testing.assert_allclose([(float(r.r[0]), float(r.z[0])) for r in records], expected, rtol=0, atol=1e-12)
    for record in records:
        assert record.semantic == "point"
        assert float(record.phi[0]) == pytest.approx(_family_phi(float(record.r[0]), float(record.z[0])), abs=1e-12)
        assert json.loads(record.provenance_json)["phi_source"] == "stored position.phi"


def test_a_probe_without_phi_stays_unknown_and_is_not_placed_in_the_top_view():
    ods = ODS(consistency_check=False)
    ods["magnetics.b_field_pol_probe.0.name"] = "no-phi probe"
    ods["magnetics.b_field_pol_probe.0.position.r"] = 0.796
    ods["magnetics.b_field_pol_probe.0.position.z"] = 0.10
    (record,) = machine_geometry_registry(ods, families=("b_field_pol_probe",))
    assert record.phi is None
    assert json.loads(record.provenance_json)["phi_source"] == "unknown (position.phi absent)"
    assert project_machine_geometry(record, "top") is None
    assert project_machine_geometry(record, "rz").kind == "points"


def test_a_toroidal_ring_refuses_a_phi_and_projects_as_a_whole_loop():
    with pytest.raises(ValueError, match="axisymmetric"):
        MachineGeometry("flux_loop", "toroidal_ring", [0.5], [0.1], phi=[0.0])
    ring = MachineGeometry("flux_loop", "toroidal_ring", [0.592], [0.685], label="FL")
    rz = project_machine_geometry(ring, "rz")
    assert rz.kind == "points" and float(rz.r[0]) == 0.592 and float(rz.z[0]) == 0.685
    top = project_machine_geometry(ring, "top")
    np.testing.assert_allclose(np.hypot(top.r, top.z), 0.592, atol=1e-12)
    assert top.kind == "polyline" and np.ptp(np.arctan2(top.z, top.r)) > 6.0  # the whole circle
    three = project_machine_geometry(ring, "3d")
    np.testing.assert_allclose(three.z, 0.685)


def test_family_selection_returns_only_the_requested_family(sample):
    for family in ("flux_loop", "b_field_pol_probe"):
        assert {r.family for r in machine_geometry_registry(sample, families=(family,))} == {family}


def test_the_top_view_now_draws_the_flux_loops(sample, table):
    """The bug: rings required position.0.phi, which the VEST mapper never stores."""
    from vaft.plot.backend.recipes import _diagnostic_layers_3d, _topview_diagnostic_layers, _topview_reads

    rings = [layer for layer in _topview_diagnostic_layers(sample)
             if layer.kind == "polyline" and layer.style.get("linestyle") == ":"]
    radii = sorted(float(np.median(np.hypot(layer.r, layer.z))) for layer in rings)
    expected = sorted(c["r"] for c in table if c["kind"] == "flux_loop")
    np.testing.assert_allclose(radii, expected, atol=1e-9)
    assert sum(1 for layer in _diagnostic_layers_3d(sample) if "magnetics.flux_loop" in (layer.group or "")) == 11
    reads = _topview_reads()
    assert "magnetics.flux_loop.{i}.position.{j}.r" in reads
    assert "magnetics.flux_loop.{i}.position.{j}.phi" not in reads


@pytest.mark.skipif(not Path(IMPA_SAMPLE.format(shot=35376)).is_file(), reason="archived IMPA sample not available")
def test_impa_probes_carry_the_port_phi_and_the_per_shot_radius():
    from vaft.machine_mapping.impa import IMPA_PORT, impa, impa_probe_indices
    from vaft.machine_mapping.registry import port_phi

    ods = ODS(consistency_check=False)
    status = impa(ods, 35376, raw_source=IMPA_SAMPLE)
    node = status["ids_node"]
    assert node == "magnetics.b_field_tor_probe"
    indices = impa_probe_indices(ods, node)
    records = machine_geometry_registry(ods, families=("b_field_tor_probe",))
    assert len(records) == len(indices) > 0
    for record, index in zip(records, indices):
        assert float(record.r[0]) == pytest.approx(float(ods[f"{node}.{index}.position.r"]))
        assert float(record.phi[0]) == pytest.approx(float(port_phi(IMPA_PORT)))
        assert record.semantic == "point"
