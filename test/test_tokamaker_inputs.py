"""Unit tests for TokaMaker input preparation from an ODS (OFT-free)."""

import sys

import numpy as np
import pytest
from omas import ODS

from vaft.code.tokamaker import (
    TokaMakerConfig,
    prepare_tokamaker_inputs,
    resolve_mesh_file,
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
    ods["dataset_description.data_entry.pulse"] = 12345
    ods["wall.description_2d.0.limiter.unit.0.outline.r"] = np.array([0.2, 0.6, 0.6, 0.2])
    ods["wall.description_2d.0.limiter.unit.0.outline.z"] = np.array([-0.4, -0.4, 0.4, 0.4])
    ods["pf_active.coil.0.name"] = "PF1"
    _add_rectangle_element(ods, 0, 0, r=0.10, z=+0.50, w=0.04, h=0.10, turns=+8.0)
    _add_rectangle_element(ods, 0, 1, r=0.10, z=-0.50, w=0.04, h=0.10, turns=+8.0)
    ods["pf_active.coil.1.name"] = "PF2"
    _add_rectangle_element(ods, 1, 0, r=0.70, z=+0.30, w=0.06, h=0.06, turns=+5.0)
    ods["pf_active.time"] = np.array([0.0, 1.0])
    ods["pf_active.coil.0.current.data"] = np.array([0.0, -2000.0])
    ods["pf_active.coil.1.current.data"] = np.array([0.0, 1000.0])
    ods["magnetics.ip.0.time"] = np.array([0.0, 1.0])
    ods["magnetics.ip.0.data"] = np.array([0.0, 100.0e3])
    ods["tf.time"] = np.array([0.0, 1.0])
    ods["tf.b_field_tor_vacuum_r.data"] = np.array([0.06, 0.06])
    ods["tf.r0"] = 0.4
    ods["equilibrium.time"] = np.array([0.30, 0.50])
    ods["equilibrium.time_slice.0.global_quantities.ip"] = 51.0e3
    ods["equilibrium.time_slice.1.global_quantities.ip"] = 60.0e3
    return ods


def test_prepare_resolves_everything_without_importing_oft(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "OpenFUSIONToolkit", raising=False)
    config = TokaMakerConfig(time=0.32, workdir=tmp_path)
    inputs = prepare_tokamaker_inputs(_build_ods(), config)

    assert "OpenFUSIONToolkit" not in sys.modules
    assert inputs.shot == 12345
    assert inputs.time == pytest.approx(0.32)
    # nearest equilibrium slice (t=0.30) supplies the Ip target
    assert inputs.targets == {"Ip": pytest.approx(51.0e3)}
    assert inputs.f0 == pytest.approx(0.06)
    # per-turn amps interpolated at t=0.32
    assert inputs.coil_currents["PF1"] == pytest.approx(-640.0)
    assert inputs.coil_currents["PF2"] == pytest.approx(320.0)
    assert inputs.mesh_file.parent == tmp_path
    assert not inputs.mesh_exists
    assert (tmp_path / "geometry.json").is_file()


def test_magnetics_source_never_reads_equilibrium(tmp_path):
    ods = _build_ods()
    del ods["equilibrium"]
    config = TokaMakerConfig(time=0.5, workdir=tmp_path, constraint_source="magnetics")
    inputs = prepare_tokamaker_inputs(ods, config)
    assert inputs.targets["Ip"] == pytest.approx(50.0e3)


def test_explicit_targets_take_precedence(tmp_path):
    config = TokaMakerConfig(
        time=0.32, workdir=tmp_path,
        ip=40.0e3, pax=500.0, ip_ratio=1.5, r0_target=0.38, v0_target=0.01,
    )
    inputs = prepare_tokamaker_inputs(_build_ods(), config)
    assert inputs.targets == {
        "Ip": pytest.approx(40.0e3),
        "pax": pytest.approx(500.0),
        "Ip_ratio": pytest.approx(1.5),
        "R0": pytest.approx(0.38),
        "V0": pytest.approx(0.01),
    }


def test_time_index_selects_equilibrium_slice(tmp_path):
    config = TokaMakerConfig(time_index=1, workdir=tmp_path)
    inputs = prepare_tokamaker_inputs(_build_ods(), config)
    assert inputs.time == pytest.approx(0.50)
    assert inputs.targets["Ip"] == pytest.approx(60.0e3)


def test_nonpositive_ip_is_rejected(tmp_path):
    config = TokaMakerConfig(time=0.32, workdir=tmp_path, ip=-30.0e3)
    with pytest.raises(ValueError, match="positive current"):
        prepare_tokamaker_inputs(_build_ods(), config)


def test_unknown_constraint_source_is_rejected(tmp_path):
    config = TokaMakerConfig(time=0.32, workdir=tmp_path, constraint_source="mixed")
    with pytest.raises(ValueError, match="constraint_source"):
        prepare_tokamaker_inputs(_build_ods(), config)


def test_missing_time_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="time"):
        prepare_tokamaker_inputs(_build_ods(), TokaMakerConfig(workdir=tmp_path))


def test_f0_precedence_explicit_over_bt0_over_tf(tmp_path):
    ods = _build_ods()
    base = dict(time=0.32, workdir=tmp_path)
    assert prepare_tokamaker_inputs(ods, TokaMakerConfig(**base)).f0 == pytest.approx(0.06)
    assert prepare_tokamaker_inputs(
        ods, TokaMakerConfig(**base, bt0=0.5)
    ).f0 == pytest.approx(0.5 * 0.40)
    assert prepare_tokamaker_inputs(
        ods, TokaMakerConfig(**base, bt0=0.5, f0=0.123)
    ).f0 == pytest.approx(0.123)


def test_explicit_coil_currents_override_ods(tmp_path):
    config = TokaMakerConfig(
        time=0.32, workdir=tmp_path, coil_currents={"pf1": 111.0, "PF2": -222.0}
    )
    inputs = prepare_tokamaker_inputs(_build_ods(), config)
    assert inputs.coil_currents == {"PF1": 111.0, "PF2": -222.0}


def test_resolve_mesh_file_explicit_vs_hashed(tmp_path):
    ods = _build_ods()
    hashed_config = TokaMakerConfig(workdir=tmp_path)
    geometry = tokamaker_geometry_from_ods(ods, hashed_config)

    hashed, exists = resolve_mesh_file(geometry, hashed_config)
    assert hashed.name.startswith("vest_gs_mesh_") and hashed.suffix == ".h5"
    assert not exists

    explicit_path = tmp_path / "shared_mesh.h5"
    explicit_path.write_bytes(b"")
    explicit_config = TokaMakerConfig(workdir=tmp_path, mesh_file=explicit_path)
    explicit, exists = resolve_mesh_file(geometry, explicit_config)
    assert explicit == explicit_path
    assert exists
