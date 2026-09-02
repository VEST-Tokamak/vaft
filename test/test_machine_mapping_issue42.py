from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from omas import ODS

from vaft.machine_mapping import (
    b_field_pol_probe_from_raw_database,
    diamagnetic_flux_rogowski_coil_from_raw_database,
    flux_loop_from_raw_database,
    ip_rogowski_coil_from_raw_database,
)
from vaft.machine_mapping.equilibrium import equilibrium
from vaft.machine_mapping.magnetics import vfit_magnetics_dynamic, vfit_mirnov_raw_dynamic
from vaft.data.eqdsk import infer_source_shot_time
from vaft.process.magnetics import VestEquilibriumMagneticsResult


DATA = Path(__file__).resolve().parents[1] / "vaft" / "data" / "efit"


def test_equilibrium_merges_and_sorts_gfiles():
    ods = ODS()
    equilibrium(
        ods,
        [DATA / "g040330.00323", DATA / "g040330.00320", DATA / "g040330.00321"],
    )

    np.testing.assert_allclose(ods["equilibrium.time"], [0.320, 0.321, 0.323])
    assert len(ods["equilibrium.time_slice"]) == 3
    assert ods["dataset_description.data_entry.pulse"] == 40330
    assert ods["equilibrium.ids_properties.homogeneous_time"] == 1


def test_equilibrium_prefers_header_time_for_high_resolution_vfit_name():
    shot, time = infer_source_shot_time(DATA / "g039020.031180")
    assert shot == 39020
    assert time == pytest.approx(0.3118)


def test_equilibrium_rejects_out_of_order_append():
    ods = ODS()
    equilibrium(ods, [DATA / "g039915.00319"])
    with pytest.raises(ValueError, match="chronological order"):
        equilibrium(ods, [DATA / "g039915.00317"])
    np.testing.assert_allclose(ods["equilibrium.time"], [0.319])


def test_equilibrium_validates_inputs_and_supports_replace(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        equilibrium(ODS(), [])
    with pytest.raises(TypeError, match="sequence"):
        equilibrium(ODS(), str(DATA / "g040330.00320"))
    with pytest.raises(ValueError, match="same shot"):
        equilibrium(ODS(), [DATA / "g040330.00320", DATA / "g039915.00319"])

    renamed = tmp_path / "equilibrium_a"
    renamed.write_bytes((DATA / "g040330.00320").read_bytes())
    ods = ODS()
    equilibrium(ods, [renamed], {"times": [0.5]})
    equilibrium(ods, [DATA / "g040330.00321"], {"replace": True})
    assert len(ods["equilibrium.time_slice"]) == 1
    np.testing.assert_allclose(ods["equilibrium.time"], [0.321])


def _fake_md(*args, **kwargs):
    del args, kwargs
    time = np.array([0.26, 0.31, 0.36])
    return VestEquilibriumMagneticsResult(
        time=time,
        flux_loops=[np.array([1.0, 2.0, 3.0])],
        probes=[np.array([4.0, 5.0, 6.0])],
        flux_loop_voltage_time=[time],
        flux_loop_voltage=[np.array([0.1, 0.2, 0.3])],
    )


@patch("vaft.machine_mapping.magnetics.vfit_equilibrium_magnetics_detailed", side_effect=_fake_md)
def test_flux_loop_mapper_only_adds_flux_loop_channels(_mock_md):
    payload = {}
    flux_loop_from_raw_database(payload, 41672, dt=0.01)
    assert "flux_loop" in payload["magnetics"]
    assert "b_field_pol_probe" not in payload["magnetics"]
    assert "ip" not in payload["magnetics"]


@patch("vaft.machine_mapping.magnetics.vfit_mirnov_raw_dynamic")
@patch("vaft.machine_mapping.magnetics.vfit_equilibrium_magnetics_detailed", side_effect=_fake_md)
def test_probe_mapper_only_adds_probe_channels(_mock_md, _mock_raw):
    payload = {}
    b_field_pol_probe_from_raw_database(payload, 41672, dt=0.01)
    assert "b_field_pol_probe" in payload["magnetics"]
    assert "flux_loop" not in payload["magnetics"]
    assert "ip" not in payload["magnetics"]


@patch("vaft.machine_mapping.magnetics.vfit_plasma_current")
def test_ip_rogowski_mapper_only_adds_ip(mock_ip):
    mock_ip.return_value = (np.array([0.26, 0.31, 0.36]), np.array([0.0, 10.0, 0.0]))
    payload = {}
    ip_rogowski_coil_from_raw_database(payload, 41672, dt=0.01)
    assert "ip" in payload["magnetics"]
    assert "diamagnetic_flux" not in payload["magnetics"]


@patch("vaft.machine_mapping.magnetics.vfit_plasma_current")
@patch("vaft.machine_mapping.magnetics.vfit_equilibrium_magnetics_detailed", side_effect=_fake_md)
def test_split_magnetics_mappers_reject_incompatible_shared_time(mock_md, mock_ip):
    del mock_md
    mock_ip.return_value = (
        np.array([0.27, 0.32, 0.37]),
        np.array([0.0, 10.0, 0.0]),
    )
    payload = {}
    flux_loop_from_raw_database(payload, 41672, dt=0.0)
    with pytest.raises(ValueError, match="different timebase"):
        ip_rogowski_coil_from_raw_database(payload, 41672, dt=0.0)
    np.testing.assert_allclose(payload["magnetics"]["time"], [0.26, 0.31, 0.36])


@patch("vaft.machine_mapping.magnetics._safe_vest_load", return_value=None)
@patch("vaft.machine_mapping.magnetics.vest_diamagnetic_flux_detailed")
@patch("vaft.machine_mapping.magnetics.vfit_plasma_current")
def test_diamagnetic_mapper_does_not_add_ip(mock_ip, mock_dia, _mock_load):
    time = np.array([0.26, 0.31, 0.36])
    mock_ip.return_value = (time, np.array([0.0, 10.0, 0.0]))
    mock_dia.return_value = (time, np.array([0.0, 1.0, 0.0]), {"n_saturated": 0, "field": 257})
    payload = {}
    diamagnetic_flux_rogowski_coil_from_raw_database(payload, 41672, dt=0.01)
    assert "diamagnetic_flux" in payload["magnetics"]
    assert "ip" not in payload["magnetics"]


@patch("vaft.machine_mapping.magnetics.vfit_mirnov_raw_dynamic")
@patch("vaft.machine_mapping.magnetics._map_diamagnetic_flux")
@patch("vaft.machine_mapping.magnetics.vfit_plasma_current")
@patch("vaft.machine_mapping.magnetics.vfit_equilibrium_magnetics_detailed", side_effect=_fake_md)
def test_integrated_magnetics_prepares_md_context_once(mock_md, mock_ip, _mock_dia, _mock_raw):
    mock_ip.return_value = (np.array([0.26, 0.31, 0.36]), np.zeros(3))
    vfit_magnetics_dynamic({}, 41672, 0.26, 0.36, 0.01)
    mock_md.assert_called_once()


def test_raw_mirnov_voltage_is_cropped_without_changing_native_sampling(tmp_path):
    """The diagnostics window is shared, but raw fluctuation bandwidth is not."""
    del tmp_path
    shot = 39915
    time_source = 0.24 + np.arange(50_000, dtype=float) * 4e-6
    samples = np.arange(time_source.size, dtype=float)

    def load(_shot, field, _raw_source=None):
        return (time_source, samples) if field == 179 else None

    ods = ODS(consistency_check=False)
    with patch("vaft.machine_mapping.magnetics._safe_vest_load", side_effect=load):
        vfit_mirnov_raw_dynamic(ods, shot, tstart=0.26, tend=0.36)

    time = np.asarray(ods["magnetics.b_field_pol_probe.0.voltage.time"])
    data = np.asarray(ods["magnetics.b_field_pol_probe.0.voltage.data"])
    assert time[0] >= 0.26
    assert time[-1] < 0.36
    assert time.size == data.size
    np.testing.assert_allclose(np.diff(time), 4e-6)
    assert time.size > 2_500  # raw sampling was not reduced to the 25 kHz grid
