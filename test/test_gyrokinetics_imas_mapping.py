"""CGYRO -> IMAS gyrokinetics_local / core_transport mapping (#1354 stage 3)."""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from vaft.code.gacode import cgyro
from vaft.code.gacode.cgyro import collect_cgyro_outputs
from vaft.machine_mapping import gyrokinetics as gk

from test_cgyro_adapter import tglf_local, write_run


def linear_case(tmp_path, **run_kwargs):
    local = cgyro.cgyro_input_from_tglf(tglf_local())
    outputs = collect_cgyro_outputs(write_run(tmp_path / "run", **run_kwargs))
    provenance = {
        "parameters": cgyro.cgyro_parameters(local, cgyro.CGYROConfig()),
        "formalism": cgyro.formalism(),
        "resolution": cgyro.CGYROConfig().resolution(),
        "gacode_commit": "b493397",
        "version": {"revision": "b493397"},
        "input_sha256": "abc",
    }
    return local, outputs, provenance


def test_conversion_factors_follow_the_gkdb_normalisation(tmp_path):
    local, outputs, _ = linear_case(tmp_path, b_gs2=0.8)
    factors = gk.conversion_factors(local, outputs)
    assert factors["rate"] == pytest.approx(1.49 / np.sqrt(2))
    assert factors["wavenumber"] == pytest.approx(np.sqrt(2) / 0.8)
    assert factors["beta"] == pytest.approx(1 / 0.64)


def test_the_linear_eigenvalue_is_written_in_the_stated_convention(tmp_path):
    local, outputs, provenance = linear_case(tmp_path, ion_direction=">", omega=0.4)
    ods = ODS()
    report = gk.gyrokinetics_local_from_cgyro(ods, local, outputs, provenance=provenance, time=0.32)
    rate = 1.49 / np.sqrt(2)
    mode = "gyrokinetics_local.linear.wavevector.0.eigenmode.0"
    assert ods[f"{mode}.growth_rate_norm"] == pytest.approx(0.25 * rate)
    # omega=+0.4 is the ion direction in this run -> written negative
    assert ods[f"{mode}.frequency_norm"] == pytest.approx(-0.4 * rate)
    assert "ion_diamagnetic_negative" in ods[f"{mode}.code.parameters"]
    assert ods["gyrokinetics_local.linear.wavevector.0.binormal_wavevector_norm"] == pytest.approx(
        0.3 * np.sqrt(2) / 0.8)
    assert report["written"]


def test_species_gradients_are_rescaled_to_r0_and_electrons_kept_last(tmp_path):
    local, outputs, provenance = linear_case(tmp_path)
    ods = ODS()
    gk.gyrokinetics_local_from_cgyro(ods, local, outputs, provenance=provenance)
    assert ods["gyrokinetics_local.species.2.charge_norm"] == -1.0
    assert ods["gyrokinetics_local.species.2.temperature_log_gradient_norm"] == pytest.approx(1.5 * 1.49)
    assert ods["gyrokinetics_local.flux_surface.r_minor_norm"] == pytest.approx(0.7 / 1.49)
    assert ods["gyrokinetics_local.flux_surface.ip_sign"] == -1.0


def test_model_flags_follow_the_field_model(tmp_path):
    local, outputs, provenance = linear_case(tmp_path)
    provenance["parameters"]["N_FIELD"] = 2
    ods = ODS()
    gk.gyrokinetics_local_from_cgyro(ods, local, outputs, provenance=provenance)
    assert ods["gyrokinetics_local.model.include_a_field_parallel"] == 1
    assert ods["gyrokinetics_local.model.include_b_field_parallel"] == 0


def test_unsupported_quantities_are_reported_not_written(tmp_path):
    local, outputs, provenance = linear_case(tmp_path)
    ods = ODS()
    report = gk.gyrokinetics_local_from_cgyro(ods, local, outputs, provenance=provenance)
    assert any("collisionality_norm" in reason for reason in report["skipped"])
    assert any("shape_coefficients" in reason for reason in report["skipped"])
    assert "collisions" not in ods["gyrokinetics_local"]
    assert "shape_coefficients_c" not in ods["gyrokinetics_local.flux_surface"]


def test_without_an_ion_direction_no_frequency_is_written(tmp_path):
    local, outputs, provenance = linear_case(tmp_path)
    outputs.ion_direction = None
    ods = ODS()
    report = gk.gyrokinetics_local_from_cgyro(ods, local, outputs, provenance=provenance)
    mode = ods["gyrokinetics_local.linear.wavevector.0.eigenmode.0"]
    assert "growth_rate_norm" in mode and "frequency_norm" not in mode
    assert any("ion direction" in reason for reason in report["skipped"])


def test_without_b_gs2_nothing_is_written(tmp_path):
    local, outputs, provenance = linear_case(tmp_path)
    outputs.equilibrium["b_gs2"] = None
    ods = ODS()
    report = gk.gyrokinetics_local_from_cgyro(ods, local, outputs, provenance=provenance)
    assert not report["written"] and "gyrokinetics_local" not in ods


def test_the_eigenfunction_is_on_the_geometric_angle_and_normalised_at_zero(tmp_path):
    local, outputs, provenance = linear_case(tmp_path)
    ods = ODS()
    gk.gyrokinetics_local_from_cgyro(ods, local, outputs, provenance=provenance)
    mode = "gyrokinetics_local.linear.wavevector.0.eigenmode.0"
    angle = ods[f"{mode}.angle_pol"]
    phi = ods[f"{mode}.fields.phi_potential_perturbed_norm"]
    assert np.all(np.diff(angle) > 0)
    assert phi.shape == (angle.size, 1)
    assert phi[np.argmin(np.abs(angle)), 0] == pytest.approx(1.0)


def test_the_geometric_angle_is_clockwise_and_keeps_the_ballooning_turns():
    local = cgyro.cgyro_input_from_tglf(tglf_local(delta_loc=0.0, kappa_loc=1.0))
    theta = np.array([0.0, np.pi / 2, 2 * np.pi + np.pi / 2])
    angle = gk.geometric_angle(theta, local)
    # circular: theta=pi/2 is straight up, i.e. -pi/2 clockwise
    assert angle == pytest.approx([0.0, -np.pi / 2, -(2 * np.pi + np.pi / 2)])


def test_the_ods_survives_a_save_load_round_trip(tmp_path):
    from omas import load_omas_json, save_omas_json

    local, outputs, provenance = linear_case(tmp_path)
    ods = ODS()
    gk.gyrokinetics_local_from_cgyro(ods, local, outputs, provenance=provenance, time=0.32)
    save_omas_json(ods, str(tmp_path / "gk.json"))
    back = load_omas_json(str(tmp_path / "gk.json"))
    assert back["gyrokinetics_local.code.name"] == "CGYRO"
    assert back["gyrokinetics_local.linear.wavevector.0.eigenmode.0.growth_rate_norm"] == pytest.approx(
        ods["gyrokinetics_local.linear.wavevector.0.eigenmode.0.growth_rate_norm"])


def test_nonlinear_fluxes_need_an_explicit_window(tmp_path):
    local = cgyro.cgyro_input_from_tglf(tglf_local())
    outputs = collect_cgyro_outputs(write_run(tmp_path / "run", n_n=4, n_field=2, flux=True,
                                              exit_message="Normal"))
    provenance = {"parameters": {"N_FIELD": 2, "NONLINEAR_FLAG": 1}}
    ods = ODS()
    report = gk.gyrokinetics_local_from_cgyro(ods, local, outputs, provenance=provenance)
    assert any("window" in reason for reason in report["skipped"])
    ods = ODS()
    gk.gyrokinetics_local_from_cgyro(ods, local, outputs, provenance=provenance,
                                     flux_window=(2.0, 5.0))
    factor = gk.conversion_factors(local, outputs)["flux"]
    energy = ods["gyrokinetics_local.non_linear.fluxes_1d.energy_phi_potential"]
    assert energy[0] == pytest.approx(2.0 * 4 * factor)
    assert "energy_a_field_parallel" in ods["gyrokinetics_local.non_linear.fluxes_1d"]


class _Profile:
    rmin = np.linspace(0.0, 0.3, 31)
    rho = np.linspace(0.0, 1.0, 31) ** 0.9


def test_core_transport_gets_its_own_model_entry_beside_tglf(tmp_path):
    local = cgyro.cgyro_input_from_tglf(tglf_local())
    outputs = collect_cgyro_outputs(write_run(tmp_path / "run", n_n=4, n_field=1, flux=True,
                                              exit_message="Normal"))
    ods = ODS()
    ods["core_transport.model.0.identifier.index"] = 6
    ods["core_transport.model.0.code.name"] = "TGLF"
    report = gk.core_transport_from_cgyro(ods, [(local, outputs, (2.0, 5.0))], _Profile(), time=0.32)
    assert report["written"]
    assert ods["core_transport.model.0.code.name"] == "TGLF"
    assert ods["core_transport.model.1.code.name"] == "CGYRO"
    base = "core_transport.model.1.profiles_1d.0"
    expected = 2.0 * 4 * local.normalisation.energy_flux
    assert ods[f"{base}.electrons.energy.flux"][0] == pytest.approx(expected)
    assert ods[f"{base}.ion.1.label"] == "C6+"


def test_every_audited_class_is_one_of_the_declared_kinds():
    kinds = {"exact", "unit", "coordinate", "derived", "convention", "unsupported"}
    assert {kind for kind, _ in gk.MAPPING_AUDIT.values()} <= kinds
