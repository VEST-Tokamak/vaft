"""NICE adapter preparation/collection tests; no NICE installation required."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from omas import ODS

from vaft.code.nice import (
    NiceConfig,
    collect_nice_outputs,
    constraint_family_configs,
    lcfs_rms_displacement,
    prepare_nice_inputs,
    summarize_window,
    vest_reference_parameter_file,
)


def _ods():
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 41672
    ods["wall.description_2d.0.limiter.unit.0.outline.r"] = [0.2, 0.6, 0.6, 0.2]
    ods["wall.description_2d.0.limiter.unit.0.outline.z"] = [-0.4, -0.4, 0.4, 0.4]
    ods["pf_active.time"] = [0.3, 0.4]
    ods["pf_active.coil.0.name"] = "PF1"
    b = "pf_active.coil.0.element.0"
    ods[f"{b}.geometry.rectangle.r"] = 0.1
    ods[f"{b}.geometry.rectangle.z"] = 0.5
    ods[f"{b}.geometry.rectangle.width"] = 0.04
    ods[f"{b}.geometry.rectangle.height"] = 0.1
    ods[f"{b}.turns_with_sign"] = 8.0
    ods["pf_active.coil.0.current.data"] = [-1000.0, -1200.0]
    ods["pf_passive.time"] = [0.3, 0.4]
    p = "pf_passive.loop.0.element.0"
    ods["pf_passive.loop.0.name"] = "W1"
    ods[f"{p}.geometry.rectangle.r"] = 0.65
    ods[f"{p}.geometry.rectangle.z"] = 0.0
    ods[f"{p}.geometry.rectangle.width"] = 0.01
    ods[f"{p}.geometry.rectangle.height"] = 0.2
    ods["pf_passive.loop.0.current"] = [10.0, 20.0]
    ods["magnetics.time"] = [0.3, 0.4]
    ods["magnetics.ip.0.data"] = [100_000.0, 120_000.0]
    ods["magnetics.ip.0.time"] = [0.3, 0.4]
    ods["magnetics.b_field_pol_probe.0.identifier"] = "BP1"
    ods["magnetics.b_field_pol_probe.0.position.r"] = 0.7
    ods["magnetics.b_field_pol_probe.0.position.z"] = 0.1
    ods["magnetics.b_field_pol_probe.0.poloidal_angle"] = 3 * math.pi / 2
    ods["magnetics.b_field_pol_probe.0.field.data"] = [0.01, 0.02]
    ods["magnetics.b_field_pol_probe.0.field.time"] = [0.3, 0.4]
    ods["magnetics.flux_loop.0.identifier"] = "FL1"
    ods["magnetics.flux_loop.0.position.0.r"] = 0.6
    ods["magnetics.flux_loop.0.position.0.z"] = 0.2
    ods["magnetics.flux_loop.0.flux.data"] = [0.001, 0.002]
    ods["magnetics.flux_loop.0.flux.time"] = [0.3, 0.4]
    ods["tf.time"] = [0.3, 0.4]
    ods["tf.b_field_tor_vacuum_r.data"] = [0.08, 0.08]
    return ods


def test_prepare_is_nice_free_and_hashes_fixed_passive_state(tmp_path):
    inputs = prepare_nice_inputs(
        _ods(),
        NiceConfig(
            time=0.35,
            workdir=tmp_path,
            source_revision="abc",
            passive_current_mode="external_coils",
        ),
    )
    assert inputs.passive_currents == pytest.approx((15.0,))
    assert not (tmp_path / "input" / "param.xml").exists()
    assert (tmp_path / "input" / "coils.txt").read_text().splitlines()[0] == "2"
    assert float(
        (tmp_path / "input" / "Bprobes.txt").read_text().splitlines()[1].split()[2]
    ) == pytest.approx(3 * np.pi / 2)
    assert float(
        (tmp_path / "input" / "Icoils.txt").read_text().splitlines()[-1]
    ) == pytest.approx(15)
    manifest = json.loads(inputs.manifest_file.read_text())
    assert manifest["nice_source_revision"] == "abc"
    assert len(manifest["passive_current_hash"]) == 64
    assert len(manifest["diagnostic_channel_hash"]) == 64
    assert {d.identifier for d in inputs.diagnostics} == {"Ip", "BP1", "FL1"}


def test_default_passive_treatment_subtracts_fixed_response(tmp_path):
    inputs = prepare_nice_inputs(_ods(), NiceConfig(time=0.35, workdir=tmp_path))
    assert (tmp_path / "input" / "coils.txt").read_text().splitlines()[0] == "1"
    assert (
        inputs.manifest["passive_current_treatment"]
        == "fixed_forward_model_subtracted_from_diagnostics"
    )
    assert next(
        d for d in inputs.diagnostics if d.identifier == "BP1"
    ).value != pytest.approx(0.015)


def test_disabled_channel_is_preserved_in_manifest_but_not_native_fit(tmp_path):
    inputs = prepare_nice_inputs(
        _ods(), NiceConfig(time=0.35, workdir=tmp_path, disabled_channels=("BP1",))
    )
    bp = next(d for d in inputs.diagnostics if d.identifier == "BP1")
    assert not bp.enabled and bp.reason
    assert (tmp_path / "input" / "Bprobes.txt").read_text().splitlines() == ["0"]


def test_collect_native_outputs_without_nice(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (tmp_path / "nice_case_manifest.json").write_text('{"cocos_out": 11}')
    (output / "dataEqui_global_quantities.txt").write_text(
        "0.331 0.4 0.2 0.5 0.01 1.0 100000 100000 0.8 0.2 0.1 1.2 0.01 0.02 0.35 0.01 1.1 3.2 0\n"
    )
    (output / "dataEqui_convergence_cost.txt").write_text("0.331 7 1e-9 2 1 0.6 0.4\n")
    (output / "dataEqui_bp.txt").write_text("0.331 1 0.01 0.011 1000\n")
    result = collect_nice_outputs(tmp_path)
    assert result.converged and result.nonlinear_iterations == 7
    assert result.ods["equilibrium.time_slice.0.global_quantities.ip"] == pytest.approx(
        100000
    )
    assert result.ods["equilibrium.time_slice.0.time"] == pytest.approx(0.331)
    assert result.diagnostic_residuals[0]["normalized_residual"] == pytest.approx(1.0)
    assert result.ods["equilibrium.code.name"] == "NICE"


def test_collect_native_unstructured_psi_as_profiles_2d(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (tmp_path / "nice_case_manifest.json").write_text(
        '{"cocos_out": 11, "process_returncode": 0}'
    )
    (output / "dataEqui_global_quantities.txt").write_text(
        "0.331 0.4 0.2 0.5 0.01 1.0 100000 100000 0.8 0.2 0.1 1.2 0.01 0.02 0.35 0.01 1.1 3.2 0\n"
    )
    (output / "dataEqui_convergence_cost.txt").write_text(
        "0.331 7 1e-11 2 1 0.6 0.4\n"
    )
    (output / "dataEqui_mesh_coord.txt").write_text(
        "0.2 -0.1\n0.6 -0.1\n0.2 0.1\n0.6 0.1\n"
    )
    # time, node count, then psi/Br/Bz/Btor blocks.
    (output / "dataEqui_psi_br_bz_btor.txt").write_text(
        "0.331 4 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0\n"
    )
    result = collect_nice_outputs(tmp_path)
    r = np.asarray(result.ods["equilibrium.time_slice.0.profiles_2d.0.grid.dim1"])
    z = np.asarray(result.ods["equilibrium.time_slice.0.profiles_2d.0.grid.dim2"])
    psi = np.asarray(result.ods["equilibrium.time_slice.0.profiles_2d.0.psi"])
    assert r.shape == z.shape == (129,)
    assert psi.shape == (129, 129)
    assert np.all(np.isfinite(psi))
    assert psi[0, 0] == pytest.approx(-2 * np.pi)


def test_parameter_cocos_must_match_config(tmp_path):
    parameter = tmp_path / "param.xml"
    parameter.write_text(
        "<parameters><inCOCOS>13</inCOCOS><outCOCOS>11</outCOCOS></parameters>"
    )
    with pytest.raises(ValueError, match="does not match"):
        prepare_nice_inputs(
            _ods(),
            NiceConfig(time=0.35, workdir=tmp_path / "case", parameter_file=parameter),
        )


def test_effective_cocos_manager_is_required(tmp_path):
    parameter = tmp_path / "param.xml"
    parameter.write_text(
        "<parameters><inCOCOS>11</inCOCOS><outCOCOS>11</outCOCOS></parameters>"
    )
    with pytest.raises(ValueError, match="Effective NICE"):
        prepare_nice_inputs(
            _ods(),
            NiceConfig(time=0.35, workdir=tmp_path / "case", parameter_file=parameter),
        )
    parameter.write_text(
        "<parameters><inCOCOS>11</inCOCOS><outCOCOS>11</outCOCOS><useNewCOCOSManager>1</useNewCOCOSManager><inoutCOCOS>11</inoutCOCOS></parameters>"
    )
    inputs = prepare_nice_inputs(
        _ods(),
        NiceConfig(
            time=0.35,
            workdir=tmp_path / "valid",
            parameter_file=parameter,
            flux_loop_input_sign=-1,
        ),
    )
    assert inputs.manifest["effective_cocos"] == 11
    probe = next(d for d in inputs.diagnostics if d.family == "bpol_probe")
    assert probe.geometry["poloidal_angle"] == pytest.approx(3 * math.pi / 2)
    native_flux = np.loadtxt(inputs.input_dir / "fluxloops_meas.txt", skiprows=1)
    flux = next(d for d in inputs.diagnostics if d.family == "flux_loop")
    assert native_flux[0] == pytest.approx(-flux.value)
    assert native_flux[1] == pytest.approx(flux.uncertainty)


def test_native_flux_profiles_convert_from_nicpp(tmp_path):
    from vaft.code.nice.outputs import _native_ods

    (tmp_path / "dataEqui_global_quantities.txt").write_text(
        "0.331 0.4 0.2 0.5 0.01 1 100000 100000 0.8 0.2 0.1 1.2 0.01 0.02 0.35 0.01 1.1 3.2\n"
    )
    (
        tmp_path
        / "dataEqui_profiles_psi_rhotornorm_pressure_f_dpdpsi_fdfdpsi_jtor_q_Ne.txt"
    ).write_text("0.331 1 0.01 0.5 1000 0.08 -30 -40 100000 2 1e18\n")
    ods, errors = _native_ods(tmp_path, 11, -1, -1)
    assert not errors
    b = "equilibrium.time_slice.0"
    assert ods[b + ".global_quantities.psi_axis"] == pytest.approx(0.01 * 2 * np.pi)
    assert ods[b + ".global_quantities.ip"] == -100000
    assert ods[b + ".profiles_1d.dpressure_dpsi"][0] == pytest.approx(-30 / (2 * np.pi))
    assert ods[b + ".profiles_1d.f"][0] == pytest.approx(-0.08)
    assert ods[b + ".profiles_1d.q"][0] == 2
    # Declared where VAFT's readers look, so none of them falls back to guessing
    # whether psi is in Wb or Wb/rad.
    from vaft.data.eqdsk import ods_psi_to_wb_per_radian_factor
    from vaft.omas.general import ods_cocos

    assert ods_cocos(ods) == 11
    assert ods_psi_to_wb_per_radian_factor(ods) == pytest.approx(1 / (2 * np.pi))



def test_psi_derivative_profiles_compare_per_radian(tmp_path, monkeypatch):
    import copy

    import vaft.data.eqdsk as eqdsk
    from vaft.code.nice import compare_equilibria
    from vaft.code.nice.outputs import _native_ods

    (tmp_path / "dataEqui_global_quantities.txt").write_text(
        "0.331 0.4 0.2 0.5 0.01 1 100000 100000 0.8 0.2 0.1 1.2 0.01 0.02 0.35 0.01 1.1 3.2\n"
    )
    (
        tmp_path
        / "dataEqui_profiles_psi_rhotornorm_pressure_f_dpdpsi_fdfdpsi_jtor_q_Ne.txt"
    ).write_text("0.331 1 0.01 0.5 1000 0.08 -30 -40 100000 2 1e18\n")
    nice, _ = _native_ods(tmp_path, 11, -1, -1)
    # The same equilibrium as a legacy EFIT artifact storing psi in Wb/rad:
    # its psi-derivatives are 2*pi larger.
    efit = copy.deepcopy(nice)
    b = "equilibrium.time_slice.0.profiles_1d"
    for name in ("dpressure_dpsi", "f_df_dpsi"):
        efit[f"{b}.{name}"] = np.asarray(nice[f"{b}.{name}"]) * 2 * np.pi
    monkeypatch.setattr(
        eqdsk,
        "ods_psi_to_wb_per_radian_factor",
        lambda ods, index=0: 1.0 if ods is efit else 1 / (2 * np.pi),
    )
    report = compare_equilibria(nice, efit, 0.331)
    assert report["profiles"]["dpressure_dpsi"]["max_abs"] == pytest.approx(0, abs=1e-12)
    assert report["profiles"]["f_df_dpsi"]["max_abs"] == pytest.approx(0, abs=1e-12)

def test_shared_constraints_preserve_raw_conditioned_and_fixed_responses(tmp_path):
    ods = _ods()
    ods["equilibrium.time"] = [0.35]
    for name, value, sigma, weight in (
        ("ip", 115000, 2000, 0.1),
        ("bpol_probe.0", 0.012, 0.002, 0.1),
        ("flux_loop.0", 0.003, 0.0002, 0),
    ):
        base = "equilibrium.time_slice.0.constraints." + name
        ods[base + ".measured"] = value
        ods[base + ".measured_error_upper"] = sigma
        ods[base + ".weight"] = weight
    inputs = prepare_nice_inputs(
        ods,
        NiceConfig(
            time=0.35,
            workdir=tmp_path,
            diagnostic_source="equilibrium_constraints",
            correct_active_response=True,
        ),
    )
    bp = next(d for d in inputs.diagnostics if d.family == "bpol_probe")
    assert bp.original_value == pytest.approx(0.015)
    assert bp.conditioned_value == 0.012
    assert bp.uncertainty == 0.002 and bp.weight == 0.1
    assert (
        bp.value + bp.passive_response + bp.active_response_correction
        == pytest.approx(0.012)
    )
    assert not next(d for d in inputs.diagnostics if d.family == "flux_loop").enabled
    for row in inputs.manifest["active_response_audit"]:
        assert (
            abs(row["native"] + row["correction"] - row["exact"])
            < 0.1 * row["uncertainty"]
        )
    with pytest.raises(ValueError, match="exact-time"):
        prepare_nice_inputs(
            ods,
            NiceConfig(
                time=0.351,
                workdir=tmp_path / "wrong",
                diagnostic_source="equilibrium_constraints",
            ),
        )


@pytest.mark.parametrize(
    "log",
    [
        "_nBoundaryInnerNodes=175 _nBoundaryInnerEdges=181\n",
        "BEGIN iter=1\ncost=nan\n",
        "plasma valid = 0\n",
    ],
)
def test_failure_logs_cannot_be_success_without_tables(tmp_path, log):
    (tmp_path / "nice.stdout.log").write_text(log)
    (tmp_path / "nice_case_manifest.json").write_text('{"process_returncode":0}')
    result = collect_nice_outputs(tmp_path)
    assert result.process_succeeded and not result.ok
    assert result.converged is False
    if "BEGIN" in log:
        assert result.nonlinear_iterations == 1


def test_contour_rejects_crossings_and_reversed_orientation():
    from vaft.code.nice.geometry import validate_contour

    geometry = {
        "limiter": [[0.2, -0.4], [0.6, -0.4], [0.6, 0.4], [0.2, 0.4]],
        "pf_active": [],
    }
    validate_contour([0.1, 0.7, 0.7, 0.1], [-0.5, -0.5, 0.5, 0.5], geometry)
    with pytest.raises(ValueError, match="intersects"):
        validate_contour([0.1, 0.5, 0.5, 0.1], [-0.5, -0.5, 0.5, 0.5], geometry)
    with pytest.raises(ValueError, match="counter-clockwise"):
        validate_contour([0.1, 0.1, 0.7, 0.7], [-0.5, 0.5, 0.5, -0.5], geometry)


def test_unsupported_family_rejected_before_resolving_executable(tmp_path):
    from vaft.code.nice import run_nice

    config = NiceConfig(
        time=0.35,
        workdir=tmp_path,
        include_bpol_probes=False,
        executable="/does/not/exist",
    )
    result = run_nice(prepare_nice_inputs(_ods(), config), config)
    assert result.returncode is None and not result.ok
    assert "unsupported" in result.termination_reason


def test_trailing_probe_validity_cannot_disable_flux_loop():
    from vaft.validation.imas import write_validity
    from vaft.code.efit.kfile import _condemned_channels

    ods = _ods()
    ods["magnetics.b_field_pol_probe.1.field.data"] = [0.0, 0.0]
    write_validity(ods, "magnetics.b_field_pol_probe.1.field", [-2, -2], scalar=-2)
    assert _condemned_channels(ods, nbprobe=1) == set()


def test_native_zero_sign_diagnostic_tables_are_rejected(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    (out / "dataEqui_bp.txt").write_text("0 1 0 0 1000\n")
    (tmp_path / "nice_case_manifest.json").write_text(
        json.dumps(
            {
                "diagnostic_channels": [
                    {
                        "family": "bpol_probe",
                        "enabled": True,
                        "value": 0.01,
                        "uncertainty": 0.001,
                    }
                ]
            }
        )
    )
    result = collect_nice_outputs(tmp_path)
    assert any(
        "unverifiable native diagnostic output sign" in e for e in result.parsing_errors
    )
    assert not result.diagnostic_residuals


def test_auxiliary_sign_warning_does_not_reject_numerical_stage_success(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    (out / "dataEqui_global_quantities.txt").write_text(
        "0.331 0.4 0.2 0.5 0.01 1.0 100000 100000 0.8 0.2 0.1 1.2 0.01 0.02 0.35 0.01 1.1 3.2 0\n"
    )
    (out / "dataEqui_convergence_cost.txt").write_text(
        "0.331 7 1e-11 2 1 0.6 0.4\n"
    )
    (out / "dataEqui_bp.txt").write_text("0.331 1 0 0 1000\n")
    (tmp_path / "nice_case_manifest.json").write_text(
        json.dumps(
            {
                "process_returncode": 0,
                "solver_tolerances": {"epsStopRecon": 1e-10},
                "diagnostic_channels": [
                    {
                        "family": "bpol_probe",
                        "enabled": True,
                        "value": 0.01,
                        "uncertainty": 0.001,
                    }
                ],
            }
        )
    )
    result = collect_nice_outputs(tmp_path)
    assert result.ok
    assert "auxiliary output warnings" in result.termination_reason
    assert any(
        "unverifiable native diagnostic output sign" in error
        for error in result.parsing_errors
    )


def test_vacth_sentinel_is_not_final_numerical_convergence(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    (out / "dataEqui_global_quantities.txt").write_text(
        "0.331 0.4 0.2 0.5 0.01 1.0 100000 100000 0.8 0.2 0.1 1.2 0.01 0.02 0.35 0.01 1.1 3.2 0\n"
    )
    (out / "dataEqui_convergence_cost.txt").write_text(
        "0.331 0 -9e40 -9e40 -9e40\n"
    )
    (tmp_path / "nice_case_manifest.json").write_text(
        '{"process_returncode": 0, "solver_tolerances": {"epsStopRecon": 1e-10}}'
    )
    result = collect_nice_outputs(tmp_path)
    assert result.converged is False
    assert not result.ok


def test_coil_audit_detects_native_response_error(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    audits, channels = [], []
    for family, prefix in (("bpol_probe", "Bprobes"), ("flux_loop", "fluxloops")):
        (out / f"vacth_{prefix}_comp.txt").write_text("1.2\n")
        (out / f"vacth_{prefix}_comp_minus_pfcoils.txt").write_text("0\n")
        (out / f"vacth_{prefix}_meas.txt").write_text("1\n")
        audits.append(
            {
                "family": family,
                "ods_path": family,
                "exact": 1,
                "native": 1,
                "correction": 0,
                "uncertainty": 1,
            }
        )
        channels.append(
            {
                "family": family,
                "ods_path": family,
                "enabled": True,
                "value": 1,
                "uncertainty": 1,
            }
        )
    (tmp_path / "nice_case_manifest.json").write_text(
        json.dumps(
            {
                "active_response_audit": audits,
                "diagnostic_channels": channels,
                "flux_loop_input_sign": -1,
            }
        )
    )
    result = collect_nice_outputs(tmp_path)
    assert "active coil response mismatch exceeds 0.1 sigma" in result.parsing_errors
    assert not result.ok
    assert len(result.provenance["initializer_residuals"]) == 2


def test_contour_rejects_passive_intersection():
    from vaft.code.nice.geometry import validate_contour

    geometry = {
        "limiter": [[0.2, -0.4], [0.6, -0.4], [0.6, 0.4], [0.2, 0.4]],
        "pf_active": [],
        "pf_passive": [
            {
                "name": "wall",
                "outline": [[0.69, -0.2], [0.71, -0.2], [0.71, 0.2], [0.69, 0.2]],
            }
        ],
    }
    with pytest.raises(ValueError, match="intersects wall"):
        validate_contour([0.1, 0.7, 0.7, 0.1], [-0.5, -0.5, 0.5, 0.5], geometry)


def test_study_helpers_preserve_failures_and_compare_lcfs():
    from vaft.code.nice import NiceResult

    results = (
        NiceResult(
            0,
            __import__("pathlib").Path("a"),
            converged=True,
            scientifically_usable=True,
            ods=object(),
        ),
        NiceResult(
            1,
            __import__("pathlib").Path("b"),
            converged=False,
            termination_reason="failed",
        ),
    )
    summary = summarize_window([0.1, 0.2], results)
    assert summary["requested"] == 2 and summary["failed_or_missing_times_s"] == [0.2]
    assert (
        lcfs_rms_displacement([1, 2, 2, 1], [0, 0, 1, 1], [2, 2, 1, 1], [1, 0, 0, 1])
        < 1e-12
    )
    configs = constraint_family_configs(NiceConfig())
    assert (
        not configs["Core"].include_flux_loops
        and configs["Full"].include_diamagnetic_flux
    )


def test_real_41672_reference_slice_maps_without_nice(tmp_path):
    import vaft

    sample = (
        Path(__file__).parents[1] / "vaft" / "data" / "samples" / "41672" / "imas.nc"
    )
    source = sample.parent / "source" / "pipeline-until-efit.json.gz"
    ods = vaft.omas.load(source)
    inputs = prepare_nice_inputs(
        ods,
        NiceConfig(
            shot=41672,
            time=0.331,
            workdir=tmp_path,
            parameter_file=vest_reference_parameter_file(),
            source_revision="7ad1ea8f3da4fee25a61a7c2c01b1773db5f4906",
        ),
    )
    assert len(inputs.geometry["pf_active"]) < 100
    assert len(inputs.geometry["pf_passive"]) == len(inputs.passive_currents) == 950
    assert sum(channel.enabled for channel in inputs.diagnostics) == 75
    assert inputs.manifest["diagnostic_channel_set_hash"]
    assert inputs.manifest["input_snapshot_hash"]


def test_two_time_shared_conditioner_keeps_time_index_and_channel_identity(tmp_path):
    from vaft.omas import load
    from vaft.code.nice.validate_reference import condition

    repo = Path(__file__).parents[1]
    ods = load(repo / "vaft/data/samples/41672/source/pipeline-until-efit.json.gz")
    report = condition(ods, 41672, [0.331, 0.332], tmp_path / "condition", repo)
    assert np.allclose(ods["equilibrium.time"], [0.331, 0.332])
    assert set(report["channel_decisions"]) == {"b_field_pol_probe", "flux_loop"}
    inputs = prepare_nice_inputs(
        ods,
        NiceConfig(
            time=0.331,
            workdir=tmp_path / "native",
            diagnostic_source="equilibrium_constraints",
        ),
    )
    assert sum(d.enabled and d.family == "flux_loop" for d in inputs.diagnostics) == 5
    ods[
        "equilibrium.time_slice.0.constraints.bpol_probe.0.source"
    ] = "wrong physical probe"
    with pytest.raises(ValueError, match="source mismatch"):
        prepare_nice_inputs(
            ods,
            NiceConfig(
                time=0.331,
                workdir=tmp_path / "wrong",
                diagnostic_source="equilibrium_constraints",
            ),
        )


def test_executable_resolves_from_nicehome_build_layouts(tmp_path, monkeypatch):
    from external_code_stubs import write_launchable_stub, write_unlaunchable_file

    from vaft.code.nice.runner import resolve_nice_executable

    monkeypatch.delenv("NICEHOME", raising=False)
    with pytest.raises(FileNotFoundError, match=r"\$NICEHOME"):
        resolve_nice_executable(NiceConfig(workdir=tmp_path))

    home = tmp_path / "nice"
    # The message names the expected path, spelled with the host separator.
    with pytest.raises(FileNotFoundError, match=r"build[/\\]nice_recon"):
        resolve_nice_executable(NiceConfig(workdir=tmp_path, nice_home=home))

    # The second documented layout is found when the first is absent.
    exe = write_launchable_stub(home / "run" / "nice_recon")
    monkeypatch.setenv("NICEHOME", str(home))
    assert resolve_nice_executable(NiceConfig(workdir=tmp_path)) == exe

    refused = write_unlaunchable_file(tmp_path / "other" / "nice_recon")
    with pytest.raises(PermissionError):
        resolve_nice_executable(NiceConfig(workdir=tmp_path, executable=refused))


def test_window_report_renders_through_vaft_plot(tmp_path):
    import copy

    from vaft.code.nice import NiceResult
    from vaft.code.nice.outputs import _native_ods
    from vaft.code.nice.study import write_window_report

    (tmp_path / "dataEqui_global_quantities.txt").write_text(
        "0.331 0.4 0.2 0.5 0.01 1 100000 100000 0.8 0.2 0.1 1.2 0.01 0.02 0.35 0.01 1.1 3.2\n"
    )
    (
        tmp_path
        / "dataEqui_profiles_psi_rhotornorm_pressure_f_dpdpsi_fdfdpsi_jtor_q_Ne.txt"
    ).write_text("0.331 1 0.01 0.5 1000 0.08 -30 -40 100000 2 1e18\n")
    equilibrium, _ = _native_ods(tmp_path, 11, -1, -1)
    results = [
        NiceResult(0, tmp_path, converged=True, scientifically_usable=True, ods=equilibrium),
        NiceResult(1, tmp_path, converged=False, termination_reason="failed"),
    ]
    files = write_window_report(
        tmp_path / "report", 41672, [0.331, 0.332], results, efit_ods=equilibrium
    )
    assert files["traces"].stat().st_size > 0
    summary = json.loads(files["summary"].read_text())
    assert summary["shot"] == 41672 and len(summary["slice_comparisons"]) == 1
    comparison = summary["slice_comparisons"][0]
    assert comparison["exact_time_match"] and comparison["efit_time_s"] == 0.331

    # A reference that lacks the requested time is compared, but not called a match.
    from vaft.code.nice import compare_equilibria

    shifted = copy.deepcopy(equilibrium)
    shifted["equilibrium.time"] = np.asarray([0.333])
    assert not compare_equilibria(equilibrium, shifted, 0.331)["exact_time_match"]
