import json
from dataclasses import replace

import numpy as np
import pytest
from omas import ODS

from vaft.code.efit import EFITConfig, generate_kfile, prepare_efit_inputs
from vaft.code.efit_config import (
    EFITConstraintConfig,
    EFITInitializationConfig,
    EFITNumericsConfig,
    EFITProfileConfig,
    EFITScientificConfig,
    efit_parameter_grid,
)


def _constraints_ods(tmp_path, *, time=0.319):
    ods = ODS(consistency_check=False)
    ods["equilibrium.time"] = np.asarray([time])
    ods["equilibrium.code.parameters.time_slice.0.IN1.INPUT_DIR"] = str(tmp_path)
    ods["equilibrium.code.parameters.time_slice.0.IN1.VCURRT"] = np.asarray([3.0, -2.0])
    for index in range(16):
        root = f"equilibrium.time_slice.0.constraints.pf_current.{index}"
        ods[f"{root}.measured"] = float(index + 1)
        ods[f"{root}.measured_error_upper"] = 0.25
        ods[f"{root}.weight"] = 2.0
    scalar_values = {
        "ip": (50_000.0, 20.0, 3.0),
        "diamagnetic_flux": (-0.004, 0.0002, 4.0),
        "b_field_tor_vacuum_r": (0.08, 0.001, 1.0),
    }
    for name, (measured, error, weight) in scalar_values.items():
        root = f"equilibrium.time_slice.0.constraints.{name}"
        ods[f"{root}.measured"] = measured
        ods[f"{root}.measured_error_upper"] = error
        ods[f"{root}.weight"] = weight
    for name, measured, error, weight in (
        ("bpol_probe", 0.01, 0.002, 5.0),
        ("flux_loop", 0.02, 0.003, 6.0),
    ):
        root = f"equilibrium.time_slice.0.constraints.{name}.0"
        ods[f"{root}.measured"] = measured
        ods[f"{root}.measured_error_upper"] = error
        ods[f"{root}.weight"] = weight
    return ods


def _kfile_text(tmp_path, scientific=None):
    ods = _constraints_ods(tmp_path)
    generate_kfile(
        ods,
        39915,
        save_dir=str(tmp_path),
        config=scientific,
    )
    return next((tmp_path / "kfile").iterdir()).read_text(encoding="utf-8")


def test_routine_defaults_preserve_documented_kfile_semantics(tmp_path):
    text = _kfile_text(tmp_path, EFITScientificConfig())

    expected = {
        "KPPCUR": "2",
        "KFFCUR": "2",
        "KPPFNC": "0",
        "KFFFNC": "0",
        "PCURBD": "1",
        "FCURBD": "1",
        "CUTIP": "5000.0",
        "RELIP": "0.4",
        "AELIP": "0.3",
        "EELIP": "1.6",
        "RELAX": "1.0",
        "ERROR": "1e-05",
        "SERROR": "0.0005",
        "MXITER": "-100",
        "IVESEL": "1",
        "IFITVS": "0",
        "KCCOILS": "12",
    }
    for key, value in expected.items():
        assert f" {key} = {value}" in text


def test_legacy_profile_order_arguments_remain_supported(tmp_path):
    ods = _constraints_ods(tmp_path)
    generate_kfile(ods, 39915, 3, 4, save_dir=str(tmp_path))
    text = next((tmp_path / "kfile").iterdir()).read_text(encoding="utf-8")

    assert " KPPCUR = 3" in text
    assert " KFFCUR = 4" in text


def test_scientific_config_rejects_conflicting_legacy_profile_orders(tmp_path):
    ods = _constraints_ods(tmp_path)
    scientific = EFITScientificConfig(profile=EFITProfileConfig(kppcur=3))

    with pytest.raises(ValueError, match="npprime conflicts"):
        generate_kfile(
            ods,
            39915,
            npprime=4,
            save_dir=str(tmp_path),
            config=scientific,
        )


def test_scientific_config_accepts_matching_legacy_profile_orders(tmp_path):
    ods = _constraints_ods(tmp_path)
    scientific = EFITScientificConfig(profile=EFITProfileConfig(kppcur=3, kffcur=4))

    generate_kfile(
        ods,
        39915,
        npprime=3,
        nffprime=4,
        save_dir=str(tmp_path),
        config=scientific,
    )
    text = next((tmp_path / "kfile").iterdir()).read_text(encoding="utf-8")

    assert " KPPCUR = 3" in text
    assert " KFFCUR = 4" in text


def test_typed_settings_reach_their_namelist_fields(tmp_path):
    profile = EFITProfileConfig(
        kppcur=3, kffcur=4, kppfnc=1, kfffnc=2, pcurbd=0, fcurbd=0
    )
    initialization = EFITInitializationConfig(
        rzero=0.45,
        zzero=0.02,
        minor_radius=0.25,
        elongation=1.8,
        current_threshold=7_500.0,
    )
    numerics = EFITNumericsConfig(
        relaxation=0.8,
        error_tolerance=2e-6,
        measurement_error_floor=1e-3,
        max_iterations=250,
    )
    constraints = EFITConstraintConfig(
        group_weights={"plasma_current": 8.0, "bpol_probe": 9.0},
        use_diamagnetic_flux=False,
        diamagnetic_flux_sign="negative",
        wall_current_mode="disabled",
        passive_structure_mode="fit_currents",
    )
    text = _kfile_text(
        tmp_path,
        EFITScientificConfig(profile, initialization, numerics, constraints),
    )

    for field, value in {
        "KPPCUR": 3,
        "KFFCUR": 4,
        "KPPFNC": 1,
        "KFFFNC": 2,
        "PCURBD": 0,
        "FCURBD": 0,
        "CUTIP": 7500.0,
        "RELIP": 0.45,
        "ZELIP": 0.02,
        "AELIP": 0.25,
        "EELIP": 1.8,
        "RELAX": 0.8,
        "ERROR": 2e-06,
        "SERROR": 0.001,
        "MXITER": -250,
        "IVESEL": 1,
        "IFITVS": 1,
    }.items():
        assert f" {field} = {value}" in text
    assert "FWTCUR= 8.0" in text
    assert "BITMPI= 9000.000" in text
    assert "FWTDLC= 0" in text
    assert "DFLUX= -4.0" in text
    assert "VCURRT= 0.0, 0.0" in text


def test_standard_deviation_mode_uses_measurement_errors(tmp_path):
    constraints = EFITConstraintConfig(uncertainty_mode="standard_deviation")
    text = _kfile_text(
        tmp_path,
        EFITScientificConfig(constraints=constraints),
    )

    assert "BITFC= 0.25" in text
    assert "BITIP= 20.0" in text
    assert "BITMPI= 0.002" in text
    assert "PSIBIT= 0.000477464829" in text
    assert "SIGDLC= 0.2" in text


def test_resolved_configuration_is_stable_and_manifest_checksums_kfiles(tmp_path):
    ods = _constraints_ods(tmp_path)
    config = EFITConfig(
        workdir=tmp_path,
        shot=39915,
        profile=EFITProfileConfig(kppcur=3),
        provenance={"geometry_version": "vest-2025-07", "source": "main"},
    )

    inputs = prepare_efit_inputs(ods, config)
    payload = json.loads(inputs.manifest.read_text(encoding="utf-8"))

    assert payload["resolved"] == inputs.configuration
    assert payload["requested"]["scientific"] == payload["resolved"]["scientific"]
    assert payload["requested"]["typed_profile_supplied"]
    assert payload["resolved"]["scientific_sha256"] == config.scientific_config().sha256
    assert payload["resolved"]["provenance"]["geometry_version"] == "vest-2025-07"
    assert len(payload["kfiles"][0]["sha256"]) == 64
    assert inputs.manifest in inputs.files


def test_parameter_grid_is_deterministic_and_validated():
    grid = efit_parameter_grid(
        EFITScientificConfig(),
        {
            "profile.kppcur": [2, 3],
            "numerics.relaxation": [1.0, 0.8],
            "constraints.group_weights.bpol_probe": [4.0, 8.0],
            "initialization.rzero": [0.4, 0.45],
        },
    )

    assert len(grid) == 16
    assert len({item.sha256 for item in grid}) == 16
    assert grid[0].profile.kppcur == 2
    assert grid[-1].profile.kppcur == 3
    assert grid[-1].constraints.group_weights["bpol_probe"] == 8.0


def test_scientific_configuration_round_trips_through_json():
    original = EFITScientificConfig(
        profile=EFITProfileConfig(kppcur=4),
        constraints=EFITConstraintConfig(
            group_weights={"flux_loop": 7.0},
            passive_structure_mode="disabled",
        ),
    )

    payload = json.loads(json.dumps(original.to_dict()))
    restored = EFITScientificConfig.from_dict(payload)

    assert restored == original
    assert restored.sha256 == original.sha256


def test_integral_scalar_types_are_canonicalized_before_hashing():
    numpy_config = EFITScientificConfig(
        profile=EFITProfileConfig(kppcur=np.int64(2)),
        numerics=EFITNumericsConfig(max_iterations=np.int64(100)),
        constraints=EFITConstraintConfig(nccoil=np.int64(0)),
    )

    assert numpy_config.to_dict() == EFITScientificConfig().to_dict()
    assert numpy_config.sha256 == EFITScientificConfig().sha256


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: EFITProfileConfig(kppcur=-1), "kppcur"),
        (lambda: EFITProfileConfig(kppcur=2.0), "kppcur"),
        (lambda: EFITProfileConfig(kppfnc=0.0), "kppfnc"),
        (lambda: EFITProfileConfig(pcurbd=1.0), "pcurbd"),
        (lambda: EFITProfileConfig(pcurbd=2), "pcurbd"),
        (lambda: EFITInitializationConfig(minor_radius=0), "minor_radius"),
        (lambda: EFITNumericsConfig(error_tolerance=0), "error_tolerance"),
        (lambda: EFITNumericsConfig(max_iterations=100.0), "max_iterations"),
        (
            lambda: EFITConstraintConfig(diamagnetic_flux_sign="legacy"),
            "diamagnetic_flux_sign",
        ),
        (
            lambda: EFITConstraintConfig(group_weights={"unknown": 1.0}),
            "unknown EFIT diagnostic",
        ),
        (lambda: EFITConstraintConfig(nccoil=0.0), "nccoil"),
        (
            lambda: EFITConstraintConfig(
                coil_constraint_matrix=((1.0, 0.0),),
                coil_constraint_targets=(0.0,),
            ),
            "targets length",
        ),
    ],
)
def test_invalid_scientific_configuration_fails_before_execution(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


def test_legacy_and_typed_profile_conflicts_fail_early():
    with pytest.raises(ValueError, match="npprime conflicts"):
        EFITConfig(npprime=4, profile=EFITProfileConfig(kppcur=3))

    with pytest.raises(ValueError, match="npprime must be a positive integer"):
        EFITConfig(npprime=2.0)


def test_custom_coil_matrix_is_validated_against_selected_machine_coils(tmp_path):
    constraints = replace(
        EFITConstraintConfig(),
        coil_constraint_matrix=((1.0,),),
        coil_constraint_targets=(0.0,),
    )

    with pytest.raises(ValueError, match="row count"):
        _kfile_text(
            tmp_path,
            EFITScientificConfig(constraints=constraints),
        )
