import json
from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest
from omas import ODS

from vaft.machine_mapping.efund_geometry import EFIT16_GROUP_NAMES
from vaft.code.efit import (
    EFITConfig,
    EFITConstraintConfig,
    EFITInitializationConfig,
    EFITNumericsConfig,
    EFITProfileConfig,
    EFITScientificConfig,
    efit_parameter_grid,
    generate_kfile,
    prepare_efit_inputs,
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



def test_the_termination_settings_are_absent_until_a_study_sets_them(tmp_path):
    """Issue #171: the four settings a VEST fit stops on are now expressible.

    Absent by default, so the routine k-file is unchanged and EFIT's own
    defaults still apply -- which is exactly the situation #171 exists to
    characterize, and it must stay reproducible while it is characterized.
    """
    routine = _kfile_text(tmp_path, EFITScientificConfig())
    for key in ("ERRMIN", "SAICON", "ICONVR", "NXITER"):
        assert f" {key} " not in routine

    tuned = _kfile_text(
        tmp_path,
        EFITScientificConfig(
            numerics=EFITNumericsConfig(
                error_minimum=1.0e-3,
                chi_squared_target=60.0,
                convergence_mode=1,
                inner_iterations=20,
            )
        ),
    )
    added = [line for line in tuned.splitlines() if line not in routine.splitlines()]
    assert sorted(line.split("=")[0].strip() for line in added) == [
        "ERRMIN",
        "ICONVR",
        "NXITER",
        "SAICON",
    ]
    assert " ERRMIN = 0.001" in tuned and " SAICON = 60.0" in tuned

    # The scientific hash must move with them, or a scan would record two
    # different configurations under one identity.
    assert EFITScientificConfig().sha256 != EFITScientificConfig(
        numerics=EFITNumericsConfig(chi_squared_target=60.0)
    ).sha256


@pytest.mark.parametrize(
    "bad",
    [
        {"error_minimum": 0.0},
        {"chi_squared_target": -1.0},
        {"convergence_mode": 0},
        {"inner_iterations": True},
    ],
)
def test_the_termination_settings_refuse_values_efit_cannot_use(bad):
    with pytest.raises(ValueError):
        EFITNumericsConfig(**bad)


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


def test_the_coil_groups_come_from_the_machine_not_from_a_written_down_list():
    """The 16- and 26-coil conventions are gone, and this is what replaced them.

    VEST energises ten PF circuits. EFIT's table carries sixteen current
    groups over them, and that mapping used to live in a hard-coded
    twenty-six-entry coil list plus a table of positions picking sixteen of
    it. Both are deleted: each group's circuit is derived from its own name
    and the grouping from each element's own position, in the one module that
    also projects the F-coil groups for EFUND.
    """
    from vaft.machine_mapping.efund_geometry import (
        EFIT16_GROUP_NAMES,
        EFIT16_SOURCE_CIRCUIT,
        efit16_group_for_element,
    )

    assert len(EFIT16_GROUP_NAMES) == 16
    assert [EFIT16_SOURCE_CIRCUIT[name] for name in EFIT16_GROUP_NAMES] == (
        ["PF1"] * 8 + ["PF5", "PF5", "PF6", "PF6", "PF9", "PF9", "PF10", "PF10"]
    )
    assert efit16_group_for_element("PF1", 0.15) == "PF1-1"
    assert efit16_group_for_element("PF1", 1.10) == "PF1-4"
    assert efit16_group_for_element("PF1", -0.15) == "PF1-5"
    assert efit16_group_for_element("PF9", 1.0) == "PF9U"
    assert efit16_group_for_element("PF9", -1.0) == "PF9L"
    for absent in ("PF2", "PF3", "PF4", "PF7", "PF8"):
        assert efit16_group_for_element(absent, 0.5) is None

    # The old names are gone from the package, not merely unused.
    import vaft.code.efit as package
    import vaft.code.efit.kfile as kfile_module
    import vaft.code.efit.legacy as legacy_module

    for gone in ("vfit_pf_active_efit26", "efit16_group_indices", "EFIT16_GROUP_POSITIONS"):
        assert not hasattr(package, gone), gone
        assert not hasattr(kfile_module, gone), gone
        assert not hasattr(legacy_module, gone), gone


def test_the_coil_currents_fan_one_circuit_out_to_its_groups():
    """A split group is bookkeeping: it does not split the current.

    Every solenoid segment carries PF1's measured current and each pair
    carries its own circuit's, which is what the deleted twenty-six-entry
    fan-out did by hand. Circuits are matched by name, so a reordered
    `pf_active` cannot silently swap two coils.
    """
    from omas import ODS

    from vaft.code.efit.kfile import build_efit_coil_currents

    source = ODS(consistency_check=False)
    source["pf_active.time"] = np.linspace(0.25, 0.40, 64)
    # Deliberately not in PF1..PF10 order: a positional reading would break.
    for index, name in enumerate(
        ["PF10", "PF9", "PF8", "PF7", "PF6", "PF5", "PF4", "PF3", "PF2", "PF1"]
    ):
        source[f"pf_active.coil.{index}.name"] = name
        source[f"pf_active.coil.{index}.current.data"] = np.full(64, float(name[2:]))

    built = ODS(consistency_check=False)["pf_active"]
    build_efit_coil_currents(built, source["pf_active"])

    assert len(built["coil"]) == 16
    currents = [float(built[f"coil.{i}.current.data"][0]) for i in range(16)]
    assert currents == [1.0] * 8 + [5.0, 5.0, 6.0, 6.0, 9.0, 9.0, 10.0, 10.0]
    assert [str(built[f"coil.{i}.name"]) for i in range(16)] == list(EFIT16_GROUP_NAMES)


def test_the_coil_currents_are_taken_on_the_time_base_they_were_measured_on():
    """The writer had VEST's analysis window written into it, and did not need it.

    `build_efit_coil_currents` resampled onto a fixed 0.26-0.36 s at 40 us.
    Every packaged product already arrives on exactly that grid -- 2500 uniform
    40 us samples from 0.26 s -- so the interpolation was a series onto its own
    abscissa, and the three machine numbers bought nothing. Not resampling is
    now the default, and a caller that needs a grid passes one.
    """
    from omas import ODS

    from vaft.code.efit.kfile import build_efit_coil_currents

    source = ODS(consistency_check=False)
    # Deliberately not the old hard-coded grid.
    base = np.linspace(0.30, 0.34, 7)
    source["pf_active.time"] = base
    for index, name in enumerate(["PF1", "PF5", "PF6", "PF9", "PF10"]):
        source[f"pf_active.coil.{index}.name"] = name
        source[f"pf_active.coil.{index}.current.data"] = np.arange(7, dtype=float)

    built = ODS(consistency_check=False)["pf_active"]
    build_efit_coil_currents(built, source["pf_active"])
    np.testing.assert_array_equal(np.asarray(built["time"]), base)
    np.testing.assert_array_equal(
        np.asarray(built["coil.0.current.data"]), np.arange(7, dtype=float)
    )


def test_a_caller_that_wants_a_grid_passes_one():
    from omas import ODS

    from vaft.code.efit.kfile import build_efit_coil_currents

    source = ODS(consistency_check=False)
    source["pf_active.time"] = np.linspace(0.30, 0.34, 5)
    for index, name in enumerate(["PF1", "PF5", "PF6", "PF9", "PF10"]):
        source[f"pf_active.coil.{index}.name"] = name
        source[f"pf_active.coil.{index}.current.data"] = np.linspace(0.0, 4.0, 5)

    built = ODS(consistency_check=False)["pf_active"]
    build_efit_coil_currents(built, source["pf_active"], window=(0.31, 0.33, 0.005))
    grid = np.asarray(built["time"])
    # `np.arange`'s half-open end is approximate in floating point, so pin the
    # contract -- start, step, and inside the record -- not the element count.
    assert grid[0] == pytest.approx(0.31)
    np.testing.assert_allclose(np.diff(grid), 0.005)
    assert grid[-1] <= 0.33 + 1e-12

    with pytest.raises(ValueError, match="step must be positive"):
        build_efit_coil_currents(
            ODS(consistency_check=False)["pf_active"], source["pf_active"], window=(0.31, 0.33, 0.0)
        )
    with pytest.raises(ValueError, match="does not overlap"):
        build_efit_coil_currents(
            ODS(consistency_check=False)["pf_active"], source["pf_active"], window=(0.9, 1.0, 0.001)
        )


def test_one_probe_count_rule_serves_every_consumer():
    """Three copies of `min(present, defined)` had accumulated, and disagreed.

    The k-file writer, the channel-decision layer and the EFUND projection each
    carried the rule. In the degenerate case -- no channel defined -- the
    writer's copy returned zero probes where the other two returned every
    probe present, so a machine without VEST's channel table would have had
    every probe constraint silently dropped from its k-file.
    """
    from vaft.code.efit.kfile import _efit_bpol_probe_count
    from vaft.machine_mapping.efund_geometry import equilibrium_probe_count as projected
    from vaft.machine_mapping.magnetics import equilibrium_probe_count
    from vaft.validation.efit_channels import efit_probe_count
    from vaft.omas.sample import sample_ods

    ods = sample_ods()
    answers = {
        equilibrium_probe_count(ods),
        efit_probe_count(ods),
        projected(ods),
        _efit_bpol_probe_count(ods["magnetics"]),
    }
    assert len(answers) == 1, answers
    # The sample carries more channels than EFIT's geometry represents; the
    # trailing toroidal-Mirnov references are diagnostics, not constraints.
    assert answers.pop() < len(ods["magnetics.b_field_pol_probe"])


def test_the_probe_count_accepts_either_shape():
    """Its callers hold a whole ODS or a magnetics sub-tree, and both must work."""
    from vaft.machine_mapping.magnetics import equilibrium_probe_count
    from vaft.omas.sample import sample_ods

    ods = sample_ods()
    assert equilibrium_probe_count(ods) == equilibrium_probe_count(ods["magnetics"])

    from omas import ODS

    assert equilibrium_probe_count(ODS(consistency_check=False)) == 0


def test_the_writer_holds_no_machine_timing():
    """A VEST window in a generic routine is machine policy in the wrong place."""
    source = Path("vaft/code/efit/kfile.py").read_text(encoding="utf-8")
    for literal in ("0.26", "0.36", "4e-5"):
        assert f"tstart = {literal}" not in source
        assert f"tend = {literal}" not in source
        assert f"dt = {literal}" not in source


def test_a_pf_active_missing_a_driven_circuit_is_refused():
    """Silently writing a zero current for a missing coil would be worse."""
    from omas import ODS

    from vaft.code.efit.kfile import build_efit_coil_currents

    partial = ODS(consistency_check=False)
    partial["pf_active.time"] = np.linspace(0.25, 0.40, 64)
    partial["pf_active.coil.0.name"] = "PF1"
    partial["pf_active.coil.0.current.data"] = np.zeros(64)
    with pytest.raises(ValueError, match="PF10, PF5, PF6, PF9"):
        build_efit_coil_currents(
            ODS(consistency_check=False)["pf_active"], partial["pf_active"]
        )


def test_a_table_describing_a_different_coilset_is_refused(tmp_path):
    """A silent trim is the shape of every future mistake in this area.

    The writer used to take `min(groups in the tree, nfsum in the table)`, so a
    constraint tree describing more coils than the table declared simply lost
    its trailing groups. The k-file stayed well formed and quietly omitted
    them, which is exactly what nobody would notice. It is an error now, and
    the message names both sides.
    """
    ods = _constraints_ods(tmp_path)
    # The fixture's tree carries sixteen PF groups; say the table has two.
    (tmp_path / "mhdin.dat").write_text(" &machinein\n nfsum = 2\n /\n", encoding="utf-8")

    with pytest.raises(ValueError, match="different coilsets"):
        generate_kfile(ods, 39915, save_dir=str(tmp_path), config=EFITScientificConfig())


def test_a_table_that_agrees_still_writes(tmp_path):
    """The other half: the check must not refuse the routine case."""
    ods = _constraints_ods(tmp_path)
    (tmp_path / "mhdin.dat").write_text(" &machinein\n nfsum = 16\n /\n", encoding="utf-8")

    generate_kfile(ods, 39915, save_dir=str(tmp_path), config=EFITScientificConfig())
    text = next((tmp_path / "kfile").iterdir()).read_text(encoding="utf-8")
    # All sixteen groups reach the writer: the repeat count in FWTFC is the
    # number of coils the k-file actually carries.
    assert "FWTFC= 16*" in text
    assert "BRSP= " in text and " KCCOILS = 12" in text


def test_the_constraints_tree_stores_no_machine_values_the_writer_overrides():
    """Five VEST numbers sat in the stored parameters and never reached a k-file.

    `generate_constraints_ods` wrote CUTIP, RZERO, RELIP, AELIP and EELIP into
    `code.parameters`, and the writer took all five from the configuration
    instead -- so they were inert, and `CUTIP` had drifted to 50000 A against
    the configuration's 5000. Anything the writer overrides must not be stored
    beside it pretending to be the value in force.
    """
    from pathlib import Path as _Path

    source = (_Path("vaft/code/efit/kfile.py")).read_text(encoding="utf-8")
    for key in ("IN1.CUTIP", "IN1.RZERO", "IN1.RELIP", "IN1.AELIP", "IN1.EELIP"):
        assert f'PM[f"time_slice.{{i}}.{key}"]' not in source, key


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
