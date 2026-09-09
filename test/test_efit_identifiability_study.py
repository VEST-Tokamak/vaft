import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "workflow"
    / "efit_identifiability"
    / "identifiability_study.py"
)


def _study():
    spec = importlib.util.spec_from_file_location("efit_identifiability_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _frozen_kfile_text(reference, table_dir: Path) -> str:
    matrix = [[0.0 for _ in range(12)] for _ in range(16)]
    for column in range(7):
        matrix[0][column] = 1.0
        matrix[column + 1][column] = -1.0
    for column, upper in enumerate((8, 10, 12, 14), start=7):
        matrix[upper][column] = 1.0
        matrix[upper + 1][column] = -1.0
    matrix[13][11] = 1.0
    matrix[14][11] = -1.0
    relations = "\n".join(
        f" CCOILS(1,{column + 1})="
        + ",".join(str(matrix[row][column]) for row in range(16))
        for column in range(12)
    )
    return f""" &IN1
 IOUT=4
 AELIP=0.3
 CUTIP=5000
 EELIP=1.6
 FCURBD=1
 FWTBP=0
 ICINIT=2
 IFITVS=0
 ERRMIN=0.01
 SAICON=80
 ICONVR=2
 NXITER=1
 IVESEL=1
 KFFCUR=2
 KFFFNC=0
 KPPCUR=2
 KPPFNC=0
 PCURBD=1
 RELIP=0.32
 RZERO=0.4
 SERROR=0.0005
 TABLE_DIR='{table_dir}/'
 ISHOT={reference.shot}
 ITIME={reference.time_ms}
 FWTFC=16*1.0
 FWTSI=1.0
 FWTMP2=1.0
 FWTCUR=1.0
 FWTDLC=1.0
 RELAX=1.0
 ERROR=1e-5
 MXITER=-100
 /
 &INWANT
{relations}
 KCCOILS=12
 NCCOIL=0
 XCOILS=12*0.0
 /
"""


def _validation(study, reference, **updates):
    values = {
        "reference": reference,
        "outcome": "accepted",
        "writer_complete": True,
        "main_export_present": True,
        "external_current_export_present": True,
        "parameter_count": 20,
        "pf_parameter_count": 16,
        "pprime_parameter_count": 2,
        "ffprime_parameter_count": 2,
        "exact_constraint_count": 0,
        "solution_scaled_error": 1e-11,
        "residual_relative_error": 1e-12,
        "singular_value_relative_error": 1e-12,
        "condition_number_relative_error": 1e-8,
        "curvature_sum_relative_error": 1e-12,
        "mode_share_sum_error": 1e-12,
        "ip_vcurrt_relative_error": 1e-10,
        "mfile_family_chi2_max_relative_error": 1e-10,
        "mfile_family_chi2_max_absolute_error": 1e-16,
        "mfile_family_chi2_passed": True,
    }
    values.update(updates)
    return study.Stage1Validation(**values)


def test_reference_set_is_exact_and_stage1_gate_requires_every_slice():
    study = _study()
    assert [(item.shot, item.time_ms) for item in study.REFERENCE_SLICES] == [
        (41672, 331),
        (41672, 342),
        (41672, 347),
        (39915, 319),
        (41524, 332),
    ]
    validations = [_validation(study, item) for item in study.REFERENCE_SLICES]
    gate = study.stage1_gate(validations)
    assert gate.passed and gate.reasons == ()

    failed = study.stage1_gate(validations[:-1])
    assert not failed.passed
    assert "missing validation slices: 41524:332" in failed.reasons


def test_stage1_gate_checks_native_and_external_blocks_and_reproduction():
    study = _study()
    validations = [_validation(study, item) for item in study.REFERENCE_SLICES]
    validations[2] = _validation(
        study,
        study.REFERENCE_SLICES[2],
        external_current_export_present=False,
        solution_scaled_error=2e-8,
        parameter_count=19,
        ip_vcurrt_relative_error=1e-6,
        mfile_family_chi2_passed=False,
    )
    gate = study.stage1_gate(validations)
    assert not gate.passed
    assert any("external_current_export_present" in reason for reason in gate.reasons)
    assert any("parameter roles" in reason for reason in gate.reasons)
    assert any("solution_scaled_error" in reason for reason in gate.reasons)
    assert any("ip_vcurrt_relative_error" in reason for reason in gate.reasons)
    assert any("m-file family totals" in reason for reason in gate.reasons)


def test_native_problem_validation_reproduces_the_solve_and_parameter_roles():
    study = _study()
    diagonal = np.linspace(1.0, 2.0, 20)
    matrix = np.diag(diagonal)
    solution = np.linspace(-1.0, 1.0, 20)
    rhs = matrix @ solution
    block = SimpleNamespace(
        solver_a=matrix,
        solver_rhs=rhs,
        solver_solution=solution,
        row_solver_weighted_residual=np.zeros(20),
        singular_values=np.linalg.svd(matrix, compute_uv=False),
        retained_mask=np.ones(20, dtype=bool),
        condno=2.0,
        solver_exact_c=np.empty((0, 20)),
        solver_exact_d=np.empty(0),
        exact_c=np.empty((0, 20)),
        exact_d=np.empty(0),
        weighted_a=matrix,
        row_family=tuple("flux_loop" for _ in range(20)),
        parameter_role=(
            *("pf_current" for _ in range(16)),
            "pprime",
            "pprime",
            "ffprime",
            "ffprime",
        ),
        ncol=20,
        nexact=0,
    )
    report = SimpleNamespace(
        modes=tuple(
            SimpleNamespace(singular_value=value, family_shares={"flux_loop": 1.0})
            for value in diagonal
        )
    )
    problem = SimpleNamespace(
        main=block,
        external_current=object(),
        schema_version=1,
        executable_sha256="a" * 64,
    )
    validation = study.validate_native_problem(
        problem,
        report,
        reference=study.REFERENCE_SLICES[0],
        outcome="accepted",
        expected_executable_sha256="a" * 64,
    )
    assert validation.solution_scaled_error < 1e-12
    assert validation.residual_relative_error == 0.0
    assert validation.singular_value_relative_error == 0.0
    assert validation.condition_number_relative_error == 0.0
    assert validation.curvature_sum_relative_error == 0.0
    assert validation.mode_share_sum_error == 0.0
    assert validation.executable_sha256_matches
    mismatch = study.validate_native_problem(
        problem,
        report,
        reference=study.REFERENCE_SLICES[0],
        outcome="accepted",
        expected_executable_sha256="b" * 64,
    )
    assert not mismatch.executable_sha256_matches
    validations = [_validation(study, item) for item in study.REFERENCE_SLICES]
    validations[0] = mismatch
    gate = study.stage1_gate(validations)
    assert not gate.passed
    assert any("executable SHA-256" in reason for reason in gate.reasons)


def test_native_row_audit_keeps_measurements_weights_and_soft_relations():
    study = _study()
    block = SimpleNamespace(
        nrow=3,
        solver_residual=np.asarray([0.5, -2.0, 0.25]),
        row_solver_weighted_residual=np.asarray([0.5, -2.0, 0.25]),
        row_uncertainty=np.asarray([2.0, 1.0, np.nan]),
        row_physical_residual=np.asarray([1.0, -2.0, 0.25]),
        row_statistical=np.asarray([True, True, False]),
        row_family=("flux_loop", "flux_loop", "pf_relation"),
        row_kind=("statistical", "statistical", "soft_structural_relation"),
        row_channel=np.asarray([1, 2, 1]),
        row_measurement=np.asarray([10.0, 11.0, np.nan]),
        row_reconstruction=np.asarray([11.0, 9.0, np.nan]),
        row_submitted_fwt=np.asarray([1.0, 1.0, np.nan]),
        row_processed_weight=np.asarray([0.5, 1.0, 1.0]),
    )
    rows = study.native_row_records(
        SimpleNamespace(main=block, external_current=None), study.REFERENCE_SLICES[0]
    )
    assert rows[0]["measurement"] == 10.0
    assert rows[0]["diagnostic_chi2"] == pytest.approx(0.25)
    assert rows[2]["kind"] == "soft_structural_relation"
    assert rows[2]["uncertainty"] is None
    summary = study.summarize_native_rows(rows)
    flux = next(item for item in summary if item["family"] == "flux_loop")
    assert flux["active_channel_count"] == 2
    assert flux["residual_bias"] == pytest.approx(-0.5)
    assert flux["residual_rms"] == pytest.approx(np.sqrt(2.5))
    assert flux["uncertainty"] == {"min": 1.0, "median": 1.5, "max": 2.0}
    assert flux["submitted_fwt"] == {"min": 1.0, "median": 1.0, "max": 1.0}


def test_mfile_family_chi2_gate_uses_a_relative_or_absolute_floor_and_excludes_ip():
    study = _study()
    rows = [
        {
            "block": "main",
            "family": "flux_loop",
            "statistical": True,
            "diagnostic_chi2": 2.0,
        },
        {
            "block": "main",
            "family": "bpol_probe",
            "statistical": True,
            "diagnostic_chi2": 3.0,
        },
        {
            "block": "main",
            "family": "diamagnetic_flux",
            "statistical": True,
            "diagnostic_chi2": 2.0e-15,
        },
        {
            "block": "main",
            "family": "pf_current",
            "statistical": True,
            "diagnostic_chi2": 2.0e-22,
        },
        {
            "block": "main",
            "family": "plasma_current",
            "statistical": True,
            "diagnostic_chi2": 1.0e-18,
        },
        {
            "block": "main",
            "family": "pf_relation",
            "statistical": False,
            "diagnostic_chi2": None,
        },
    ]
    audit = study.compare_mfile_family_chi2(
        rows,
        {
            "flux_loop": 2.0 * (1.0 + 2.0e-8),
            "bpol_probe": 3.0 * (1.0 - 2.0e-8),
            # Large relative differences at numerical floor still pass on
            # absolute error, matching actual EFIT float32 m-file behavior.
            "diamagnetic_flux": 2.12e-15,
            "pf_current": 2.04e-22,
        },
    )
    assert audit["passed"]
    assert audit["families"]["diamagnetic_flux"]["relative_error"] > 0.05
    assert "plasma_current" in audit["excluded"]
    assert "pf_relation" in audit["excluded"]

    failed = study.compare_mfile_family_chi2(
        rows,
        {
            "flux_loop": 2.1,
            "bpol_probe": 3.0,
            "diamagnetic_flux": 2.0e-15,
            "pf_current": 2.0e-22,
        },
    )
    assert not failed["passed"]
    assert not failed["families"]["flux_loop"]["passed"]


def test_mfile_family_chi2_audit_reads_the_produced_array_names(tmp_path):
    study = _study()
    from scipy.io import netcdf_file

    path = tmp_path / "m041672.00331"
    values = {
        "saisil": 2.0,
        "saimpi": 3.0,
        "chidflux": 4.0e-15,
        "chifcc": 5.0e-22,
        # This must never be compared with the native Ip solve row.
        "chipasma": 74.0,
    }
    with netcdf_file(path, "w") as dataset:
        dataset.createDimension("channel", 1)
        for name, value in values.items():
            variable = dataset.createVariable(name, "d", ("channel",))
            variable[:] = [value]
    rows = [
        {
            "block": "main",
            "family": family,
            "statistical": True,
            "diagnostic_chi2": value,
        }
        for family, value in (
            ("flux_loop", 2.0),
            ("bpol_probe", 3.0),
            ("diamagnetic_flux", 4.0e-15),
            ("pf_current", 5.0e-22),
            ("plasma_current", 1.0e-18),
        )
    ]
    audit = study.audit_mfile_family_chi2(path, rows)
    assert audit["passed"]
    assert audit["sha256"] == study.sha256_file(path)
    assert set(audit["families"]) == set(study.MFILE_DIAGNOSTIC_CHI2_VARIABLES)


def test_case_identity_hashes_kfile_executable_direction_and_restart(tmp_path):
    study = _study()
    executable = tmp_path / "efit"
    kfile = tmp_path / "k041672.00331"
    direction = tmp_path / "direction.nc"
    restart = tmp_path / "esave.dat"
    for path, value in (
        (executable, b"binary"),
        (kfile, b"&IN1 /"),
        (direction, b"direction"),
        (restart, b"restart"),
    ):
        path.write_bytes(value)
    base = dict(
        stage=2,
        reference=study.REFERENCE_SLICES[0],
        kind="direction",
        kfile=kfile,
        executable=executable,
        scientific_sha256="a" * 64,
        table_identity={"sha256": "b" * 64},
        analysis_identity={"cutoff": "condin"},
        direction_file=direction,
        restart_parent=restart,
    )
    first, identity = study.case_identity(study.CaseSpec(**base))
    assert identity["direction_sha256"] == study.sha256_file(direction)
    assert identity["restart_parent_sha256"] == study.sha256_file(restart)
    restart.write_bytes(b"changed")
    second, _ = study.case_identity(study.CaseSpec(**base))
    assert first != second


def test_build_provenance_requires_clean_control_and_five_slice_equivalence(
    monkeypatch, tmp_path
):
    study = _study()
    executable = tmp_path / "efit"
    install_manifest = tmp_path / "install_manifest.txt"
    executable.write_bytes(b"efit")
    install_manifest.write_text("efit\n", encoding="utf-8")
    comparisons = []
    monkeypatch.setattr(
        study,
        "_validate_archived_issue_663_regression",
        lambda proof: comparisons.append(("archived", proof)),
    )
    monkeypatch.setattr(
        study,
        "_validate_same_metadata_regression",
        lambda proof, record: comparisons.append(("same", proof)),
    )
    monkeypatch.setattr(
        study,
        "_validate_final_clean_regression",
        lambda proof, record: comparisons.append(("final", proof)),
    )
    record = study.build_provenance_record(
        executable,
        {
            "source_base_revision": "4d10ed592f8c9d295d393d0cf331f2d8f6be3034",
            "source_base_is_ancestor": True,
            "source_revision": "b" * 40,
            "source_dirty": False,
            "build_type": "Release",
            "compiler": "GNU Fortran",
            "compiler_version": "15.2",
            "cmake_options": {"USE_NETCDF": True},
            "install_manifest": {
                "path": str(install_manifest),
                "sha256": study.sha256_file(install_manifest),
            },
            "install_manifest_sha256": study.sha256_file(install_manifest),
            "ctest": {
                "control": {"passed": 44, "total": 44},
                "instrumented": {"passed": 44, "total": 44},
                "additional_failures": 0,
            },
            "control_comparison": {
                "issue_663_baseline": {"proof": "archived"},
                "same_metadata_default_off_regression": {"proof": "same"},
                "final_clean_commit_validation": {"proof": "final"},
            },
        },
    )
    study.validate_build_provenance(record)
    assert [name for name, _ in comparisons] == ["archived", "same", "final"]
    assert record["executable"]["sha256"] == study.sha256_file(executable)
    dirty = {**record, "source_dirty": True}
    with pytest.raises(ValueError, match="clean EFIT"):
        study.validate_build_provenance(dirty)
    wrong_base = {**record, "source_base_revision": "a" * 40}
    with pytest.raises(ValueError, match="base commit"):
        study.validate_build_provenance(wrong_base)
    regression = {
        **record,
        "ctest": {
            "control": {"passed": 44, "total": 44},
            "instrumented": {"passed": 43, "total": 44},
            "additional_failures": 1,
        },
    }
    with pytest.raises(ValueError, match="no additional"):
        study.validate_build_provenance(regression)


def test_resume_requires_every_artifact_hash_to_still_match(tmp_path):
    study = _study()
    artifact = tmp_path / "response.nc"
    artifact.write_bytes(b"complete")
    identity = {"kfile_sha256": "abc"}
    manifest = tmp_path / "case_manifest.json"
    study.write_json_atomic(
        manifest,
        {
            "schema_version": study.SCHEMA_VERSION,
            "status": "succeeded",
            "identity": identity,
            "artifacts": [
                {"path": str(artifact), "sha256": study.sha256_file(artifact)}
            ],
        },
    )
    assert study.resumable_case_manifest(manifest, identity) is not None
    artifact.write_bytes(b"changed")
    assert study.resumable_case_manifest(manifest, identity) is None


def test_plan_resolves_one_kfile_per_slice_and_writes_atomic_case_manifests(tmp_path):
    study = _study()
    kfiles = tmp_path / "frozen"
    kfiles.mkdir()
    tables = tmp_path / "tables"
    tables.mkdir()
    for reference in study.REFERENCE_SLICES:
        (kfiles / f"k0{reference.shot}.{reference.time_ms:05d}").write_text(
            _frozen_kfile_text(reference, tables), encoding="utf-8"
        )
    executable = tmp_path / "efit"
    executable.write_bytes(b"efit")
    output = tmp_path / "study"
    payload = study.build_plan_manifest(
        output=output,
        kfile_root=kfiles,
        executable=executable,
        scientific_sha256=study.fixed_scientific_config().sha256,
        table_identity={
            "sources_by_shot": {
                str(reference.shot): {"path": str(tables)}
                for reference in study.REFERENCE_SLICES
            },
            "mhdin_sha256": "2" * 64,
        },
    )
    assert len(payload["cases"]) == 5
    assert not payload["stage2_authorized"]
    assert (output / study.STUDY_MANIFEST).is_file()
    for case in payload["cases"]:
        manifest = Path(case["workdir"]) / study.STAGE1_CASE_MANIFEST
        written = json.loads(manifest.read_text(encoding="utf-8"))
        assert written["case_id"] == case["case_id"]
        assert written["kfile_semantic_audit"]["passed"]


def test_duplicate_frozen_kfiles_must_be_byte_identical(tmp_path):
    study = _study()
    reference = study.REFERENCE_SLICES[0]
    name = f"k0{reference.shot}.{reference.time_ms:05d}"
    for directory in (tmp_path / "a", tmp_path / "b"):
        directory.mkdir()
        (directory / name).write_text("same", encoding="utf-8")
    assert study.resolve_reference_kfile(tmp_path, reference).name == name
    (tmp_path / "b" / name).write_text("different", encoding="utf-8")
    with pytest.raises(ValueError, match="ambiguous non-identical"):
        study.resolve_reference_kfile(tmp_path, reference)


def test_frozen_kfile_semantic_gate_rejects_same_name_tampering(tmp_path):
    study = _study()
    reference = study.REFERENCE_SLICES[0]
    tables = tmp_path / "tables"
    tables.mkdir()
    path = tmp_path / f"k0{reference.shot}.{reference.time_ms:05d}"
    path.write_text(_frozen_kfile_text(reference, tables), encoding="utf-8")
    identity = {
        "sources_by_shot": {str(reference.shot): {"path": str(tables)}}
    }
    audit = study.audit_frozen_kfile(
        path,
        reference=reference,
        scientific=study.fixed_scientific_config(),
        table_identity=identity,
    )
    assert audit["passed"]
    assert all(audit["active_families"].values())

    path.write_text(
        path.read_text(encoding="utf-8").replace("KPPCUR=2", "KPPCUR=3"),
        encoding="utf-8",
    )
    tampered = study.audit_frozen_kfile(
        path,
        reference=reference,
        scientific=study.fixed_scientific_config(),
        table_identity=identity,
    )
    assert not tampered["passed"]
    assert any("KPPCUR" in error for error in tampered["errors"])
    with pytest.raises(ValueError, match="semantic audit failed"):
        study.require_frozen_kfile_audit(tampered)

    path.write_text(
        _frozen_kfile_text(reference, tables).replace(
            "CCOILS(1,1)=1.0,-1.0", "CCOILS(1,1)=1.0,-0.5", 1
        ),
        encoding="utf-8",
    )
    altered_relations = study.audit_frozen_kfile(
        path,
        reference=reference,
        scientific=study.fixed_scientific_config(),
        table_identity=identity,
    )
    assert not altered_relations["passed"]
    assert any("CCOILS differs" in error for error in altered_relations["errors"])


def test_complete_663_tree_resolves_only_the_explicit_baseline(tmp_path):
    study = _study()
    reference = study.REFERENCE_SLICES[0]
    name = f"k0{reference.shot}.{reference.time_ms:05d}"
    baseline = tmp_path / f"shot_{reference.shot}" / "baseline" / "kfile"
    variant = tmp_path / f"shot_{reference.shot}" / "diamagnetic_flux_x1000" / "kfile"
    baseline.mkdir(parents=True)
    variant.mkdir(parents=True)
    (baseline / name).write_text("FWTDLC=1", encoding="utf-8")
    (variant / name).write_text("FWTDLC=1000", encoding="utf-8")
    assert study.resolve_reference_kfile(tmp_path, reference) == baseline / name


def test_diamagnetic_kfile_derivation_changes_only_fwtdlc(tmp_path):
    study = _study()
    source = tmp_path / "k041672.00331"
    source.write_text(
        "&IN1\n FWTCUR=3.0\n FWTDLC = 1.0\n IOUT=4\n/\n",
        encoding="utf-8",
    )
    destination = tmp_path / "derived" / source.name
    record = study.derive_diamagnetic_kfile(source, destination, 10 ** 0.25)
    assert source.read_text(encoding="utf-8").count("FWTDLC = 1.0") == 1
    assert "FWTDLC = 1.7782794100389228" in destination.read_text(encoding="utf-8")
    assert record["source_sha256"] == study.sha256_file(source)
    assert record["derived_sha256"] == study.sha256_file(destination)


def test_native_svd_family_shares_sum_to_one_and_scaling_is_fixed():
    study = _study()
    matrix = np.asarray(
        [
            [2.0, 0.0, 1.0],
            [0.0, 3.0, 1.0],
            [1.0, 1.0, 2.0],
            [1.0, -1.0, 0.5],
        ]
    )
    families = ("plasma_current", "pf_current", "flux_loop", "flux_loop")
    scaling = study.fixed_column_scaling(matrix)
    singular, _, shares = study.mode_family_shares(
        matrix, families, scaling, np.eye(matrix.shape[1])
    )
    total = sum(shares.values())
    assert singular.size == 3
    assert np.allclose(total[singular > 0], 1.0, atol=1e-12)
    assert np.allclose(np.linalg.norm(matrix * scaling, axis=0), 1.0)


def test_zero_norm_columns_keep_unit_scale_and_are_flagged_unobservable():
    study = _study()
    scaling, unobservable = study.fixed_column_scaling(
        np.asarray([[2.0, 0.0], [0.0, 0.0]]), return_unobservable=True
    )
    assert np.allclose(scaling, [0.5, 1.0])
    assert unobservable.tolist() == [False, True]


def test_true_constraints_are_projected_through_a_null_space():
    study = _study()
    constraint = np.asarray([[1.0, 1.0, 0.0]])
    basis = study.null_space(constraint)
    assert basis.shape == (3, 2)
    assert np.linalg.norm(constraint @ basis) < 1e-12
    assert np.allclose(basis.T @ basis, np.eye(2))


def test_information_classification_uses_rank_lift_and_special_labels():
    study = _study()
    independent = study.classify_family_information(
        {
            "rank_gain_by_cutoff": [1, 1, 1],
            "weak_subspace_action_fraction": 0.1,
            "weak_singular_value_lift": 1.0,
            "curvature_inverse_trace_reduction": 0.0,
        }
    )
    assert independent == ("independent information",)
    labels = study.classify_family_information(
        {
            "rank_gain_by_cutoff": [0, 0, 0],
            "weak_subspace_action_fraction": 0.0,
            "weak_singular_value_lift": 10.0,
            "baseline_solver_objective_share": 1e-4,
            "nonlinear_material_response": True,
            "pf_structural_response": True,
        }
    )
    assert labels == ("reinforcing", "overwhelmed", "structural anchor")


def test_nonlinear_and_prior_evidence_are_merged_into_final_family_labels():
    study = _study()
    payload = {
        "family_results": [
            {
                "reference": "41672:347",
                "family": "diamagnetic_flux",
                "rank_gain_by_cutoff": [0, 0, 0],
                "weak_subspace_action_fraction": 0.0,
                "weak_singular_value_lift": 1.0,
                "curvature_inverse_trace_reduction": 0.0,
                "baseline_solver_objective_share": 1e-6,
            },
            {
                "reference": "41672:347",
                "family": "pf_relation",
                "rank_gain_by_cutoff": [0, 0, 0],
                "weak_subspace_action_fraction": 0.0,
                "weak_singular_value_lift": 1.0,
                "curvature_inverse_trace_reduction": 0.0,
                "baseline_solver_objective_share": 0.0,
            },
        ]
    }
    merged = study.merge_nonlinear_family_evidence(
        payload,
        continuation_records=[
            {
                "reference": "41672:347",
                "control": "diamagnetic_flux_scale",
                "displacement": 1.2,
            }
        ],
        prior_evidence={
            "41672:347": {
                "pf_relation": {
                    "classification": ["structural anchor"],
                    "structural_effect": True,
                }
            }
        },
    )
    by_family = {item["family"]: item for item in merged["family_results"]}
    assert "overwhelmed" in by_family["diamagnetic_flux"]["classification"]
    assert "structural anchor" in by_family["pf_relation"]["classification"]


def test_mode_selection_is_weak_only_deduplicated_and_deterministic():
    study = _study()
    singular = [100.0, 5.0, 0.05, 0.01]
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.9, 0.0],
            [0.0, 0.9, 0.1, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    selected = study.select_target_modes(
        singular_values=singular,
        parameter_vectors=vectors,
        parameter_roles=("pf_current", "pprime", "ffprime", "pf_current"),
        diamagnetic_shares=(0.0, 0.7, 0.8, 0.9),
        tau=1.0,
    )
    assert [item.selector for item in selected] == [
        "profile",
        "pf_current",
        "diamagnetic_flux",
    ]
    assert len({item.mode_index for item in selected}) == 3
    assert all(item.mode_index != 0 for item in selected)


def test_mode_selection_accepts_the_public_report_json_shape():
    study = _study()
    report = {
        "nominal_cutoff": 1.0,
        "modes": [
            {
                "index": 0,
                "singular_value": 20.0,
                "state": "resolved",
                "parameter_role_shares": {"pprime": 1.0},
                "family_shares": {"diamagnetic_flux": 1.0},
            },
            {
                "index": 1,
                "singular_value": 2.0,
                "state": "borderline",
                "parameter_role_shares": {"pprime": 0.8, "ffprime": 0.1},
                "family_shares": {"diamagnetic_flux": 0.1},
            },
            {
                "index": 2,
                "singular_value": 0.5,
                "state": "borderline",
                "parameter_role_shares": {"pf_current": 0.9},
                "family_shares": {"diamagnetic_flux": 0.2},
            },
            {
                "index": 3,
                "singular_value": 0.05,
                "state": "unresolved",
                "parameter_role_shares": {"pf_current": 0.1},
                "family_shares": {"diamagnetic_flux": 0.9},
            },
        ],
    }
    selected = study.select_target_modes_from_report(report)
    assert [item.mode_index for item in selected] == [1, 2, 3]
    assert [item.selector for item in selected] == [
        "profile",
        "pf_current",
        "diamagnetic_flux",
    ]


def test_direction_control_matches_native_v1_schema(tmp_path):
    study = _study()
    block = SimpleNamespace(
        ncol=3,
        parameter_name=("PF1", "PPRIME1", "FFPRIME1"),
        parameter_role=("pf_current", "pprime", "ffprime"),
        parameter_index=np.asarray([1, 1, 1]),
        parameter_units=("A", "Pa/Wb", "T2/Wb"),
    )
    problem = SimpleNamespace(
        main=block,
        final_brsp=np.asarray([10.0, 20.0, 30.0]),
        sha256="a" * 64,
    )
    mode = SimpleNamespace(
        index=2,
        state="borderline",
        scaled_parameter_direction=np.asarray([0.5, -0.25, 1.0]),
    )
    report = SimpleNamespace(
        parameter_scale=np.asarray([2.0, 4.0, 0.5]),
        modes=(mode,),
    )
    path = tmp_path / "direction.nc"
    record = study.write_direction_control(
        path, problem=problem, report=report, mode_index=2, target=0.125
    )
    from scipy.io import netcdf_file

    with netcdf_file(path, "r", mmap=False) as dataset:
        assert dataset.schema_id == b"efit_direction_constraint_v1"
        assert int(dataset.schema_version) == 1
        assert int(dataset.writer_complete) == 1
        assert dataset.source_sidecar_sha256 == b"a" * 64
        assert np.allclose(dataset.variables["theta0"][:], [10.0, 20.0, 30.0])
        assert np.allclose(dataset.variables["physical_c"][:], [0.25, -0.0625, 2.0])
        assert float(dataset.variables["target"].data) == pytest.approx(0.125)
    assert record["parameter_order_sha256"] == study.parameter_order_sha256(block)


def test_diamagnetic_and_directional_plans_have_explicit_warm_ancestry():
    study = _study()
    assert study.diamagnetic_log10_schedule() == tuple(index / 4 for index in range(17))
    assert study.diamagnetic_scales()[0] == 1.0
    assert study.diamagnetic_scales()[-1] == pytest.approx(10_000.0)
    points = study.diamagnetic_continuation_plan()
    increasing = [item for item in points if item.chain.endswith("increasing")]
    decreasing = [item for item in points if item.chain.endswith("decreasing")]
    assert len(increasing) == len(decreasing) == 17
    assert increasing[0].cold_start
    assert decreasing[0].parent_chain == "diamagnetic_increasing"
    assert not decreasing[0].cold_start

    mode = study.TargetMode(2, "profile", 0.5, "borderline", True, 0.8, 0.1, 0.2)
    directions = study.directional_continuation_plan(mode)
    assert len(directions) == 12
    assert {item.parent_chain for item in directions if item.order == 0} == {"baseline"}
    assert all(not item.cold_start for item in directions)
    # A retained-but-borderline mode is not assigned a resolved-mode 1/s scale.
    assert directions[0].requested_value == pytest.approx(-0.125)
    assert directions[0].parent_order is None

    resolved = study.TargetMode(1, "profile", 0.5, "resolved", True, 0.8, 0.1, 0.2)
    assert study.directional_continuation_plan(resolved)[0].requested_value == pytest.approx(-0.25)


def test_direction_linearity_compares_adjacent_not_baseline_secants():
    study = _study()
    records = [
        {"status": "succeeded", "alpha": -0.125, "parameter_state": [-0.125]},
        {"status": "succeeded", "alpha": -0.25, "parameter_state": [-0.25]},
        {"status": "succeeded", "alpha": 0.125, "parameter_state": [0.125]},
        # The baseline secant is only 4% different, but the adjacent
        # derivative from .125 to .25 is 8% different and must fail.
        {"status": "succeeded", "alpha": 0.25, "parameter_state": [0.26]},
    ]
    result = study._direction_linearity(records, [0.0], [1.0], np.eye(1))
    assert result["negative"]["passed"]
    assert not result["positive"]["passed"]
    assert result["positive"]["derivative_relative_difference"] == pytest.approx(0.08)
    assert not result["passed"]


def test_native_restart_stdout_audit_is_machine_readable():
    study = _study()
    audit = study._native_restart_audit(
        "EFIT_RESTART_STATE physical_state_loaded=1 signature_match=0 "
        "minite_iteration_offset=8\n"
        "EFIT_RESTART_HISTORY fresh_response_accepted=1 response_iteration=9\n"
    )
    assert audit["physical_state_loaded"] is True
    assert audit["signature_match"] is False
    assert audit["minite_iteration_offset"] == 8
    assert audit["objective_history_reused"] is False
    assert audit["fresh_response_iterations"] == [9]


def test_cutoff_sensitive_rank_gain_is_not_mislabeled_inactive():
    study = _study()
    metrics = {
        "rank_gain_by_cutoff": [1, 0, 0],
        "weak_subspace_action_fraction": 0.2,
        "weak_singular_value_lift": 1.0,
        "curvature_inverse_trace_reduction": 0.0,
        "baseline_solver_objective_share": 0.1,
    }
    assert study.classify_family_information(metrics) == (
        "cutoff-sensitive/inconclusive",
    )
    record = {
        "reference": "41672:331",
        "family": "flux_loop",
        "classification": ["cutoff-sensitive/inconclusive"],
        **metrics,
    }
    assert any(
        "needs scientific resolution" in value
        for value in study._family_record_errors(record)
    )


def test_mixed_rank_gain_with_large_lift_is_reinforcing():
    study = _study()
    assert study.classify_family_information(
        {
            "rank_gain_by_cutoff": [1, 1, 0],
            "weak_subspace_action_fraction": 0.2,
            "weak_singular_value_lift": 28.8,
            "curvature_inverse_trace_reduction": 0.0,
            "baseline_solver_objective_share": 0.1,
        }
    ) == ("reinforcing",)


def test_pf_current_is_a_valid_structural_anchor_family():
    study = _study()
    record = {
        "reference": "41672:331",
        "family": "pf_current",
        "classification": ["inactive/redundant", "structural anchor"],
        "rank_gain_by_cutoff": [0, 0, 0],
        "weak_subspace_action_fraction": 0.0,
        "weak_singular_value_lift": 1.0,
        "curvature_inverse_trace_reduction": 0.0,
        "baseline_solver_objective_share": 0.0,
        "pf_structural_response": True,
    }
    assert study._family_record_errors(record) == []


def test_continuation_success_without_case_manifest_is_rejected(tmp_path):
    study = _study()
    with pytest.raises(ValueError, match="case manifest is missing"):
        study._validated_continuation_case(
            {
                "case_id": "fabricated",
                "workdir": str(tmp_path / "absent"),
                "status": "succeeded",
            }
        )


def test_zero_target_failure_can_only_block_its_own_nonzero_chain():
    study = _study()
    zero = {
        "case_id": "zero",
        "reference": "41672:331",
        "chain": "mode_2_zero_target_gate",
        "status": "succeeded",
        "zero_target_gate": True,
        "mode_interpretable": False,
    }
    records = [zero]
    for index, alpha in enumerate((0.125, 0.25)):
        records.append(
            {
                "reference": "41672:331",
                "chain": "mode_2_positive",
                "alpha": alpha,
                "status": "blocked",
                "blocked_by_case_id": "zero",
            }
        )
    assert study._schedule_chain_errors(
        records,
        reference="41672:331",
        chain="mode_2_positive",
        coordinate="alpha",
        requested=(0.125, 0.25),
        allowed_blocker_ids={"zero"},
    ) == []
    records[-1]["blocked_by_case_id"] = "other"
    assert study._schedule_chain_errors(
        records,
        reference="41672:331",
        chain="mode_2_positive",
        coordinate="alpha",
        requested=(0.125, 0.25),
        allowed_blocker_ids={"zero"},
    )


def test_direction_zero_gate_requires_one_row_and_reproduced_truncated_solve():
    study = _study()
    case = {
        "status": "succeeded",
        "displacement": 0.001,
        "exact_constraint_count": 1,
        "solver_method": "truncated_svd_reduced_dgglse_exact_direction",
        "linearization_attributes": {
            "direction_solver_algorithm": "truncated_svd_reduced_dgglse_exact_direction"
        },
        "linear_solve_validation": {
            "singular_value_relative_error": 1e-12,
            "exact_constraint_residual_norm": 1e-15,
            "exact_constraint_tolerance": 1e-13,
        },
        "reduced_linear_solve_validation": {
            "status": 0,
            "transform_relative_error": 1e-12,
            "reproduction_relative_error": 1e-10,
        },
    }
    assert study.validate_direction_zero_target(case, repeat_noise=0.001)["passed"]
    missing_reproduction = {
        **case,
        "reduced_linear_solve_validation": {
            **case["reduced_linear_solve_validation"],
            "reproduction_relative_error": None,
        },
    }
    assert not study.validate_direction_zero_target(
        missing_reproduction, repeat_noise=0.001
    )["passed"]
    assert not study.validate_direction_zero_target(
        {**case, "exact_constraint_count": 2}, repeat_noise=0.001
    )["passed"]
    assert not study.validate_direction_zero_target(
        {**case, "solver_method": "dgglse"}, repeat_noise=0.001
    )["passed"]


def test_diamagnetic_chain_passes_each_successful_restart_to_the_next(monkeypatch, tmp_path):
    study = _study()
    calls = []

    def fake_point(**kwargs):
        parent = kwargs["parent"]
        exponent = kwargs["exponent"]
        calls.append((exponent, parent["case_id"]))
        return {
            "case_id": f"case-{exponent}",
            "status": "succeeded",
            "restart_file": str(tmp_path / f"restart-{exponent}"),
            "snapshot": {"value": exponent},
            "log10_scale": exponent,
            "adjacent_displacement": 0.2,
        }

    monkeypatch.setattr(study, "_diamagnetic_point", fake_point)
    results, parent, complete = study._execute_diamagnetic_chain(
        output=tmp_path,
        reference=study.REFERENCE_SLICES[0],
        baseline={"snapshot": {"value": 0.0}},
        chain="diamagnetic_increasing",
        exponents=(0.25, 0.5),
        initial_parent={
            "case_id": "baseline",
            "status": "succeeded",
            "restart_file": str(tmp_path / "baseline-restart"),
            "snapshot": {"value": 0.0},
        },
        initial_exponent=0.0,
        repeat_noise=0.0,
    )
    assert complete
    assert calls == [(0.25, "baseline"), (0.5, "case-0.25")]
    assert [item["case_id"] for item in results] == ["case-0.25", "case-0.5"]
    assert parent["case_id"] == "case-0.5"


def test_control_record_replacement_preserves_the_other_continuation():
    study = _study()
    merged = study.replace_control_records(
        [
            {"case_id": "old-diamag", "control": "diamagnetic_flux_scale"},
            {"case_id": "direction", "control": "direction"},
        ],
        [{"case_id": "new-diamag", "control": "diamagnetic_flux_scale"}],
        control="diamagnetic_flux_scale",
    )
    assert [item["case_id"] for item in merged] == ["direction", "new-diamag"]


def test_diamagnetic_branch_localization_requires_a_persistent_final_jump(
    monkeypatch, tmp_path
):
    study = _study()
    counter = iter(range(20))

    def fake_point(**kwargs):
        index = next(counter)
        return {
            "case_id": f"middle-{index}",
            "status": "succeeded",
            "snapshot": {"index": index},
        }

    monkeypatch.setattr(study, "_diamagnetic_point", fake_point)
    monkeypatch.setattr(
        study,
        "compare_snapshots",
        lambda *_: (study.TransitionMetrics(lcfs_rms_mm=1.0), False),
    )
    result = study._localize_branch(
        output=tmp_path,
        reference=study.REFERENCE_SLICES[0],
        baseline={},
        lower={"case_id": "low", "snapshot": {"index": -1}},
        upper={"case_id": "high", "snapshot": {"index": 99}},
        lower_exponent=0.0,
        upper_exponent=0.25,
        chain="diamagnetic_increasing",
        repeat_noise=0.0,
    )
    assert result["width_log10"] <= 0.025
    assert result["adjacent_displacement"] == pytest.approx(0.2)
    assert not result["resolved"]


def test_diamagnetic_branch_localization_qualifies_failed_boundary(
    monkeypatch, tmp_path
):
    study = _study()
    counter = iter(range(100))

    def failed_point(**kwargs):
        index = next(counter)
        return {
            "case_id": f"failed-{index}",
            "status": "failed",
            "failed": False,
            "attempt": kwargs.get("attempt", "primary"),
        }

    monkeypatch.setattr(study, "_diamagnetic_point", failed_point)
    localized = study._localize_branch(
        output=tmp_path,
        reference=study.REFERENCE_SLICES[0],
        baseline={},
        lower={"case_id": "low", "status": "succeeded", "snapshot": {}},
        upper={"case_id": "high", "status": "succeeded", "snapshot": {}},
        lower_exponent=0.0,
        upper_exponent=0.25,
        chain="diamagnetic_increasing",
        repeat_noise=0.0,
    )
    assert localized["resolved"]
    assert localized["transition_kind"] == "deterministic_outcome"
    assert localized["width_log10"] <= 0.025
    qualified = localized["_case_records"][-1]
    assert qualified["deterministic_failure_repeats"] == 2
    assert len(qualified["failure_repeat_case_ids"]) == 2


def test_direction_branch_must_persist_through_midpoint_bisection(monkeypatch, tmp_path):
    study = _study()

    def fake_point(**kwargs):
        alpha = kwargs["alpha"]
        return {
            "case_id": f"mid-{alpha}",
            "status": "succeeded",
            "snapshot": {"branch": 10.0 if alpha >= 1.0 else 0.0},
        }

    def fake_compare(left, right):
        jump = abs(left["branch"] - right["branch"])
        return study.TransitionMetrics(lcfs_rms_mm=5.0 * jump), False

    monkeypatch.setattr(study, "_direction_point", fake_point)
    monkeypatch.setattr(study, "compare_snapshots", fake_compare)
    localized = study._localize_direction_branch(
        output=tmp_path,
        reference=study.REFERENCE_SLICES[0],
        baseline={},
        parent={"case_id": "low", "snapshot": {"branch": 0.0}},
        candidate={"case_id": "high", "snapshot": {"branch": 10.0}},
        problem=object(),
        report=object(),
        mode=study.TargetMode(1, "profile", 0.1, "borderline", True, 1, 0, 0),
        parent_alpha=0.0,
        candidate_alpha=2.0,
        chain="mode_1_positive",
        repeat_noise=0.0,
    )
    assert localized["resolved"]
    assert len(localized["case_ids"]) == 3
    assert localized["adjacent_displacement"] == pytest.approx(10.0)


def test_branch_metric_includes_beta_li_profiles_noise_and_rank_switching():
    study = _study()
    metrics = study.TransitionMetrics(
        lcfs_rms_mm=0.5,
        beta_p_relative=0.021,
        li_relative=0.005,
        profile_relative_rms={"pressure": 0.01, "pprime": 0.02},
    )
    result = study.classify_transition(
        metrics,
        repeat_noise=0.001,
        interval_log10=0.02,
        previous_retained_mask=(True, False),
        retained_mask=(False, False),
    )
    assert result.displacement == pytest.approx(1.05)
    assert result.response_onset and result.material_branch and result.rank_switching

    failed = study.classify_transition(
        study.TransitionMetrics(),
        repeat_noise=0.0,
        converged=False,
        bisection_exhausted=True,
        deterministic_failure_repeats=2,
    )
    assert failed.failed and not failed.material_branch


def test_branch_metric_uses_combined_profile_family_rms():
    study = _study()
    metrics = study.TransitionMetrics(
        profile_relative_rms={"pressure": 0.05, "pprime": 0.0}
    )
    assert study.nonlinear_displacement(metrics) == pytest.approx(1 / np.sqrt(2))


def test_snapshot_comparison_rejects_missing_or_nonfinite_physics(monkeypatch):
    study = _study()
    support = SimpleNamespace(
        _curve_distance=lambda *_: {"rms_m": 0.0},
        _relative_rms=lambda left, right, start=0: float(
            np.linalg.norm(np.asarray(left)[start:] - np.asarray(right)[start:])
        ),
    )
    monkeypatch.setattr(study, "_profile_support", lambda: support)
    snapshot = {
        "gfile": {
            "boundary": {"r": [0.2, 0.3, 0.2], "z": [0.0, 0.1, 0.0]},
            "profiles": {
                name: [1.0] * (10 if name == "q" else 3)
                for name in ("pressure", "pprime", "ffprime", "jphi_reference_r", "q")
            },
        },
        "afile": {"scalars": {"area": 1.0, "volume": 1.0, "betap": 1.0, "li": 1.0}},
        "retained_mask": [True],
    }
    study.compare_snapshots(snapshot, snapshot)
    corrupted = json.loads(json.dumps(snapshot))
    corrupted["afile"]["scalars"]["li"] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        study.compare_snapshots(corrupted, snapshot)
    corrupted = json.loads(json.dumps(snapshot))
    del corrupted["gfile"]["profiles"]["pprime"]
    with pytest.raises(ValueError, match="profile 'pprime'"):
        study.compare_snapshots(corrupted, snapshot)


def test_347_assessment_requires_projection_prediction_and_stable_rank():
    study = _study()
    baseline = {
        "analysis": {
            "parameter_scale": [1.0, 1.0, 1.0],
            "modes": [
                {
                    "state": "borderline",
                    "scaled_parameter_direction": [1.0, 0.0, 0.0],
                },
                {
                    "state": "unresolved",
                    "scaled_parameter_direction": [0.0, 1.0, 0.0],
                },
                {
                    "state": "resolved",
                    "scaled_parameter_direction": [0.0, 0.0, 1.0],
                },
            ],
        },
        "parameter_state": [0.0, 0.0, 0.0],
        "snapshot": {"retained_mask": [True, True, False]},
    }

    def point(case_id, exponent, coordinate, mask=(True, True, False)):
        return {
            "case_id": case_id,
            "reference": "41672:347",
            "chain": "diamagnetic_increasing",
            "control": "diamagnetic_flux_scale",
            "status": "succeeded",
            "log10_scale": exponent,
            "parameter_state": [coordinate, 0.0, 0.0],
            "snapshot": {"retained_mask": list(mask)},
            "beta_p_relative": 2.0 * coordinate,
            "li_relative": -coordinate,
            "displacement": 0.2,
            "material_branch": False,
            "response_onset": True,
        }

    records = [
        point("anchor-1", 0.25, 0.1),
        point("anchor-2", 0.5, 0.2),
        point("x1000", 3.0, 0.4),
        point("x10000", 4.0, 0.5),
    ]
    result = study.assess_347_diamagnetic_response(baseline, records)
    assert result["passed"]
    assert result["classification"] == "weak-mode explained"
    assert result["minimum_target_projection_fraction"] == pytest.approx(1.0)
    assert all(
        item["passed"]
        for predictions in result["prediction"].values()
        for item in predictions
    )
    assert result["required_target_coverage"] == {"x1000": True, "x10000": True}

    switched = [*records[:-1], point("x10000-switch", 4.0, 0.5, (True, False, False))]
    result = study.assess_347_diamagnetic_response(baseline, switched)
    assert not result["passed"]
    assert result["classification"] == "truncation/rank switching"


def test_report_keeps_chipasma_outside_the_solve_objective(tmp_path):
    study = _study()
    payload = {
        "stage1_gate": {"passed": True, "reasons": []},
        "family_results": [
            {
                "reference": "41672:331",
                "family": "flux_loop",
                "rank_gain_by_cutoff": [0, 0, 0],
                "weak_subspace_action_fraction": 0.01,
                "weak_singular_value_lift": 1.1,
                "curvature_inverse_trace_reduction": 0.01,
            }
        ],
        "continuation_results": [],
    }
    json_path, markdown_path, plots = study.write_report(payload, tmp_path)
    assert json_path.is_file() and markdown_path.is_file()
    assert {path.name for path in plots} == {
        "conditional_rank_gain.png",
        "diagnostic_vs_solver_objective.png",
    }
    text = markdown_path.read_text(encoding="utf-8")
    assert "Stage-1 gate: **PASS**" in text
    assert "`chipasma` is reported only as an Ip/VCURRT accounting diagnostic" in text


def test_stage2_plan_fails_closed_then_uses_hashed_baseline_restart(
    monkeypatch, tmp_path
):
    study = _study()
    with pytest.raises(RuntimeError, match="requires an existing Stage-1"):
        study.build_stage2_plan(tmp_path)

    cases = []
    modes = [
        {
            "index": 1,
            "singular_value": 2.0,
            "state": "borderline",
            "parameter_role_shares": {"pprime": 0.8},
            "family_shares": {"diamagnetic_flux": 0.1},
        },
        {
            "index": 2,
            "singular_value": 0.5,
            "state": "borderline",
            "parameter_role_shares": {"pf_current": 0.8},
            "family_shares": {"diamagnetic_flux": 0.2},
        },
        {
            "index": 3,
            "singular_value": 0.05,
            "state": "unresolved",
            "parameter_role_shares": {"pf_current": 0.1},
            "family_shares": {"diamagnetic_flux": 0.9},
        },
    ]
    for reference in study.REFERENCE_SLICES:
        case_id = f"case-{reference.key}"
        workdir = tmp_path / case_id
        workdir.mkdir()
        restart = workdir / "esave.dat"
        sidecar = workdir / "response.nc"
        restart.write_bytes(reference.key.encode())
        sidecar.write_bytes(b"sidecar")
        payload = {
            "schema_version": study.SCHEMA_VERSION,
            "issue": study.ISSUE,
            "case_id": case_id,
            "status": "succeeded",
            "reference": {
                "shot": reference.shot,
                "time_ms": reference.time_ms,
                "role": reference.role,
            },
            "workdir": str(workdir),
            "restart_file": str(restart),
            "linearization_file": str(sidecar),
            "analysis": {"nominal_cutoff": 1.0, "modes": modes},
        }
        study.write_json_atomic(workdir / study.STAGE1_CASE_MANIFEST, payload)
        cases.append({"case_id": case_id, "workdir": str(workdir)})
    study.write_json_atomic(
        tmp_path / study.STUDY_MANIFEST,
        {
            "schema_version": study.SCHEMA_VERSION,
            "stage1_gate": {"passed": True, "reasons": []},
            "stage2_authorized": True,
            "cases": cases,
        },
    )
    stage1 = {
        "schema_version": study.SCHEMA_VERSION,
        "issue": study.ISSUE,
        "stage1_gate": {"passed": True, "reasons": []},
        "_manifest_sha256": study.sha256_file(tmp_path / study.STUDY_MANIFEST),
        "_validated_case_payloads": {
            f"{item['reference']['shot']}:{item['reference']['time_ms']}": item
            for item in (
                json.loads(
                    (Path(summary["workdir"]) / study.STAGE1_CASE_MANIFEST).read_text()
                )
                for summary in cases
            )
        },
    }
    monkeypatch.setattr(study, "require_stage2_gate", lambda _output: stage1)
    plan = study.build_stage2_plan(tmp_path)
    assert len(plan["references"]) == 5
    assert plan["restart_validation"]["reference"] == "41672:331"
    assert plan["references"][0]["baseline_restart_sha256"] == study.sha256_file(
        Path(plan["references"][0]["baseline_restart"])
    )
    assert len(plan["references"][0]["diamagnetic_chains"]) == 34
    assert (tmp_path / study.STAGE2_PLAN).is_file()


def test_incompatible_663_evidence_has_a_concrete_instrumented_rerun(monkeypatch, tmp_path):
    study = _study()
    executable = tmp_path / "build" / "efit" / "efit"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"instrumented")
    tables = tmp_path / "tables_primary"
    confirmation_tables = tmp_path / "tables_confirmation"
    tables.mkdir()
    confirmation_tables.mkdir()
    cases = []
    for reference in study.REFERENCE_SLICES:
        case_dir = tmp_path / f"case-{reference.key}"
        case_dir.mkdir()
        kfile = case_dir / f"k0{reference.shot}.{reference.time_ms:05d}"
        selected_tables = tables if reference.shot == 41672 else confirmation_tables
        kfile.write_text(f"TABLE_DIR = '{selected_tables}/'\n", encoding="utf-8")
        study.write_json_atomic(
            case_dir / study.STAGE1_CASE_MANIFEST,
            {
                "reference": {
                    "shot": reference.shot,
                    "time_ms": reference.time_ms,
                    "role": reference.role,
                },
                "kfile": str(kfile),
            },
        )
        cases.append({"workdir": str(case_dir)})
    stage1 = {
        "build_provenance": {
            "executable": {
                "path": str(executable),
                "sha256": study.sha256_file(executable),
            }
        },
        "table_identity": {"source": str(tables)},
        "cases": cases,
    }
    monkeypatch.setattr(study, "require_stage2_gate", lambda _output: stage1)
    monkeypatch.setattr(
        study, "fixed_scientific_config", lambda: SimpleNamespace(sha256="science")
    )
    plan = study.build_issue_663_reproduction_plan(tmp_path)
    assert plan["environment"] == {"EFIT": str(executable)}
    assert set(plan["identity"]["variants"]) == set(
        study.ISSUE_663_REPRODUCTION_VARIANTS
    )
    assert len(plan["invocations"]) == 2
    grouped_shots = sorted(
        tuple(invocation["shots"]) for invocation in plan["invocations"]
    )
    assert grouped_shots == [(39915, 41524), (41672,)]
    assert plan["result"].endswith("constraint_information.json")


def test_663_evidence_rejects_vacuous_passive_agreement(tmp_path):
    study = _study()
    cases = []
    for reference in study.REFERENCE_SLICES:
        kfile = (
            tmp_path
            / f"shot_{reference.shot}"
            / "baseline"
            / "kfile"
            / f"k0{reference.shot}.{reference.time_ms:05d}"
        )
        kfile.parent.mkdir(parents=True, exist_ok=True)
        kfile.write_text(f"slice={reference.key}\n", encoding="utf-8")
        cases.append(
            {
                "reference": {
                    "shot": reference.shot,
                    "time_ms": reference.time_ms,
                    "role": reference.role,
                },
                "identity": {
                    "scientific_sha256": "science",
                    "kfile_sha256": study.sha256_file(kfile),
                },
            }
        )
    variants = {
        name: {"summary": {"plasma_produced": 1}}
        for name in study.ISSUE_663_REPRODUCTION_VARIANTS
    }
    shots = {
        str(shot): {
            "classification": {
                "plasma_current": {"classification": ["accounting-confounded"]}
            },
            "variants": variants,
            "passive_attribution": {
                "agreement": True,
                "comparison": {"geometry": {"common_produced": 0}},
            },
        }
        for shot in (41672, 39915, 41524)
    }
    prior_path = tmp_path / "constraint_information.json"
    study.write_json_atomic(
        prior_path,
        {
            "issue": 663,
            "fixed_scientific_sha256": "science",
            "toolchain": {"efit": {"sha256": "executable"}},
            "table": {},
            "shots": shots,
        },
    )
    audit = study.validate_issue_663_evidence(
        prior_path,
        {
            "cases": cases,
            "build_provenance": {"executable": {"sha256": "executable"}},
            "table_identity": {},
        },
    )
    assert not audit["reusable"]
    assert any("lacks explicit common-slice agreement" in item for item in audit["reasons"])


def test_restart_proof_uses_cold_repeats_and_native_parent(monkeypatch, tmp_path):
    study = _study()
    reference = study.REFERENCE_SLICES[0]
    workdir = tmp_path / "baseline"
    workdir.mkdir()
    executable = tmp_path / "efit"
    kfile = tmp_path / f"k0{reference.shot}.{reference.time_ms:05d}"
    restart = workdir / "esave.dat"
    sidecar = workdir / "response.nc"
    executable.write_bytes(b"efit")
    kfile.write_text("&IN1\n/\n", encoding="utf-8")
    restart.write_bytes(b"restart")
    sidecar.write_bytes(b"sidecar")
    identity = {
        "scientific_sha256": "sci",
        "executable_sha256": study.sha256_file(executable),
        "table_identity": {"sha256": "table"},
        "build_provenance": {"executable": {"path": str(executable)}},
    }
    baseline = {
        "schema_version": study.SCHEMA_VERSION,
        "case_id": "baseline-case",
        "status": "succeeded",
        "reference": {
            "shot": reference.shot,
            "time_ms": reference.time_ms,
            "role": reference.role,
        },
        "workdir": str(workdir),
        "kfile": str(kfile),
        "restart_file": str(restart),
        "linearization_file": str(sidecar),
        "snapshot": {"tag": "baseline", "retained_mask": [True]},
        "restart_control_sha256": "a" * 64,
        "identity": identity,
    }
    study.write_json_atomic(workdir / study.STAGE1_CASE_MANIFEST, baseline)
    study.write_json_atomic(
        tmp_path / study.STUDY_MANIFEST,
        {
            "stage1_gate": {"passed": True, "reasons": []},
            "stage2_authorized": True,
            "cases": [{"case_id": "baseline-case", "workdir": str(workdir)}],
        },
    )
    study.write_json_atomic(tmp_path / study.STAGE2_PLAN, {"restart_validation": {}})
    stage1 = {
        "_manifest_sha256": study.sha256_file(tmp_path / study.STUDY_MANIFEST),
        "_validated_case_payloads": {reference.key: baseline},
    }
    monkeypatch.setattr(study, "require_stage2_gate", lambda _output: stage1)
    monkeypatch.setattr(
        study, "fixed_scientific_config", lambda: SimpleNamespace(sha256="sci")
    )
    calls = []

    def fake_execute(spec, **_):
        calls.append(spec)
        tag = "restart" if spec.restart_parent is not None else spec.kind
        case_workdir = tmp_path / "stage2" / spec.kind
        case_workdir.mkdir(parents=True, exist_ok=True)
        study.write_json_atomic(
            case_workdir / study.STAGE1_CASE_MANIFEST,
            {"case_id": spec.kind, "status": "succeeded"},
        )
        return {
            "case_id": spec.kind,
            "status": "succeeded",
            "snapshot": {"tag": tag},
            "restart_control_sha256": "a" * 64,
            "workdir": str(case_workdir),
        }

    monkeypatch.setattr(study, "execute_nonlinear_case", fake_execute)

    def fake_compare(left, right):
        value = 0.05 if "restart" in {left["tag"], right["tag"]} else 0.01
        return study.TransitionMetrics(lcfs_rms_mm=5.0 * value), False

    monkeypatch.setattr(study, "compare_snapshots", fake_compare)
    proof = study.execute_restart_proof(tmp_path)
    assert proof["passed"]
    assert proof["repeat_noise"] == pytest.approx(0.01)
    assert proof["restart_displacement"] == pytest.approx(0.05)
    assert proof["restart_control_binding_passed"]
    assert len(calls) == 3
    assert calls[-1].restart_parent == restart
