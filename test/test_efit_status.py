import json
import subprocess

import numpy as np

from vaft.code.efit import (
    EFIT_FAILURE_CODES,
    EFITConfig,
    EFITInputs,
    EFITSliceStatus,
    EFITValidationConfig,
    apply_temporal_continuity,
    collect_efit_outputs,
    run_efit,
    validate_efit_slice,
)
from vaft.data.resources import data_path

from external_code_stubs import write_launchable_stub


def _valid_geqdsk(**updates):
    equilibrium = {
        "SIMAG": -0.1,
        "SIBRY": 0.2,
        "RMAXIS": 0.5,
        "ZMAXIS": 0.0,
        "CURRENT": 50_000.0,
        "RBBBS": np.array([0.2, 0.8, 0.8, 0.2]),
        "ZBBBS": np.array([-0.4, -0.4, 0.4, 0.4]),
        "PRES": np.array([10.0, 5.0, 0.0]),
        "PPRIME": np.array([-2.0, -1.0, 0.0]),
        "FFPRIM": np.array([-0.2, -0.1, 0.0]),
        "QPSI": np.array([1.0, 1.5, 2.0]),
    }
    equilibrium.update(updates)
    return equilibrium


def _output_files(tmp_path):
    paths = {}
    for kind in ("kfile", "gfile"):
        path = tmp_path / kind[0]
        path.write_text(kind, encoding="utf-8")
        paths[kind] = path
    return paths


def test_zero_returncode_without_gfile_is_not_usable(tmp_path):
    kfile = tmp_path / "k039915.00319"
    kfile.touch()

    status = validate_efit_slice(
        shot=39915,
        time=0.319,
        runtime_status="completed",
        returncode=0,
        kfile=kfile,
        gfile=None,
    )

    assert status.runtime_ok
    assert not status.output_ok
    assert not status.usable
    assert status.failure_codes == ("missing_gfile",)


def test_process_ok_remains_backward_compatible_but_usable_requires_output(tmp_path):
    kdir = tmp_path / "kfile"
    kdir.mkdir()
    (kdir / "k039915.00319").touch()

    result = collect_efit_outputs(
        tmp_path,
        EFITConfig(shot=39915),
        returncode=0,
        runtime_status="completed",
    )

    assert result.ok
    assert not result.usable
    assert result.slice_statuses[0].failure_codes == ("missing_gfile",)


def test_negative_pressure_is_distinct_from_numerical_failure(tmp_path):
    paths = _output_files(tmp_path)

    status = validate_efit_slice(
        shot=39915,
        time=0.319,
        runtime_status="completed",
        returncode=0,
        geqdsk=_valid_geqdsk(PRES=np.array([2.0, -0.5, 0.0])),
        **paths,
    )

    assert status.runtime_ok and status.output_ok and status.numerical_ok
    assert not status.physical_ok
    assert status.overall_status == "physical_failed"
    assert status.failure_codes == ("negative_pressure",)


def test_optional_a_and_m_files_are_not_required_by_default(tmp_path):
    paths = _output_files(tmp_path)

    optional = validate_efit_slice(
        shot=39915,
        time=0.319,
        runtime_status="collected",
        returncode=None,
        geqdsk=_valid_geqdsk(),
        **paths,
    )
    required = validate_efit_slice(
        shot=39915,
        time=0.319,
        runtime_status="collected",
        returncode=None,
        geqdsk=_valid_geqdsk(),
        config=EFITValidationConfig(require_afile=True, require_mfile=True),
        **paths,
    )

    assert optional.usable
    assert "missing_afile" not in optional.failure_codes
    assert "missing_mfile" not in optional.failure_codes
    assert {"missing_afile", "missing_mfile"} <= set(required.failure_codes)


def test_failure_taxonomy_is_exercised_by_synthetic_statuses(tmp_path):
    paths = _output_files(tmp_path)
    observed = set()

    def capture(**kwargs):
        arguments = {
            "shot": 39915,
            "time": 0.319,
            "runtime_status": "completed",
            "returncode": 0,
            "geqdsk": _valid_geqdsk(),
            **paths,
        }
        arguments.update(kwargs)
        status = validate_efit_slice(**arguments)
        observed.update(status.failure_codes)
        return status

    capture(runtime_status="timeout", returncode=None)
    capture(returncode=2)
    capture(kfile=None)
    capture(gfile=None)
    capture(config=EFITValidationConfig(require_afile=True, require_mfile=True))
    capture(parse_error="invalid g-file")
    capture(converged=False)
    capture(geqdsk=_valid_geqdsk(RBBBS=[], ZBBBS=[], SIMAG=0.0, SIBRY=0.0))
    capture(geqdsk=_valid_geqdsk(PPRIME=[np.nan], QPSI=[0.0]))
    capture(
        geqdsk=_valid_geqdsk(PRES=[-1.0]),
        metrics={
            "stored_energy": -1.0,
            "li": -1.0,
            "beta": -1.0,
            "diagnostic_residual": 2.0,
        },
        config=EFITValidationConfig(
            volume_range=(0.0, 0.1), maximum_diagnostic_residual=1.0
        ),
    )

    first = capture()
    second = validate_efit_slice(
        shot=39915,
        time=0.320,
        runtime_status="completed",
        returncode=0,
        geqdsk=_valid_geqdsk(RMAXIS=0.9),
        **paths,
    )
    observed.update(
        apply_temporal_continuity(
            (first, second), EFITValidationConfig(maximum_axis_step=0.1)
        )[1].failure_codes
    )

    assert observed == set(EFIT_FAILURE_CODES)


def test_status_json_round_trip(tmp_path):
    paths = _output_files(tmp_path)
    status = validate_efit_slice(
        shot=39915,
        time=0.319,
        runtime_status="completed",
        returncode=0,
        geqdsk=_valid_geqdsk(),
        provenance={
            "executable": "/opt/efit/bin/efit",
            "workdir": tmp_path,
        },
        **paths,
    )

    payload = json.loads(json.dumps(status.to_dict()))

    assert EFITSliceStatus.from_dict(payload) == status


def test_collection_preserves_partial_success_and_time_alignment(tmp_path):
    (tmp_path / "kfile").mkdir()
    (tmp_path / "gfile").mkdir()
    reference = data_path("efit/g039915.00319").read_text(encoding="utf-8")
    for suffix in ("00319", "00320"):
        (tmp_path / "kfile" / f"k039915.{suffix}").write_text("input", encoding="utf-8")
    (tmp_path / "gfile" / "g039915.00319").write_text(reference, encoding="utf-8")
    (tmp_path / "gfile" / "g039915.00320").write_text("not a g-file", encoding="utf-8")

    result = collect_efit_outputs(
        tmp_path, EFITConfig(shot=39915, times=(0.319, 0.320, 0.321))
    )

    assert [status.time for status in result.slice_statuses] == [0.319, 0.32, 0.321]
    assert result.slice_statuses[0].usable
    assert not result.slice_statuses[1].usable
    assert "parse_error" in result.slice_statuses[1].failure_codes
    assert {"missing_kfile", "missing_gfile"} <= set(
        result.slice_statuses[2].failure_codes
    )
    np.testing.assert_allclose(result.ods["equilibrium.time"], [0.319])
    assert result.usable


def test_timeout_has_slice_status_and_attempt_logs(tmp_path, monkeypatch):
    executable = write_launchable_stub(tmp_path / "efit")
    kdir = tmp_path / "kfile"
    kdir.mkdir()
    kfile = kdir / "k039915.00319"
    kfile.write_text("input", encoding="utf-8")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1.0)

    monkeypatch.setattr(subprocess, "run", timeout)
    config = EFITConfig(
        executable=str(executable),
        workdir=tmp_path,
        shot=39915,
        timeout=1.0,
        stack_size_kb=None,
    )

    result = run_efit(EFITInputs(tmp_path, kfiles=(kfile,)), config)

    assert result.status == "failed"
    assert result.slice_statuses[0].overall_status == "runtime_failed"
    assert {"runtime_error", "timeout"} <= set(result.slice_statuses[0].failure_codes)
    assert {path.name for path in result.logs} == {"run_efit.err", "run_efit.out"}


def test_skipped_run_does_not_collect_stale_outputs(tmp_path, monkeypatch):
    monkeypatch.delenv("EFITHOME", raising=False)
    monkeypatch.delenv("EFIT", raising=False)
    kdir = tmp_path / "kfile"
    gdir = tmp_path / "gfile"
    kdir.mkdir()
    gdir.mkdir()
    kfile = kdir / "k039915.00319"
    kfile.write_text("input", encoding="utf-8")
    (gdir / "g039915.00319").write_text(
        data_path("efit/g039915.00319").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    result = run_efit(
        EFITInputs(tmp_path, kfiles=(kfile,)),
        EFITConfig(workdir=tmp_path, shot=39915),
    )

    assert result.status == "skipped"
    assert result.gfiles == ()
    assert result.geqdsk == ()
    assert result.ods is None
    assert result.slice_statuses[0].overall_status == "runtime_failed"


def test_continuity_compares_across_a_failed_middle_slice(tmp_path):
    paths = _output_files(tmp_path)
    first = validate_efit_slice(
        shot=39915,
        time=0.1,
        runtime_status="completed",
        returncode=0,
        geqdsk=_valid_geqdsk(RMAXIS=0.4),
        **paths,
    )
    failed = validate_efit_slice(
        shot=39915,
        time=0.2,
        runtime_status="completed",
        returncode=0,
        kfile=paths["kfile"],
        gfile=None,
    )
    third = validate_efit_slice(
        shot=39915,
        time=0.3,
        runtime_status="completed",
        returncode=0,
        geqdsk=_valid_geqdsk(RMAXIS=0.8),
        **paths,
    )

    statuses = apply_temporal_continuity(
        (first, failed, third), EFITValidationConfig(maximum_axis_step=0.1)
    )

    assert "temporal_discontinuity" in statuses[2].failure_codes


def test_early_time_file_names_match_configured_slice(tmp_path):
    (tmp_path / "kfile").mkdir()
    (tmp_path / "gfile").mkdir()
    (tmp_path / "kfile" / "k039915.0050").write_text("input", encoding="utf-8")
    (tmp_path / "gfile" / "g039915.0050").write_text(
        data_path("efit/g039915.00319").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    result = collect_efit_outputs(
        tmp_path, EFITConfig(shot=39915, times=(0.05,))
    )

    assert len(result.slice_statuses) == 1
    assert result.slice_statuses[0].time == 0.05
    assert result.slice_statuses[0].usable
