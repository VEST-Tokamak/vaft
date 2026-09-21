"""EFIT's Picard trajectory as a typed history (issue #1038, phase 1)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.io import netcdf_file

from vaft.code.efit import (
    EFITConfig,
    EFITInputs,
    EFITIterationHistory,
    collect_efit_outputs,
    compare_iteration_histories,
    iteration_history_from_workdir,
    parse_iteration_history,
    parse_slices,
    plot_iteration_convergence,
    read_iteration_history,
    resolved_efit_configuration,
    run_efit,
)
from vaft.code.efit.config import EFITNumericsConfig
from vaft.code.efit.iteration_history import SIDECAR_NAME
from vaft.data.resources import data_path

from external_code_stubs import write_launchable_stub

# Two slices as EFIT prints them with NXITER=1 (format 10019), then a slice
# that collapsed before its first step.
LOG = """\
 r=  0 t=   319 it=  1 chi2=1.53E-08 zm= 2.16E-19 err=1.642E+01 dz= 7.145E-17 chigam= 0.00E+00
 r=  0 t=   319 it=  2 chi2=2.09E-08 zm= 7.77E-19 err=4.877E-01 dz= 5.615E-19 chigam= 0.00E+00
 r=  0 t=   319 it=  3 chi2=2.31E-08 zm= 1.23E-03 err=8.586E-03 dz= 4.534E-19 chigam= 0.00E+00
WARNING in fit at r=  0, t=   319: iconvr=2 satisfied, exiting
 r=  0 t=   320 it=  1 chi2=***** zm= 3.00E-03 err=2.000E+00 dz=-1.000E-02 chigam= 0.00E+00
 r=  0 t=   320 it=  2 chi2=1.00E+02 zm= NaN err=1.000E-01 dz= 0.000E+00 chigam= 0.00E+00
ERROR in bound at r=  0, t=   321: First and last contour points are too far apart
"""


def _mfile(path: Path, time_ms: float, cerror, cchisq, czmaxi_cm) -> Path:
    with netcdf_file(path, "w") as dataset:
        dataset.createDimension("dim_time", 1)
        dataset.createDimension("dim_nitera", len(cerror))
        time = dataset.createVariable("time", "f", ("dim_time",))
        time[:] = [time_ms]
        for name, values in (("cerror", cerror), ("cchisq", cchisq), ("czmaxi", czmaxi_cm)):
            variable = dataset.createVariable(name, "f", ("dim_time", "dim_nitera"))
            variable[:] = [values]
    return path


def _configuration(inner_iterations=None):
    return resolved_efit_configuration(
        EFITConfig(shot=39915, numerics=EFITNumericsConfig(inner_iterations=inner_iterations))
    )


def test_log_only_history_keeps_every_step_and_the_termination():
    history = parse_iteration_history(LOG)

    assert history.times.tolist() == [0.319, 0.32, 0.321]
    first, second, collapsed = history.slices
    assert first.cumulative.tolist() == [1, 2, 3]
    np.testing.assert_allclose(first.error, [16.42, 0.4877, 8.586e-3])
    np.testing.assert_allclose(first.axis_z, [2.16e-19, 7.77e-19, 1.23e-3])
    np.testing.assert_allclose(first.dz, [7.145e-17, 5.615e-19, 4.534e-19])
    assert first.exit_path == "iconvr=2" and first.sources == ("log",)
    # A starred chi2 and a NaN axis are still steps, not a lost slice.
    assert math.isnan(second.chi2[0]) and math.isnan(second.axis_z[1])
    assert second.exit_path == "iterations_exhausted"
    # The slice that failed before its first step is its own, empty slice.
    assert collapsed.iterations_n == 0 and collapsed.exit_path == "solver_error"
    assert collapsed.solver_errors[0]["routine"] == "bound"


def test_the_public_termination_record_is_unchanged():
    # parse_slices is what stored study tables are written against.
    record = parse_slices(LOG)[0]
    assert "iterations" not in record
    assert record["iterations_n"] == 3 and record["gs_error"] == 8.586e-3


def test_outer_and_inner_are_recorded_only_when_nxiter_is_one():
    default = parse_iteration_history(LOG, configuration=_configuration())
    assert default.numbering["nxiter"] == 1
    assert "EFIT default" in default.numbering["nxiter_source"]
    assert [(s.outer, s.inner) for s in default.slices[0].iterations] == [(1, 1), (2, 1), (3, 1)]

    inner = parse_iteration_history(LOG, configuration=_configuration(inner_iterations=3))
    assert inner.numbering["nxiter"] == 3
    assert "not recorded" in inner.numbering["outer_inner"]
    assert all(s.outer is None and s.inner is None for s in inner.slices[0].iterations)
    assert inner.slices[0].cumulative.tolist() == [1, 2, 3]

    unknown = parse_iteration_history(LOG)
    assert unknown.numbering["nxiter"] is None
    assert unknown.slices[0].iterations[0].outer is None


def test_mfile_values_replace_the_printed_ones_and_czmaxi_is_centimetres(tmp_path):
    mfile = _mfile(
        tmp_path / "m039915.00319", 319.0,
        cerror=[16.424274, 0.48768866, 0.0085861],
        cchisq=[1.5339523e-08, 2.0914861e-08, 2.3140037e-08],
        czmaxi_cm=[0.0, 0.0, 0.123],
    )
    history = parse_iteration_history(LOG, mfiles=[mfile])
    item = history.at_time(0.319)

    assert item.sources == ("log", "mfile") and item.mfile == str(mfile)
    np.testing.assert_allclose(item.error, [16.424274, 0.48768866, 0.0085861], rtol=1e-6)
    np.testing.assert_allclose(item.axis_z[-1], 1.23e-3, rtol=1e-6)
    # The log still supplies what the m-file does not carry.
    np.testing.assert_allclose(item.dz, [7.145e-17, 5.615e-19, 4.534e-19])
    assert item.exit_path == "iconvr=2"


def test_a_count_mismatch_keeps_the_log_and_says_so(tmp_path):
    mfile = _mfile(tmp_path / "m039915.00319", 319.0, [1.0, 0.1], [1.0, 1.0], [0.0, 0.0])
    item = parse_iteration_history(LOG, mfiles=[mfile]).at_time(0.319)

    assert item.sources == ("log",)
    np.testing.assert_allclose(item.error, [16.42, 0.4877, 8.586e-3])
    assert "the log's values were kept" in item.notes[0]


def test_a_slice_with_no_printed_step_is_not_given_the_mfile_placeholder(tmp_path):
    # EFIT writes one zero into each array of a slice it ran no Picard step
    # on (seen on 39915 at 331 ms); that zero is not an iteration.
    mfile = _mfile(tmp_path / "m039915.00331", 331.0, [0.0], [0.0], [0.0])
    item = parse_iteration_history(LOG, mfiles=[mfile]).at_time(0.331)

    assert item.iterations_n == 0 and item.exit_path is None
    assert "no Picard step" in item.notes[0]


def test_without_a_log_the_mfile_is_the_trajectory(tmp_path):
    mfile = _mfile(tmp_path / "m039915.00319", 319.0, [2.0, 0.5, 0.01], [3.0, 2.0, 1.0], [1.0, 2.0, 3.0])
    item = parse_iteration_history("", mfiles=[mfile]).at_time(0.319)

    assert item.sources == ("mfile",) and item.cumulative.tolist() == [1, 2, 3]
    np.testing.assert_allclose(item.axis_z, [0.01, 0.02, 0.03], rtol=1e-6)
    assert all(math.isnan(value) for value in item.dz)


def test_at_time_matches_by_time_and_refuses_a_miss():
    history = parse_iteration_history(LOG)
    assert history.at_time(0.3202).time == 0.32
    with pytest.raises(KeyError, match="no slice within"):
        history.at_time(0.35)


def test_json_round_trip_writes_nan_as_null(tmp_path):
    history = parse_iteration_history(LOG, configuration=_configuration())
    path = history.write_json(tmp_path / "history.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"] == "vaft.efit.iteration_history"
    assert payload["slices"][1]["iterations"][0]["chi2"] is None
    back = read_iteration_history(path)
    assert back.to_dict() == history.to_dict()
    assert math.isnan(back.slices[1].chi2[0])
    assert back.provenance["numerics"]["error_tolerance"] == 1e-5


def test_a_newer_schema_is_refused(tmp_path):
    payload = parse_iteration_history(LOG).to_dict()
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="newer"):
        EFITIterationHistory.from_dict(payload)


def test_the_option_accepts_only_what_exists_today():
    assert EFITConfig().iteration_history == "none"
    assert resolved_efit_configuration(EFITConfig(iteration_history="summary"))["execution"][
        "iteration_history"
    ] == "summary"
    # A later phase's level must not silently run as "none".
    with pytest.raises(ValueError, match="phases 2-3"):
        EFITConfig(iteration_history="state")
    with pytest.raises(ValueError, match="must be one of"):
        EFITConfig(iteration_history="everything")


def test_the_option_does_not_change_the_scientific_identity():
    plain = resolved_efit_configuration(EFITConfig(shot=39915))
    recorded = resolved_efit_configuration(EFITConfig(shot=39915, iteration_history="summary"))
    assert plain["scientific_sha256"] == recorded["scientific_sha256"]


def _collect_case(tmp_path, level):
    (tmp_path / "gfile").mkdir()
    (tmp_path / "gfile" / "g039915.00319").write_text(
        data_path("efit/g039915.00319").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "run_efit.out").write_text(LOG, encoding="utf-8")
    return collect_efit_outputs(
        tmp_path, EFITConfig(shot=39915, times=(0.319,), iteration_history=level)
    )


def test_summary_writes_the_sidecar_and_only_a_pointer_into_the_ods(tmp_path):
    result = _collect_case(tmp_path, "summary")

    assert result.iteration_history_file == tmp_path / SIDECAR_NAME
    assert str(result.iteration_history_file) in result.artifact_hashes
    assert result.iteration_history.at_time(0.319).iterations_n == 3
    root = "equilibrium.code.parameters.iteration_history"
    assert result.ods[f"{root}.path"] == str(result.iteration_history_file)
    assert result.ods[f"{root}.level"] == "summary"
    assert result.ods[f"{root}.schema_version"] == 1
    # Iteration is a numerical axis: the physical one is untouched.
    assert list(result.ods["equilibrium.time"]) == [0.319]
    assert read_iteration_history(result.iteration_history_file).to_dict() == (
        result.iteration_history.to_dict()
    )


def test_none_writes_nothing_and_leaves_the_ods_alone(tmp_path):
    result = _collect_case(tmp_path, "none")

    assert result.iteration_history is None and result.iteration_history_file is None
    assert not (tmp_path / SIDECAR_NAME).exists()
    assert "iteration_history" not in result.ods["equilibrium.code.parameters"]


def test_run_efit_reads_this_runs_log_and_drops_a_stale_sidecar(tmp_path):
    from external_code_stubs import RecordingBackend
    from vaft.code.execution import ExecutionResult

    executable = write_launchable_stub(tmp_path / "efit")
    (tmp_path / "kfile").mkdir()
    kfile = tmp_path / "kfile" / "k039915.00319"
    kfile.write_text("input", encoding="utf-8")
    stale = tmp_path / SIDECAR_NAME
    stale.write_text("{}", encoding="utf-8")

    backend = RecordingBackend(ExecutionResult(returncode=0, stdout=LOG, stderr=""))
    base = dict(executable=str(executable), workdir=tmp_path, shot=39915,
                stack_size_kb=None, backend=backend)
    result = run_efit(EFITInputs(tmp_path, kfiles=(kfile,)),
                      EFITConfig(**base, iteration_history="summary"))
    assert result.iteration_history.at_time(0.319).iterations_n == 3
    assert result.iteration_history.provenance["runtime_status"] == "completed"

    stale.write_text("{}", encoding="utf-8")
    quiet = run_efit(EFITInputs(tmp_path, kfiles=(kfile,)), EFITConfig(**base))
    assert quiet.iteration_history is None and not stale.exists()


def test_a_run_that_never_launched_does_not_read_an_old_log(tmp_path):
    (tmp_path / "run_efit.out").write_text(LOG, encoding="utf-8")
    result = collect_efit_outputs(
        tmp_path,
        EFITConfig(shot=39915, iteration_history="summary"),
        runtime_status="runtime_error",
    )
    assert len(result.iteration_history) == 0
    assert result.iteration_history.provenance["log"] is None


def test_history_from_workdir_needs_no_option(tmp_path):
    (tmp_path / "run_efit.out").write_text(LOG, encoding="utf-8")
    _mfile(tmp_path / "m039915.00319", 319.0, [16.4, 0.49, 0.0086], [1e-8, 2e-8, 2e-8], [0.0, 0.0, 0.1])
    history = iteration_history_from_workdir(tmp_path, shot=39915)

    assert history.at_time(0.319).sources == ("log", "mfile")
    assert history.provenance["log"].endswith("run_efit.out")


def test_plots_draw_one_line_per_slice_with_the_run_thresholds():
    import matplotlib

    matplotlib.use("Agg")
    history = parse_iteration_history(LOG, configuration=_configuration())
    figure, axes = plot_iteration_convergence(history)
    assert len(axes.ravel()) == 3
    labels = [line.get_label() for line in axes.ravel()[1].get_lines()]
    assert sum(label.startswith("t=") for label in labels) == 2  # 321 ms has no step
    assert any(label.startswith("ERROR = 1e-05") for label in labels)
    assert any("ERRMIN = 1e-02 (EFIT default" in label for label in labels)

    other = parse_iteration_history(LOG, configuration=_configuration(inner_iterations=3))
    figure, axes = compare_iteration_histories({"NXITER=1": history, "NXITER=3": other}, 0.319)
    labels = [line.get_label() for line in axes.ravel()[0].get_lines()]
    assert labels[:2] == ["NXITER=1: 3 it, iconvr=2", "NXITER=3: 3 it, iconvr=2"]
    with pytest.raises(ValueError, match="no slice"):
        plot_iteration_convergence(history, time=0.321)
    matplotlib.pyplot.close("all")
