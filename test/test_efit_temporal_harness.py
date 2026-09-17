"""The cadence/window study harness (issue #468) and the k-file's sub-millisecond time (#468)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "workflow/efit_temporal/run_cadence_study.py"
SPEC = importlib.util.spec_from_file_location("run_cadence_study", SCRIPT)
STUDY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(STUDY)

LOG = """
 r=  0 t=   318 it=  1 chi2=7.09E+02 zm= 1.19E-07 err=3.905E+00 dz= 1.192E-07 chigam= 0.00E+00
 r=  0 t=   318 it=  2 chi2=1.02E-06 zm= 3.86E-03 err=3.517E-01 dz= 2.511E-04 chigam= 0.00E+00
 r=  0 t=   318 it=  1 chi2=6.97E+02 zm= 1.00E-03 err=3.870E+00 dz=-2.853E-03 chigam= 0.00E+00
 r=  0 t=   318 it=  2 chi2=1.19E-06 zm= 3.82E-03 err=3.522E-01 dz= 2.362E-04 chigam= 0.00E+00
 r=  0 t=   318 it=  3 chi2=1.20E-06 zm= 3.82E-03 err=2.000E-01 dz= 2.362E-04 chigam= 0.00E+00
ERROR in bound at r=  0, t=   318: First and last contour points are too far apart
 r=  0 t=   319 it=  1 chi2=7.32E+02 zm= 9.81E-04 err=3.867E+00 dz=-2.799E-03 chigam= 0.00E+00
INFO in efit at r=  0, t=   319: Done processing
"""


def test_iterations_are_split_where_the_counter_restarts_not_on_the_printed_time():
    blocks = STUDY.iterations_from_log(LOG)
    assert [b["iterations_n"] for b in blocks] == [2, 3, 1]
    assert [b["bound_error"] for b in blocks] == [False, True, False]
    assert blocks[1]["gs_error_log"] == 0.2 and blocks[0]["chi2_log"] == 1.02e-6


def test_slice_times_are_snapped_to_the_diagnostics_grid_and_inclusive():
    times = STUDY.slice_times(0.3063, 0.3308, 0.0004)
    assert abs(times[0] - 0.3063) <= STUDY.DIAGNOSTIC_DT and times[-1] <= 0.3308 + STUDY.DIAGNOSTIC_DT
    assert np.allclose(np.diff(times), 0.0004, atol=STUDY.DIAGNOSTIC_DT)
    assert np.allclose(times / STUDY.DIAGNOSTIC_DT, np.round(times / STUDY.DIAGNOSTIC_DT))
    assert STUDY._key_us("k041524.00320_400") == 320400 and STUDY._key_us("k041524.00321") == 321000


def test_a_sub_millisecond_slice_writes_itimeu_and_a_whole_one_does_not(tmp_path):
    """EFIT reads the slice time as ITIME [ms] + ITIMEU [us] and names its
    outputs with both; a 0.4 ms slice must not collapse onto its neighbour."""
    import warnings

    import vaft
    from vaft.code.efit.kfile import generate_constraints_ods, generate_kfile

    ods = vaft.omas.sample_ods()
    times = np.array([0.319, 0.31932])
    ods["equilibrium.time"] = times
    ods["pf_passive.time"] = np.asarray(ods["magnetics.time"], dtype=float)
    for i in range(len(ods["pf_passive.loop"])):
        ods[f"pf_passive.loop.{i}.current"] = np.zeros(ods["pf_passive.time"].size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tables = str(Path(vaft.__file__).parent / "data" / "efit") + "/"
        generate_constraints_ods(
            ods, 39915, str(tmp_path), tables, times,
            [1e-4, 1e-4, 5e-2, 3e-2, 1e-2, 1e-1, 1e-2, 1e-1, 1e-2], [1, 1, 1, 0.1, 0.1, 0.1, 0.01, 0.01],
            broken=[], fit=0,
        )
    generate_kfile(ods, 39915, save_dir=str(tmp_path))
    whole = (tmp_path / "kfile" / "k039915.00319").read_text()
    sub = (tmp_path / "kfile" / "k039915.00319_320").read_text()  # exact microsecond, as EFIT names it
    assert " ITIME = 319\n" in whole and "ITIMEU" not in whole
    assert " ITIME = 319\n" in sub and " ITIMEU = 320\n" in sub
    # The writer's own names decode back to the times it was given.
    from vaft.code.efit.slice_name import split_slice_file_name

    written = sorted(split_slice_file_name(path)[1] for path in (tmp_path / "kfile").glob("k039915.*"))
    assert written == [319000, 319320]


def test_slice_names_round_trip_with_and_without_a_microsecond_part():
    from vaft.code.efit.slice_name import (
        decode_time_suffix,
        encode_time_suffix,
        file_name_microseconds,
        slice_file_name,
        split_slice_file_name,
        time_to_microseconds,
    )

    for seconds, suffix in ((0.306, "00306"), (0.30632, "00306_320"), (0.30601, "00306_010"), (0.3065, "00306_500")):
        microseconds = time_to_microseconds(seconds)
        assert encode_time_suffix(microseconds) == suffix
        assert decode_time_suffix(suffix) == microseconds
        assert decode_time_suffix(suffix) / 1.0e6 == seconds
        assert slice_file_name("g", 39915, seconds) == f"g039915.{suffix}"
        for name in (f"k039915.{suffix}", f"a039915.{suffix}", f"m039915.{suffix}.nc"):
            assert split_slice_file_name(name) == ("039915", microseconds)
    assert file_name_microseconds("run_efit.out") is None
    assert file_name_microseconds("g039915.00306_32x") is None


def test_sub_millisecond_outputs_keep_their_time_and_stay_distinct_cases(tmp_path):
    """cold review efit F1: ``g039915.00306_320`` used to decode to 306.32 s and
    every configured time of one millisecond collapsed onto a single case."""
    from vaft.code.efit import EFITConfig, collect_efit_outputs
    from vaft.data.resources import data_path

    reference = data_path("efit/g039915.00319").read_text(encoding="utf-8")
    (tmp_path / "kfile").mkdir()
    (tmp_path / "gfile").mkdir()
    for suffix in ("00306", "00306_320"):
        (tmp_path / "kfile" / f"k039915.{suffix}").write_text("input", encoding="utf-8")
        (tmp_path / "gfile" / f"g039915.{suffix}").write_text(reference, encoding="utf-8")

    result = collect_efit_outputs(
        tmp_path, EFITConfig(shot=39915, times=(0.306, 0.30601, 0.30632, 0.3065))
    )

    assert [status.time for status in result.slice_statuses] == [0.306, 0.30601, 0.30632, 0.3065]
    assert [status.usable for status in result.slice_statuses] == [True, False, True, False]
    np.testing.assert_array_equal(result.ods["equilibrium.time"], [0.306, 0.30632])
    assert float(result.ods["equilibrium.time_slice.1.time"]) == 0.30632


def test_the_kinetic_base_kfile_is_chosen_by_decoded_time_not_by_float_of_the_suffix():
    """``float("00306_320")`` is 306320.0, so a sub-ms name never won the nearest match."""
    from vaft.code.efit.kinetic import _select_kfile

    kfiles = [Path("k039915.00300"), Path("k039915.00306_320"), Path("k039915.00312")]
    assert _select_kfile(kfiles, 306.3) == Path("k039915.00306_320")
    assert _select_kfile([Path("k039915.00306_320")] + kfiles[:1], 301.0) == Path("k039915.00300")


def test_below_cut_slices_are_counted_against_the_configured_cut_on_the_magnitude(tmp_path, monkeypatch):
    """cold review efit-workflows F6: the count used a signed ``Ip < 50000``
    while the k-file carried ``CUTIP = 15000`` on ``abs(Ip)``, so a 30 kA slice
    EFIT attempted and failed on was excused, and a negative-Ip discharge would
    have had every slice excused."""
    import types

    rows = [{"constraint_ip": 30.0e3}, {"constraint_ip": -80.0e3}, {"constraint_ip": -5.0e3}, {"constraint_ip": None}]
    assert STUDY.count_below_current_cut(rows, 15.0e3) == 1

    times = np.array([0.310, 0.311])
    currents = {310000: 30.0e3, 311000: -80.0e3}
    product = {"equilibrium.time": times}
    for index, value in enumerate(currents.values()):
        product[f"equilibrium.time_slice.{index}.constraints.ip.measured"] = value
    monkeypatch.setattr(STUDY, "generate_constraints_ods", lambda *args, **kwargs: None)
    kfiles = [Path("k039915.00310"), Path("k039915.00311")]
    monkeypatch.setattr(STUDY, "prepare_efit_inputs", lambda ods, config: types.SimpleNamespace(kfiles=kfiles))
    monkeypatch.setattr(
        STUDY,
        "run_efit",
        lambda inputs, config: types.SimpleNamespace(stdout="", returncode=0, status="completed", afiles=[], mfiles=[]),
    )
    import omas

    monkeypatch.setattr(omas, "load_omas_json", lambda *args, **kwargs: product)
    scientific = STUDY.EFITScientificConfig()
    case = STUDY.run_case(
        {}, shot=39915, times=times, window_s=5.0e-4, workdir=tmp_path / "case", efit="/nowhere/efit",
        tables="/nowhere/", scientific=scientific, uncertainty=(), weighting=(),
    )
    assert case["current_cut"] == scientific.initialization.current_threshold == 15.0e3
    assert [row["constraint_ip"] for row in case["slices"]] == [30.0e3, -80.0e3]
    assert case["below_current_cut"] == 0
    assert "below 15 kA" in STUDY.markdown({"cases": {"only": case}})
