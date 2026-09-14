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
