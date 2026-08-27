"""EFIT a-file parsing, mapped from ``EFIT/efit/write_a.f90`` (issue #139).

The a-file states EFIT's own acceptance verdict and its final Grad-Shafranov
error, neither of which appears anywhere else in the pipeline.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from vaft.data import read_aeqdsk
from vaft.data.aeqdsk import (
    AEQDSKError,
    HEAD_SCALARS,
    TAIL_SCALARS,
    TRAILING_SCALARS,
    read_aeqdsk as _read,
)
from vaft.data.eqdsk import read_geqdsk
from vaft.data.resources import data_path

AFILE = "efit/a039915.00319"
PARTNER_GFILE = "efit/g039915.00319"


@pytest.fixture(scope="module")
def afile():
    return read_aeqdsk(data_path(AFILE))


def test_identification_and_status_header(afile):
    assert afile.shot == 39915
    assert afile.time_ms == pytest.approx(319.0)
    assert afile.time_seconds == pytest.approx(0.319)
    assert (afile.jflag, afile.lflag) == (1, 0)
    assert afile.accepted is True
    assert afile.limloc == "IN"
    assert afile.qmflag == "CLC"
    assert (afile.nco2v, afile.nco2r) == (3, 2)
    assert (afile.nlold, afile.nlnew) == (40, 41)
    # write_a.f90 sets fit_type from kprfit/kstark/iconvr; VEST fits magnetics.
    assert afile.fit_type == "MAG"


def test_the_record_count_matches_what_write_a_emits(afile):
    # 24 head scalars, the interferometer arrays sized by the header, then 44.
    expected = len(HEAD_SCALARS) + 2 * afile.nco2v + 2 * afile.nco2r + len(TAIL_SCALARS)
    assert expected == 78
    assert dict(afile.counts) == {"nsilop": 11, "magpri": 64, "nfsum": 16, "nesum": 0}
    assert {name: values.size for name, values in afile.arrays.items()} == {
        "rco2v": 3, "dco2v": 3, "rco2r": 2, "dco2r": 2,
        "csilop": 11, "cmpr2": 64, "ccbrsp": 16, "eccurt": 0,
    }
    assert set(HEAD_SCALARS) | set(TAIL_SCALARS) | set(TRAILING_SCALARS) <= set(
        afile.scalars
    )


def test_the_scalar_positions_are_pinned_by_the_partner_gfile(afile):
    # An independent EFIT artifact for the same slice must agree with them.
    geqdsk = read_geqdsk(data_path(PARTNER_GFILE))
    assert afile["bcentr"] == pytest.approx(float(geqdsk["BCENTR"]), rel=1e-9)
    assert afile["rcencm"] == pytest.approx(100.0 * float(geqdsk["RCENTR"]), rel=1e-9)
    assert afile["ipmhd"] == pytest.approx(float(geqdsk["CURRENT"]), rel=1e-4)
    assert afile["ipmeas"] == pytest.approx(afile["ipmhd"], rel=1e-4)


def test_physically_plausible_values_confirm_the_later_positions(afile):
    # The tail scalars sit past the variable-length interferometer arrays, so a
    # mis-sized header would put every one of these in the wrong slot.
    assert afile.chisq == pytest.approx(77.6181756, rel=1e-9)
    assert afile.terror == pytest.approx(7.693150e-4, rel=1e-6)
    assert 0.0 < afile["li"] < 3.0
    assert 0.0 < afile["betap"] < 5.0
    assert 1.0 < afile["q95"] < 30.0
    assert 0.5 < afile["elongm"] < 3.0
    # Areas and lengths are centimetre-based in the a-file.
    assert 1e3 < afile["area"] < 1e5
    assert afile["wmhd"] > 0.0


def test_terror_is_the_normalized_grad_shafranov_iteration_error(afile):
    # residu(): errorm = max|psi - psi_prev| over the grid / |sidif| / relax,
    # and fit(): terror(jtime) = errorm. Dimensionless and O(1) at worst.
    assert 0.0 < afile.terror < 1.0


def test_negative_values_are_not_split_on_whitespace(afile):
    # Fixed-width e16.9 records: a negative value runs together with the value
    # before it, so whitespace tokenizing silently loses values.
    text = Path(data_path(AFILE)).read_text()
    lines = text.splitlines()
    run_together = [line for line in lines if "E+0" in line and "-0." in line[2:]]
    assert run_together, "the fixture must exercise the run-together case"
    naive = sum(len(line.split()) for line in lines[4:25])
    assert naive < len(HEAD_SCALARS) + 2 * afile.nco2v + 2 * afile.nco2r + len(TAIL_SCALARS)


def test_to_omas_stores_the_verdict_the_error_and_the_arrays(afile):
    ods = afile.to_omas(time_index=2)
    root = "equilibrium.code.parameters.time_slice.2.aeqdsk"
    assert ods[f"{root}.jflag"] == 1
    assert ods[f"{root}.lflag"] == 0
    assert ods[f"{root}.accepted"] == 1
    assert ods[f"{root}.chisq"] == pytest.approx(77.6181756)
    assert ods[f"{root}.terror"] == pytest.approx(7.693150e-4, rel=1e-6)
    assert ods[f"{root}.counts.magpri"] == 64
    assert len(ods[f"{root}.arrays.cmpr2"]) == 64
    assert ods[f"{root}.mapping_source"] == "EFIT/efit/write_a.f90"


def test_accepted_is_not_the_same_claim_as_converged(afile):
    # jflag starts at 1 in write_a and only drops when chkerr objects, so this
    # attribute must not be read as "reached the iteration tolerance".
    assert "accepted" in type(afile).__dict__
    assert not hasattr(afile, "converged")


def test_sentinels_become_nan():
    from vaft.data.aeqdsk import SENTINELS, _clean

    for sentinel in SENTINELS:
        assert math.isnan(_clean(sentinel))
    assert _clean(1.5) == 1.5


def test_unavailable_gaps_read_as_nan(afile):
    # The packaged slice has 1e11 sentinels among the gap measurements.
    assert math.isnan(afile["gapout"]) or math.isfinite(afile["gapout"])
    assert any(math.isnan(value) for value in afile.scalars.values())


@pytest.mark.parametrize(
    "content, match",
    [
        ("one\ntwo\n", "too short"),
        (" x\n 39915  1\n 0.3\n not a header\n 0.1\n", "status header"),
        (" x\n notashot\n 0.3\n * 1 1 0 IN 3 2 CLC 1 1\n 0.1\n", "shot number"),
        (" x\n 39915  1\n 0.3\n * 319.0 1 0 IN\n 0.1\n", "expected 9"),
    ],
)
def test_a_malformed_file_raises_rather_than_returning_wrong_values(
    tmp_path, content, match
):
    path = tmp_path / "a000000.00000"
    path.write_text(content)
    with pytest.raises(AEQDSKError, match=match):
        read_aeqdsk(path)


def test_a_layout_that_does_not_match_write_a_is_refused(tmp_path):
    # The guard that matters: a build whose a-file differs must fail loudly
    # rather than hand back values read from the wrong positions.
    original = Path(data_path(AFILE)).read_text().splitlines()
    truncated = original[:20] + original[25:]
    path = tmp_path / "a000000.00000"
    path.write_text("\n".join(truncated) + "\n")
    with pytest.raises(AEQDSKError, match="write_a.f90 with nco2v"):
        read_aeqdsk(path)


def test_a_wrong_counts_record_is_refused(tmp_path):
    original = Path(data_path(AFILE)).read_text().splitlines()
    original[25] = "    11   32   16    0"  # magpri halved
    path = tmp_path / "a000000.00000"
    path.write_text("\n".join(original) + "\n")
    with pytest.raises(AEQDSKError, match="after the counts record"):
        read_aeqdsk(path)


def test_indexing_reaches_both_scalars_and_arrays(afile):
    assert afile["terror"] == afile.terror
    assert isinstance(afile["cmpr2"], np.ndarray)
    assert "terror" in afile and "cmpr2" in afile
    assert "no_such_field" not in afile
