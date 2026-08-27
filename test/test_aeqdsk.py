"""EFIT a-file parsing (issue #139).

The a-file states EFIT's own convergence verdict, which nothing else in the
pipeline provides. Only positions verified against a real file and its partner
g-file are parsed into named fields; see the module docstring for why the rest
is preserved verbatim instead of being read from a guessed index.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from vaft.data import read_aeqdsk
from vaft.data.aeqdsk import AEQDSK, AEQDSKError
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
    assert afile.converged is True
    assert afile.limloc == "IN"
    assert afile.qmflag == "CLC"
    assert (afile.mco2v, afile.mco2r) == (3, 2)
    assert (afile.nlold, afile.nlnew) == (40, 41)


def test_first_scalar_record_matches_the_partner_gfile(afile):
    # This is what pins the positions: an independent EFIT artifact for the same
    # slice must agree with them.
    geqdsk = read_geqdsk(data_path(PARTNER_GFILE))
    assert afile.bcentr == pytest.approx(float(geqdsk["BCENTR"]), rel=1e-9)
    assert afile.rcencm == pytest.approx(100.0 * float(geqdsk["RCENTR"]), rel=1e-9)
    assert afile.cpasma == pytest.approx(float(geqdsk["CURRENT"]), rel=1e-4)
    assert afile.pasmat == pytest.approx(afile.cpasma, rel=1e-4)
    # Total chi-square, O(degrees of freedom) for a fit consistent with its
    # assigned uncertainties.
    assert afile.tsaisq == pytest.approx(77.6181756, rel=1e-9)


def test_the_unverified_remainder_is_preserved_not_guessed(afile):
    assert len(afile.raw_scalars) == 78
    assert afile.raw_counts == (11, 64, 16, 0)
    assert len(afile.raw_trailing) == 151
    # The mainline ordering predicts 82 scalars and 91 trailing values for these
    # counts. It does not describe this build, which is why nothing past the
    # first record is read positionally.
    assert len(afile.raw_scalars) != 82
    assert len(afile.raw_trailing) != sum(afile.raw_counts)


def test_negative_values_are_not_split_on_whitespace(afile):
    # Fixed-width e16.9 fields: a negative value runs together with the value
    # before it, so whitespace tokenizing loses values.
    text = Path(data_path(AFILE)).read_text()
    run_together = [line for line in text.splitlines() if "E+0" in line and "-0." in line[2:]]
    assert run_together, "the fixture must exercise the run-together case"
    naive = sum(len(line.split()) for line in text.splitlines()[4:25])
    assert naive < len(afile.raw_scalars)


def test_to_omas_writes_the_verdict_and_keeps_the_remainder(afile):
    ods = afile.to_omas(time_index=2)
    root = "equilibrium.code.parameters.time_slice.2.aeqdsk"
    assert ods[f"{root}.jflag"] == 1
    assert ods[f"{root}.lflag"] == 0
    assert ods[f"{root}.converged"] == 1
    assert ods[f"{root}.tsaisq"] == pytest.approx(77.6181756)
    assert ods[f"{root}.cpasma"] == pytest.approx(77040.7487, rel=1e-6)
    assert len(ods[f"{root}.raw.scalars"]) == 78
    assert ods[f"{root}.raw.layout_verified_through"] == "cpasma"


def test_sentinels_become_nan():
    from vaft.data.aeqdsk import SENTINELS, _clean

    for sentinel in SENTINELS:
        assert math.isnan(_clean(sentinel))
    assert _clean(1.5) == 1.5


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


def test_a_file_without_enough_scalars_raises(tmp_path):
    path = tmp_path / "a000000.00000"
    path.write_text(
        " hdr\n 39915  1\n 0.3\n * 319.000 1 0 IN 3 2 CLC 40 41\n"
        "  0.100000000E+01 0.200000000E+01\n"
    )
    with pytest.raises(AEQDSKError, match="expected at least the first 5"):
        read_aeqdsk(path)


def test_converged_needs_both_flags():
    def build(jflag, lflag):
        return AEQDSK(
            shot=1, time_ms=1.0, jflag=jflag, lflag=lflag, limloc="IN", mco2v=0,
            mco2r=0, qmflag="CLC", nlold=0, nlnew=0, tsaisq=1.0, rcencm=40.0,
            bcentr=0.1, pasmat=1.0, cpasma=1.0,
        )

    assert build(1, 0).converged is True
    assert build(0, 0).converged is False
    assert build(1, 2).converged is False
