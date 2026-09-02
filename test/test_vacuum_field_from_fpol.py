"""The vacuum toroidal field a g-file records twice, and what to do when the
two records disagree.

A g-file states its vacuum field as ``BCENTR`` at ``RCENTR``, and again as
``FPOL`` at the boundary -- ``f = R*B_phi``, which stops varying outside the
plasma because no poloidal current flows there. The two are the same number.

On the VEST shot database they are not: ``BCENTR`` drifts by up to 101% within a
single shot while ``FPOL`` holds flat to under 1%, and shot 41672 stores the
correct ``RCENTR = 0.4`` while its ``BCENTR`` still swings (issue #325). ``FPOL``
is the half the Grad-Shafranov solution actually used, so it wins -- loudly.
"""
import warnings

import numpy as np
import pytest

pytest.importorskip("omas")

from vaft.data import read_geqdsk
from vaft.data.eqdsk import _vacuum_b0
from vaft.data.resources import data_path


def _gfile(name="efit/g039915.00319"):
    return read_geqdsk(data_path(name))


# --- the helper in isolation -------------------------------------------------

def test_a_consistent_g_file_is_left_alone_and_stays_silent():
    data = {"BCENTR": 0.15, "RCENTR": 0.4, "FPOL": np.full(5, 0.06)}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        assert _vacuum_b0(data) == pytest.approx(0.15)


def test_a_drifting_bcentr_loses_to_fpol_and_says_so():
    # The 39915 pathology: FPOL says 0.0599 T.m, BCENTR claims a field 73%
    # larger than that reference radius supports.
    data = {"BCENTR": 0.25899, "RCENTR": 0.4, "FPOL": np.full(5, 0.059906)}
    with pytest.warns(RuntimeWarning, match=r"BCENTR=0\.25899.*disagrees.*FPOL"):
        b0 = _vacuum_b0(data)
    assert b0 == pytest.approx(0.059906 / 0.4)


def test_the_field_and_reference_radius_stay_a_consistent_pair():
    # Nothing inside a g-file can validate RCENTR, so the guarantee is the
    # product, not either member: b0*r0 reproduces FPOL[-1] even when RCENTR is
    # the back-solved 0.2313 the database stores.
    fpol_edge = -0.059906
    for rcentr in (0.4, 0.231317, 0.36917):
        data = {"BCENTR": 0.25899, "RCENTR": rcentr, "FPOL": np.full(5, fpol_edge)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            b0 = _vacuum_b0(data)
        assert b0 * rcentr == pytest.approx(fpol_edge)


def test_the_sign_of_fpol_is_carried_through():
    # f is negative under the stored COCOS on VEST; taking a magnitude here
    # would invent a field direction.
    data = {"BCENTR": -0.15, "RCENTR": 0.4, "FPOL": np.full(5, -0.06)}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _vacuum_b0(data) == pytest.approx(-0.15)


@pytest.mark.parametrize(
    "data",
    [
        {"BCENTR": 0.15, "RCENTR": 0.4},                              # no FPOL
        {"BCENTR": 0.15, "RCENTR": 0.4, "FPOL": np.array([])},        # empty FPOL
        {"BCENTR": 0.15, "RCENTR": 0.4, "FPOL": np.zeros(5)},         # no field
        {"BCENTR": 0.15, "RCENTR": 0.4, "FPOL": np.full(5, np.nan)},  # unusable
        {"BCENTR": 0.15, "RCENTR": 0.0, "FPOL": np.full(5, 0.06)},    # no radius
    ],
)
def test_bcentr_is_kept_without_complaint_when_fpol_cannot_answer(data):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _vacuum_b0(data) == pytest.approx(0.15)


def test_a_missing_bcentr_is_supplied_from_fpol_without_a_disagreement_warning():
    # Nothing to disagree with -- deriving the value is the whole service here,
    # not a correction worth interrupting anyone over.
    data = {"RCENTR": 0.4, "FPOL": np.full(5, 0.06)}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _vacuum_b0(data) == pytest.approx(0.15)


# --- through to_omas, where three leaves depend on it ------------------------

def test_the_packaged_g_files_are_self_consistent_and_convert_unchanged():
    g = _gfile()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        ods = g.to_omas()
    b0 = float(np.asarray(ods["equilibrium.vacuum_toroidal_field.b0"]).flat[0])
    assert b0 == pytest.approx(float(g["BCENTR"]), rel=1e-3)


def test_a_contaminated_bcentr_no_longer_reaches_b0_or_the_axis_field():
    g = _gfile()
    good_b0 = float(g["BCENTR"])
    g["BCENTR"] = good_b0 * 1.73  # the 39915 drift, injected

    with pytest.warns(RuntimeWarning, match="issue #325"):
        ods = g.to_omas()

    eqt = ods["equilibrium.time_slice.0"]
    b0 = float(np.asarray(ods["equilibrium.vacuum_toroidal_field.b0"]).flat[0])
    r0 = float(ods["equilibrium.vacuum_toroidal_field.r0"])
    fpol_edge = float(np.asarray(g["FPOL"]).reshape(-1)[-1])

    # b0 came from FPOL, not from the contaminated BCENTR.
    assert b0 * r0 == pytest.approx(fpol_edge, rel=1e-9)
    assert b0 == pytest.approx(good_b0, rel=1e-3)

    # ...and so did the axis field, which used to be BCENTR*RCENTR/RMAXIS.
    assert eqt["global_quantities.magnetic_axis.b_field_tor"] == pytest.approx(
        fpol_edge / float(g["RMAXIS"]), rel=1e-9
    )


def test_rho_tor_norm_is_indifferent_to_the_contamination_but_rho_tor_is_not():
    # B0 enters rho_tor as 1/sqrt(B0) and cancels in the normalization, so the
    # canonical coordinate from issue #276 was never at risk here -- but the
    # absolute rho_tor was, and is what the fix rescues.
    clean = _gfile()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reference = clean.to_omas()["equilibrium.time_slice.0.profiles_1d"]

    dirty = _gfile()
    dirty["BCENTR"] = float(dirty["BCENTR"]) * 1.73
    with pytest.warns(RuntimeWarning):
        got = dirty.to_omas()["equilibrium.time_slice.0.profiles_1d"]

    np.testing.assert_allclose(
        np.asarray(got["rho_tor_norm"]), np.asarray(reference["rho_tor_norm"]), rtol=1e-12
    )
    if "rho_tor" in reference:
        np.testing.assert_allclose(
            np.asarray(got["rho_tor"]), np.asarray(reference["rho_tor"]), rtol=1e-9
        )
