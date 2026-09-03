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


# --- the summary layer, which repairs what is already stored -----------------

def _legacy_ods(b0_values, r0, f_edge, tf_r0=0.4, tf_field=0.060133):
    """An ODS shaped like a VEST database shot: drifting b0, clean f."""
    from omas import ODS

    ods = ODS(consistency_check=False)
    ods["equilibrium.time"] = np.arange(len(b0_values), dtype=float) * 1e-3
    ods["equilibrium.vacuum_toroidal_field.r0"] = r0
    ods["equilibrium.vacuum_toroidal_field.b0"] = np.asarray(b0_values, float)
    for index in range(len(b0_values)):
        ts = ods[f"equilibrium.time_slice.{index}"]
        ts["time"] = index * 1e-3
        ts["profiles_1d.f"] = np.linspace(f_edge * 1.08, f_edge, 9)
    if tf_r0 is not None:
        ods["tf.r0"] = tf_r0
        ods["tf.b_field_tor_vacuum_r.data"] = np.full(4, tf_field)
        ods["tf.time"] = np.arange(4, dtype=float)
    return ods


def test_a_drifting_stored_b0_is_rebuilt_flat_from_f():
    from vaft.database._summary import _normalize_vacuum_field

    # Shot 39915's actual numbers: b0 climbing 73%, f_edge flat.
    drifting = [0.14980, 0.16032, 0.16707, 0.17224, 0.17586, 0.19268, 0.23335, 0.25899]
    ods = _legacy_ods(drifting, r0=0.231317, f_edge=-0.059906)

    _normalize_vacuum_field(ods, 39915)

    b0 = np.asarray(ods["equilibrium.vacuum_toroidal_field.b0"], float).reshape(-1)
    r0 = float(ods["equilibrium.vacuum_toroidal_field.r0"])
    assert r0 == pytest.approx(0.4)                     # tf's, not the back-solve
    assert b0.max() / b0.min() == pytest.approx(1.0)    # the drift is gone
    assert b0[0] * r0 == pytest.approx(0.059906)        # and it is the right field


def test_the_correct_r0_shot_is_repaired_too():
    # 41672 stores the right r0 and a b0 that swings 101% anyway -- the case
    # that rules out r0 as the cause.
    from vaft.database._summary import _normalize_vacuum_field

    ods = _legacy_ods([0.17339, 0.34901], r0=0.4, f_edge=-0.069, tf_field=0.07021)
    _normalize_vacuum_field(ods, 41672)

    b0 = np.asarray(ods["equilibrium.vacuum_toroidal_field.b0"], float).reshape(-1)
    assert b0[0] == pytest.approx(b0[1])
    assert b0[0] == pytest.approx(0.069 / 0.4)


def test_a_healthy_shot_is_left_where_it_was():
    from vaft.database._summary import _normalize_vacuum_field

    ods = _legacy_ods([0.15, 0.15], r0=0.4, f_edge=-0.06, tf_field=0.06)
    _normalize_vacuum_field(ods, 12345)

    b0 = np.asarray(ods["equilibrium.vacuum_toroidal_field.b0"], float).reshape(-1)
    np.testing.assert_allclose(b0, [0.15, 0.15], rtol=1e-12)
    assert float(ods["equilibrium.vacuum_toroidal_field.r0"]) == pytest.approx(0.4)


def test_the_stored_sign_survives_the_rebuild():
    # f is negative and b0 positive on VEST; flipping the column's sign would
    # look like a physics change rather than a repair.
    from vaft.database._summary import _normalize_vacuum_field

    ods = _legacy_ods([-0.15, -0.20], r0=0.4, f_edge=-0.06, tf_field=0.06)
    _normalize_vacuum_field(ods, 12345)

    b0 = np.asarray(ods["equilibrium.vacuum_toroidal_field.b0"], float).reshape(-1)
    assert np.all(b0 < 0)
    assert b0[0] == pytest.approx(-0.15)


def test_without_f_nothing_is_touched():
    from omas import ODS
    from vaft.database._summary import _normalize_vacuum_field

    ods = ODS(consistency_check=False)
    ods["equilibrium.vacuum_toroidal_field.r0"] = 0.231317
    ods["equilibrium.vacuum_toroidal_field.b0"] = np.array([0.15, 0.26])
    ods["equilibrium.time_slice.0.time"] = 0.0
    ods["equilibrium.time_slice.1.time"] = 1e-3

    _normalize_vacuum_field(ods, 12345)

    b0 = np.asarray(ods["equilibrium.vacuum_toroidal_field.b0"], float).reshape(-1)
    np.testing.assert_allclose(b0, [0.15, 0.26], rtol=1e-12)
    assert float(ods["equilibrium.vacuum_toroidal_field.r0"]) == pytest.approx(0.231317)


def test_a_clean_f_does_not_excuse_a_corrupt_r0():
    """``profiles_1d.f`` must not become the basis of the R_0 cross-check.

    It is the better field, which is exactly why it is the wrong test: f is the
    vacuum ``B*R`` whatever reference radius the file names, so it matches ``tf``
    even when ``r0`` is the back-solved 0.2313. Only ``b0*r0`` carries ``r0``
    into the comparison. Swapping one for the other silently disables the
    detection while every other test still passes.
    """
    import vaft.omas as vomas

    ods = _legacy_ods(
        [0.14980, 0.16032, 0.16707, 0.17224, 0.17586, 0.19268, 0.23335, 0.25899],
        r0=0.231317,          # the corruption
        f_edge=-0.059906,     # clean, and agreeing with tf below
        tf_field=0.060133,
    )
    assert vomas.resolve_reference_major_radius(ods) == pytest.approx(0.4)


def test_a_self_consistent_unconventional_r0_is_not_overruled():
    # A machine may legitimately reference its field somewhere other than tf.r0;
    # what marks the VEST corruption is the pair disagreeing about the physics,
    # not r0 differing from tf.r0.
    import vaft.omas as vomas

    ods = _legacy_ods([0.060133 / 0.3] * 3, r0=0.3, f_edge=-0.060133, tf_field=0.060133)
    assert vomas.resolve_reference_major_radius(ods) == pytest.approx(0.3)


def test_both_g_file_readers_take_the_field_from_the_same_place():
    """``to_omas`` and ``as_equilibrium`` must not disagree about B0.

    They used to: one read BCENTR and the other would have kept reading it, so
    a file whose two records differ by round-off alone put the descriptor paths
    3e-9 apart, and a corrupt file would have put them 73% apart. One helper
    feeds both.
    """
    from vaft.process.equilibrium import as_equilibrium

    g = _gfile()
    g["BCENTR"] = float(g["BCENTR"]) * 1.73

    with pytest.warns(RuntimeWarning, match="issue #325"):
        via_ods = as_equilibrium(g.to_omas())
    with pytest.warns(RuntimeWarning, match="issue #325"):
        direct = as_equilibrium(g)

    assert direct.bt0 == pytest.approx(via_ods.bt0, rel=1e-12)
    fpol_edge = float(np.asarray(g["FPOL"]).reshape(-1)[-1])
    assert direct.bt0 * float(g["RCENTR"]) == pytest.approx(fpol_edge, rel=1e-9)
