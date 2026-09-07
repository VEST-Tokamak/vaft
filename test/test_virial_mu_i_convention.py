"""Which side of the mu_i convention boundary each function sits on.

Three review rounds of #546 each found a sign defect in a different function
that touches mu_i, because there are two conventions in this codebase and
nothing said which one a given function expects:

* **volume** -- ``mu_i = <B_tv^2 - B_t^2> / B_pa^2``, positive for a
  diamagnetic plasma. This is what the three virial relations require.
* **flux** (written ``mu_i_hat``) -- ``4*pi*B_t*R_0*dphi / (B_pa^2 * V)`` with
  ``dphi = integral (B_t - B_tv) dA``. This is EFIT's ``xmui``, and it is the
  **negative** of the volume form, because
  ``B_tv^2 - B_t^2 = (F_b-F)(F_b+F)/R^2 ~ -2*F_b*(F-F_b)/R^2``.

The rule, with its one exception, pinned here so a future change that puts a
flux-sign value into a volume-sign consumer fails at this file rather than three
layers downstream:

* every **producer** whose name carries ``hat`` or ``from_phi`` yields the flux
  convention; every other producer yields the volume convention;
* every **consumer** takes the volume convention, except
  ``virial_beta_pd_from_S_mu_rt``, which takes the flux one and says so in its
  Convention section.

Converting a *measured* diamagnetic flux is deliberately not done anywhere:
that needs the sign the measurement carries relative to the stored field, and
VEST data does not settle it -- shot 39915 stores ``f = +0.0598`` in the
packaged sample and ``-0.0598`` in the database with an identical measurement,
and the database declares no COCOS at all.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

import vaft
from vaft.formula.equilibrium import (
    virial_beta_p_lao_from_S_mu_rt,
    virial_beta_pd_from_S_mu_rt,
    virial_full_123_from_S_alpha_rt,
    virial_identity_residuals,
    virial_muihat_from_Bt_R0_dphi,
    virial_pair_12_from_S_mu_rt,
    virial_pair_13_from_S_alpha_mu,
    virial_pair_23_from_S_alpha_mu_rt,
)
from vaft.omas.sample import sample_ods
from vaft.process.equilibrium import computed_diamagnetism_from_phi

#: A consistent (beta_p, li, mu_i) with mu_i in the volume sign, and the
#: Shafranov integrals manufactured from it so all three relations hold exactly.
BETA_P, LI, MU_I, ALPHA, RT = 0.85, 0.72, 0.31, 1.6, 1.05
S2 = (BETA_P + LI + MU_I) / RT
S1 = 3 * BETA_P + LI - MU_I - S2
S3 = BETA_P - (ALPHA - 1.0) * LI - MU_I


# --- the two conventions, and that they are opposites ------------------------


def test_the_flux_form_is_the_negative_of_the_volume_form_on_real_geometry():
    """Not by construction -- against the volume integral the wrapper computes
    from the F profile, which knows nothing about the flux form.

    The two are equal and opposite to first order in (F - F_b)/F_b, so this
    checks the sign exactly and the magnitude loosely.
    """
    row = vaft.omas.compute_virial_equilibrium_quantities_ods(
        sample_ods(), time_slice=0
    )[0]
    volume = row["mui"]                      # exact volume integral
    flux = row["mui_from_flux"]              # EFIT xmui on the same slice
    assert np.isfinite(volume) and np.isfinite(flux)
    assert np.sign(volume) == -np.sign(flux), "the two conventions are opposites"
    assert flux / volume == pytest.approx(-1.0, rel=0.05)


def test_every_closure_takes_the_volume_convention():
    """Feed each closure the consistent volume-sign mu_i: it must recover the
    (beta_p, li) the inputs were built from. With the flux sign it cannot."""
    for name, got in (
        ("pair_12", virial_pair_12_from_S_mu_rt(S1, S2, MU_I, RT)),
        ("pair_13", virial_pair_13_from_S_alpha_mu(S1, S2, S3, ALPHA, MU_I)),
        ("pair_23", virial_pair_23_from_S_alpha_mu_rt(S2, S3, ALPHA, MU_I, RT)),
    ):
        assert got[0] == pytest.approx(BETA_P, abs=1e-12), name
        assert got[1] == pytest.approx(LI, abs=1e-12), name
    # The Lao beta_p is pair_12's, so it is on the same side of the boundary.
    assert virial_beta_p_lao_from_S_mu_rt(S1, S2, MU_I, RT) == pytest.approx(BETA_P, abs=1e-12)


def test_the_full_solve_returns_the_volume_convention():
    """It is handed no mu_i at all, so the sign it returns is the sign the
    relations themselves imply -- the definition of the volume convention."""
    beta_p, li, mu_i = virial_full_123_from_S_alpha_rt(S1, S2, S3, ALPHA, RT)
    assert beta_p == pytest.approx(BETA_P, abs=1e-12)
    assert li == pytest.approx(LI, abs=1e-12)
    assert mu_i == pytest.approx(MU_I, abs=1e-12)


def test_the_diamagnetic_beta_p_is_the_documented_exception():
    """virial_beta_pd_from_S_mu_rt is the one consumer here that takes the
    flux-sign quantity, which its Convention section now states outright. Given
    the flux sign it returns the same beta_p the volume-side closures do; given
    the volume sign it returns beta_p - 2*mu_i, with nothing in the number to
    show it -- which is exactly what happened when its call sites were switched
    and the formula was not."""
    assert virial_beta_pd_from_S_mu_rt(S1, S2, -MU_I, RT) == pytest.approx(BETA_P, abs=1e-12)
    assert virial_beta_pd_from_S_mu_rt(S1, S2, MU_I, RT) == pytest.approx(
        BETA_P - 2 * MU_I, abs=1e-12
    )


def test_the_flux_sign_would_be_caught_at_every_closure():
    """The guard this file exists to be: hand each consumer the wrong-sign
    mu_i and it must not still look right. A 2*mu_i offset is systematic, so
    without this nothing downstream looks random enough to notice."""
    wrong = -MU_I
    assert virial_pair_12_from_S_mu_rt(S1, S2, wrong, RT)[0] != pytest.approx(BETA_P, abs=1e-6)
    assert virial_pair_13_from_S_alpha_mu(S1, S2, S3, ALPHA, wrong)[1] != pytest.approx(LI, abs=1e-6)
    assert virial_pair_23_from_S_alpha_mu_rt(S2, S3, ALPHA, wrong, RT)[0] != pytest.approx(BETA_P, abs=1e-6)
    assert virial_beta_pd_from_S_mu_rt(S1, S2, MU_I, RT) != pytest.approx(BETA_P, abs=1e-6)
    e1, e2, e3 = virial_identity_residuals(BETA_P, LI, wrong, S1, S2, S3, ALPHA, RT)
    assert (e1, e2, e3) == pytest.approx((2 * MU_I, -2 * MU_I, 2 * MU_I), abs=1e-12)


def test_the_hat_named_producers_are_the_flux_side_of_the_boundary():
    """The naming rule, pinned: `hat` and `from_phi` mean flux, and the pair of
    them agree with each other so a caller can substitute one for the other."""
    args = (0.15, 0.4, -1.44e-3, 0.042, 0.956)   # B_t, R0, dphi, B_pa, Omega
    hat = virial_muihat_from_Bt_R0_dphi(*args)
    from_phi = computed_diamagnetism_from_phi(args[2], args[0], args[1], args[4], args[3])
    assert hat == pytest.approx(from_phi, rel=1e-12), "the two flux-side names must agree"
    # virial_beta_pd_from_S_mu_rt is the one consumer on this side of the
    # boundary, and it must agree with the volume-side Lao beta_p on the same
    # physical diamagnetism -- which is what makes the two sides commensurable.
    assert virial_beta_pd_from_S_mu_rt(S1, S2, -MU_I, RT) == pytest.approx(
        virial_beta_p_lao_from_S_mu_rt(S1, S2, MU_I, RT), abs=1e-15
    )


def test_the_input_ods_is_not_mutated_by_the_convention_work():
    before = sample_ods()
    snapshot = copy.deepcopy(before)
    vaft.omas.compute_virial_equilibrium_quantities_ods(before, time_slice=0)
    assert set(before.flat()) >= set(snapshot.flat())




# --- the changes mutation testing found nothing pinning ----------------------


def test_alpha_survives_a_nan_cell_outside_the_plasma():
    """The psi-gradient field fallback marks the R = 0 column NaN on purpose,
    and 0 * nan is nan, so np.sum let one column outside the plasma void alpha
    for a whole slice -- and `alpha_den == 0.0` never fires on a NaN."""
    from vaft.process.equilibrium import efit_virial_volume_integrals, shafranov_integrals

    r = np.linspace(0.0, 2.0, 41)          # includes R = 0
    z = np.linspace(-0.6, 0.6, 31)
    rm, zm = np.meshgrid(r, z, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        b_r = np.where(rm == 0.0, np.nan, 0.2 / np.where(rm == 0.0, np.nan, rm))
    b_z = np.ones_like(rm)
    th = np.linspace(0, 2 * np.pi, 241)
    rb, zb = 1.0 + 0.35 * np.cos(th), 0.30 * np.sin(th)

    terms = efit_virial_volume_integrals(rm, zm, rb, zb, b_r, b_z)
    assert np.isfinite(terms["alpha"]), "one NaN column must not void alpha"
    *_, alpha = shafranov_integrals(
        rb, zb, np.full_like(rb, 0.2), rm, zm, b_r, b_z, B_ref=0.2, volume=1.0
    )
    assert np.isfinite(alpha) and alpha != 0.0


def test_mu_i_excludes_a_nan_cell_rather_than_counting_its_vacuum_term():
    """nan_to_num on the two toroidal terms separately let a cell with a NaN F
    contribute +B_tv^2 instead of nothing."""
    row = vaft.omas.compute_virial_equilibrium_quantities_ods(
        sample_ods(), time_slice=0
    )[0]
    assert np.isfinite(row["mui"])
    # The exact volume integral must equal the closure identity it feeds:
    # full_123 solves for mu_i without being told it, so a mu_i inflated by
    # stray vacuum terms would show up as a gap here.
    assert abs(row["mui"] - row["mu_i_sources"]["volume"]) < 1e-12


def test_the_identity_grade_uses_the_evaluable_subset_not_the_strict_rms():
    """Two statistics, deliberately: `rms` is virial_residual_rms exactly (NaN
    unless all three hold) and `rms_evaluable` is what the grading uses, so E1
    and E3 still count when RT/R0 is undetermined."""
    from vaft.validation.equilibrium import _virial_identity_check

    row = vaft.omas.compute_virial_equilibrium_quantities_ods(
        sample_ods(), time_slice=0
    )[0]
    result = _virial_identity_check(row)
    assert result["status"] == "pass"
    assert np.isfinite(result["rms_evaluable"])
    # E2 is the RT-dependent identity and this slice's RT has cancelled away.
    assert result["e2_excluded_for_rt"] is True
    # The strict rms includes E2 and is a different number; grading it would
    # score the slice on an RT/R0 the rest of the report refuses to use.
    assert result["rms_evaluable"] != pytest.approx(result["rms"], rel=1e-3)
    assert result["rms_evaluable"] < result["rms"]

    # And it must be the graded one. On the sample both land in the same band,
    # so drive a row where they do not: E1 and E3 clean, E2 wild. Grading the
    # strict rms would call this `fail`; grading the evaluable subset, `pass`.
    loud = copy.deepcopy(row)
    loud["identity"] = dict(
        loud["identity"],
        e1_normalized=0.01, e2_normalized=9.0, e3_normalized=0.01,
        rms=float(np.sqrt((0.01**2 + 9.0**2 + 0.01**2) / 3.0)),
        rms_evaluable=0.01,
    )
    assert _virial_identity_check(loud)["status"] == "pass"


def test_plausibility_grades_the_rt_free_closure_not_the_lao_one():
    """Grading the Lao beta_p put this check in contradiction with
    virial_conditioning, which calls those same numbers undetermined. pair_13
    needs no RT/R0, so the check can answer its question."""
    from vaft.validation.equilibrium import _virial_result

    # Slice 5 is the discriminating one: its Lao beta_p is negative (-0.056) so
    # the old rule reported FAIL, while pair_13 is +0.020 and inside the bounds.
    # A test on slice 0, where both are inside, cannot tell the two rules apart.
    row = vaft.omas.compute_virial_equilibrium_quantities_ods(
        sample_ods(), time_slice=5
    )[5]
    assert row["beta_p"] < 0 < row["pair_13"]["beta_p"], "fixture must discriminate"
    result = _virial_result(row)
    assert result["status"] == "pass"
    assert 0 < result["beta_p_pair_13"] <= 10
    # The Lao value is still reported, just not graded on.
    assert result["beta_p"] == pytest.approx(row["beta_p"])


def test_the_report_declares_the_schema_it_actually_has():
    """schema_version must move when the report renames a key or adds checks;
    a consumer keying off it has no other signal."""
    from vaft.validation import validate_equilibrium

    report = validate_equilibrium(sample_ods(), checks="physical_validity", time_slice=0)
    assert report["schema_version"] >= 2
    assert "virial" not in report["physical_validity"]
    assert "virial_parameter_plausibility" in report["physical_validity"]


def test_a_missing_conditioning_flag_is_unknown_not_false():
    """bool(None) is False, which would claim RT/R0 was unavailable when the
    producer simply said nothing."""
    from vaft.validation.equilibrium import _virial_conditioning

    absent = _virial_conditioning({"conditioning": {"denominators": {"pair_23": 1.4}}})
    assert absent["rt_over_r0_available"] is None
    present = _virial_conditioning(
        {"conditioning": {"denominators": {"pair_23": 1.4}, "rt_over_r0_available": False}}
    )
    assert present["rt_over_r0_available"] is False
