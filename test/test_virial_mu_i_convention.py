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

The naming rule the module follows: **a name carrying ``hat`` or ``from_phi``
is the flux convention; everything else is the volume convention.** These tests
pin that rule at every site, so a future change that puts a flux-sign value into
a volume-sign consumer fails here rather than three layers downstream.
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
    virial_mu_i_from_diamagnetic_flux,
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


def test_converting_the_slices_own_flux_reproduces_its_own_mu_i():
    """The self-consistency that a mismatched toroidal field silently breaks.

    mu_i from the volume integral is quadratic in F and so cannot notice a sign
    convention on the field; the conversion from a flux is linear and can. Using
    equilibrium.vacuum_toroidal_field.b0 here put the two on opposite signs on
    210 of 226 rows of the equilibrium history, because f and b0 follow
    different COCOS on VEST (see vaft/database/_summary.py). The conversion must
    use the same F the volume integral was built from.
    """
    rows = vaft.omas.compute_virial_equilibrium_quantities_ods(sample_ods())
    checked = 0
    for row in rows.values():
        if not np.isfinite(row["mui"]) or not np.isfinite(row["f_boundary"]):
            continue
        recovered = virial_mu_i_from_diamagnetic_flux(
            row["f_boundary"] / 1.0, 1.0, row["phi_dia_comp"], row["B_pa"], row["V_p"]
        )
        assert recovered / row["mui"] == pytest.approx(1.0, rel=0.10), (
            "converting a slice's own flux must reproduce its own mu_i"
        )
        checked += 1
    assert checked >= 8


# --- the boundary: which convention each site expects ------------------------


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


def test_the_diamagnetic_beta_p_takes_the_volume_convention_too():
    """virial_beta_pd_from_S_mu_rt was written for the flux sign and its call
    sites were switched to the volume sign without it, which made it return
    beta_p - 2*mu_i. It is the Lao beta_p on a measured mu_i, so on the same
    mu_i the two must agree exactly."""
    assert virial_beta_pd_from_S_mu_rt(S1, S2, MU_I, RT) == pytest.approx(BETA_P, abs=1e-12)
    assert virial_beta_pd_from_S_mu_rt(S1, S2, MU_I, RT) == pytest.approx(
        virial_beta_p_lao_from_S_mu_rt(S1, S2, MU_I, RT), abs=1e-15
    )


def test_the_flux_sign_would_be_caught_at_every_closure():
    """The guard this file exists to be: hand each consumer the wrong-sign
    mu_i and it must not still look right. A 2*mu_i offset is systematic, so
    without this nothing downstream looks random enough to notice."""
    wrong = -MU_I
    assert virial_pair_12_from_S_mu_rt(S1, S2, wrong, RT)[0] != pytest.approx(BETA_P, abs=1e-6)
    assert virial_pair_13_from_S_alpha_mu(S1, S2, S3, ALPHA, wrong)[1] != pytest.approx(LI, abs=1e-6)
    assert virial_pair_23_from_S_alpha_mu_rt(S2, S3, ALPHA, wrong, RT)[0] != pytest.approx(BETA_P, abs=1e-6)
    assert virial_beta_pd_from_S_mu_rt(S1, S2, wrong, RT) != pytest.approx(BETA_P, abs=1e-6)
    e1, e2, e3 = virial_identity_residuals(BETA_P, LI, wrong, S1, S2, S3, ALPHA, RT)
    assert (e1, e2, e3) == pytest.approx((2 * MU_I, -2 * MU_I, 2 * MU_I), abs=1e-12)


def test_the_hat_named_producers_are_the_flux_side_of_the_boundary():
    """The naming rule, pinned: `hat` and `from_phi` mean flux, and the pair of
    them agree with each other so a caller can substitute one for the other."""
    args = (0.15, 0.4, -1.44e-3, 0.042, 0.956)   # B_t, R0, dphi, B_pa, Omega
    hat = virial_muihat_from_Bt_R0_dphi(*args)
    from_phi = computed_diamagnetism_from_phi(args[2], args[0], args[1], args[4], args[3])
    assert hat == pytest.approx(from_phi, rel=1e-12), "the two flux-side names must agree"
    assert virial_mu_i_from_diamagnetic_flux(*args) == pytest.approx(-hat, rel=1e-12)
    # A diamagnetic (negative) flux is a positive mu_i in the volume sign.
    assert args[2] < 0 < virial_mu_i_from_diamagnetic_flux(*args)


def test_the_measured_and_reconstructed_mu_i_are_comparable_on_the_sample():
    """What the whole boundary is for: the two must be on one convention, so a
    difference between them is physics and not bookkeeping. On this sample they
    disagree in sign -- that is #385, and it is only meaningful because both are
    in the volume convention."""
    row = vaft.omas.compute_virial_equilibrium_quantities_ods(
        sample_ods(), time_slice=0
    )[0]
    sources = row["mu_i_sources"]
    assert sources["volume"] < 0, "this reconstruction is paramagnetic"
    assert sources["measured"] > 0, "its loop measures a diamagnetic plasma"
    # Both are half_diff + mu_i, so they differ by exactly the mu_i
    # disagreement and by nothing else -- which is what makes the difference
    # readable as a measurement-vs-reconstruction statement.
    assert row["beta_pd_vir"] - row["beta_p"] == pytest.approx(
        sources["measured"] - sources["volume"], rel=1e-9
    )


def test_the_input_ods_is_not_mutated_by_the_convention_work():
    before = sample_ods()
    snapshot = copy.deepcopy(before)
    vaft.omas.compute_virial_equilibrium_quantities_ods(before, time_slice=0)
    assert set(before.flat()) >= set(snapshot.flat())


def test_the_conversion_does_not_depend_on_the_stored_field_sign():
    """The same shot carries opposite F signs in different sources.

    Shot 39915 has ``profiles_1d.f = +0.0598`` in the packaged sample and
    ``-0.0598`` in the database -- the same magnitude under a different COCOS.
    The volume mu_i is quadratic in F and cannot notice; a conversion linear in
    the field flips, so the same plasma would read diamagnetic from one source
    and paramagnetic from the other. Only the magnitude of the field may enter;
    the sign comes from the flux, whose convention this repository pins in
    test_diamagnetic_flux_sign.py.
    """
    row = vaft.omas.compute_virial_equilibrium_quantities_ods(
        sample_ods(), time_slice=0
    )[0]
    args = (row["f_boundary"], 1.0, row["phi_dia_comp"], row["B_pa"], row["V_p"])
    flipped = (-row["f_boundary"],) + args[1:]
    assert virial_mu_i_from_diamagnetic_flux(*args) == pytest.approx(
        virial_mu_i_from_diamagnetic_flux(*flipped), rel=1e-12
    ), "flipping the stored field sign must not change the answer"
    # ... while flipping the flux, which is the physical measurement, must.
    negated_flux = args[:2] + (-args[2],) + args[3:]
    assert virial_mu_i_from_diamagnetic_flux(*args) == pytest.approx(
        -virial_mu_i_from_diamagnetic_flux(*negated_flux), rel=1e-12
    )


def test_a_diamagnetic_loop_reads_diamagnetic_whatever_the_field_sign():
    """The end-to-end statement: a negative stored flux is a positive volume
    mu_i, for either sign of the stored toroidal field."""
    for b_t in (+0.15, -0.15):
        mu = virial_mu_i_from_diamagnetic_flux(b_t, 0.4, -1.44e-3, 0.042, 0.956)
        assert mu > 0, f"a diamagnetic flux must give a positive mu_i (B_t={b_t})"
