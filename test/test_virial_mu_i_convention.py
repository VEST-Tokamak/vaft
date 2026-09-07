"""Which mu_i convention each virial function takes.

There are two in this codebase, and they are negatives of each other:

* **volume** -- ``mu_i = <B_tv^2 - B_t^2> / B_pa^2``, positive for a diamagnetic
  plasma. This is what the three virial relations require.
* **flux** (written ``mu_i_hat``) -- ``4*pi*B_t*R_0*dphi / (B_pa^2 * V)`` with
  ``dphi = integral (B_t - B_tv) dA``. This is EFIT's ``xmui``, and it is the
  negative of the volume form because
  ``B_tv^2 - B_t^2 = (F_b-F)(F_b+F)/R^2 ~ -2*F_b*(F-F_b)/R^2``.

The naming rule: **a name carrying ``hat`` or ``from_phi`` is the flux
convention; everything else is the volume convention.** Nothing said this
before, so the pipeline fed the flux quantity to closures that need the volume
one, putting a systematic ``2*mu_i`` into every identity residual and turning
every closure's internal inductance negative on real reconstructions.

Every assertion here is against a quantity computed some other way -- an
identity the relations imply, or a value the wrapper derives independently --
never against a restatement of the formula under test. Re-typing a formula only
proves the code matches the typing, which is how the sign survived.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

import vaft
from vaft.formula.equilibrium import (
    virial_beta_p_lao_from_S_mu_rt,
    virial_bongard_from_S_alpha_mu,
    virial_identity_residuals,
    virial_muihat_from_Bt_R0_dphi,
)
from vaft.omas.sample import sample_ods
from vaft.process.equilibrium import computed_diamagnetism_from_phi

#: A consistent (beta_p, li, mu_i) with mu_i in the volume sign, and the
#: Shafranov integrals manufactured from it so all three relations hold exactly.
BETA_P, LI, MU_I, ALPHA, RT = 0.85, 0.72, 0.31, 1.6, 1.05
S2 = (BETA_P + LI + MU_I) / RT
S1 = 3 * BETA_P + LI - MU_I - S2
S3 = BETA_P - (ALPHA - 1.0) * LI - MU_I


def test_the_closures_take_the_volume_convention():
    """Fed the volume mu_i the closures recover the (beta_p, li) their inputs
    were built from; fed the flux sign they cannot, and the error is a
    systematic 2*mu_i rather than anything that looks like noise."""
    beta_p, li = virial_bongard_from_S_alpha_mu(S1, S2, S3, ALPHA, MU_I)
    assert beta_p == pytest.approx(BETA_P, abs=1e-12)
    assert li == pytest.approx(LI, abs=1e-12)
    assert virial_beta_p_lao_from_S_mu_rt(S1, S2, MU_I, RT) == pytest.approx(BETA_P, abs=1e-12)

    wrong_beta, wrong_li = virial_bongard_from_S_alpha_mu(S1, S2, S3, ALPHA, -MU_I)
    assert wrong_beta != pytest.approx(BETA_P, abs=1e-6)
    assert wrong_li != pytest.approx(LI, abs=1e-6)
    assert virial_identity_residuals(
        BETA_P, LI, -MU_I, S1, S2, S3, ALPHA, RT
    ) == pytest.approx((2 * MU_I, -2 * MU_I, 2 * MU_I), abs=1e-12)


def test_the_hat_named_producers_are_the_flux_side():
    """The rule, pinned: the two flux-side names agree with each other, so a
    caller may substitute one for the other, and neither is the volume form."""
    args = (0.15, 0.4, -1.44e-3, 0.042, 0.956)   # B_t, R0, dphi, B_pa, Omega
    hat = virial_muihat_from_Bt_R0_dphi(*args)
    from_phi = computed_diamagnetism_from_phi(args[2], args[0], args[1], args[4], args[3])
    assert hat == pytest.approx(from_phi, rel=1e-12)


def test_the_wrapper_reports_both_conventions_under_names_that_say_which():
    """`mui` is the volume integral the relations need; `mui_hat` keeps the flux
    quantity it has always carried. They must be opposite in sign, and close in
    magnitude -- the flux form is the volume one to first order in (F-F_b)/F_b."""
    row = vaft.omas.compute_virial_equilibrium_quantities_ods(sample_ods(), time_slice=0)[0]
    volume, flux = row["mui"], row["mui_hat"]
    assert np.isfinite(volume) and np.isfinite(flux)
    assert np.sign(volume) == -np.sign(flux)
    assert flux / volume == pytest.approx(-1.0, rel=0.05)
    assert row["mui_from_flux"] == pytest.approx(flux, rel=1e-15)


def test_the_closures_agree_with_the_volume_integrals_on_the_sample():
    """The check that fails loudly if the wrapper hands the closures the wrong
    convention: beta_p and li from the RT-free Bongard closure must land on the
    volume integrals the wrapper computes by a completely separate route. With
    the flux sign the closure's l_i comes out negative against a positive
    volume l_i."""
    rows = vaft.omas.compute_virial_equilibrium_quantities_ods(sample_ods())
    checked = 0
    for row in rows.values():
        li_volume = row.get("volume", {}).get("li")
        li_closure = row.get("li_vir_bongard")
        if not (np.isfinite(li_volume) and np.isfinite(li_closure)):
            continue
        assert li_closure > 0, "a physical equilibrium has positive internal inductance"
        assert li_closure == pytest.approx(li_volume, rel=0.15), (
            f"closure l_i {li_closure} should track the volume integral {li_volume}"
        )
        checked += 1
    assert checked >= 8


def test_alpha_survives_a_nan_cell_outside_the_plasma():
    """The psi-gradient field fallback marks the R = 0 column NaN on purpose,
    and 0 * nan is nan, so np.sum let one column outside the plasma void alpha
    for a whole slice -- and `alpha_den == 0.0` never fires on a NaN."""
    from vaft.process.equilibrium import efit_virial_volume_integrals, shafranov_integrals

    r = np.linspace(0.0, 2.0, 41)          # includes R = 0
    z = np.linspace(-0.6, 0.6, 31)
    rm, zm = np.meshgrid(r, z, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        b_r = 0.2 / np.where(rm == 0.0, np.nan, rm)
    b_z = np.ones_like(rm)
    th = np.linspace(0, 2 * np.pi, 241)
    rb, zb = 1.0 + 0.35 * np.cos(th), 0.30 * np.sin(th)

    assert np.isnan(b_r).any(), "fixture must actually carry a NaN column"
    assert np.isfinite(efit_virial_volume_integrals(rm, zm, rb, zb, b_r, b_z)["alpha"])
    *_, alpha = shafranov_integrals(
        rb, zb, np.full_like(rb, 0.2), rm, zm, b_r, b_z, B_ref=0.2, volume=1.0
    )
    assert np.isfinite(alpha) and alpha != 0.0


def test_mu_i_excludes_a_nan_cell_rather_than_counting_its_vacuum_term():
    """nan_to_num on the two toroidal terms separately lets a cell with a NaN F
    contribute +B_tv^2 instead of nothing.

    This drives the wrapper rather than re-implementing the arithmetic beside
    it: a version that built both forms itself passed under the mutation it
    names, because nothing it asserted ran the code being mutated.
    """
    ods = copy.deepcopy(sample_ods())
    clean = vaft.omas.compute_virial_equilibrium_quantities_ods(
        copy.deepcopy(ods), time_slice=0
    )[0]["mui"]
    f = np.asarray(ods["equilibrium.time_slice.0.profiles_1d.f"], float).copy()
    f[len(f) // 2] = np.nan          # a NaN band in F_2d, inside the plasma
    ods["equilibrium.time_slice.0.profiles_1d.f"] = f
    holed = vaft.omas.compute_virial_equilibrium_quantities_ods(ods, time_slice=0)[0]["mui"]

    assert np.isfinite(holed), "a NaN band must not void mu_i"
    # Excluding those cells barely moves mu_i. Counting their vacuum term alone
    # moves it a lot, and toward the positive vacuum term against a negative
    # mu_i, so the magnitude collapses.
    assert abs(holed - clean) < 0.05 * abs(clean), f"{clean} -> {holed}"
