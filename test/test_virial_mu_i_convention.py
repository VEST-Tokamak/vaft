"""Which mu_i convention each virial function takes.

There are two in this codebase, and they are negatives of each other:

* **volume** -- ``mu_i = <B_tv^2 - B_t^2> / B_pa^2``, positive for a diamagnetic
  plasma. This is what the three virial relations require.
* **flux** (written ``mu_i_hat``) -- ``4*pi*B_t*R_0*dphi / (B_pa^2 * V)`` with
  ``dphi = integral (B_t - B_tv) dA``. This is EFIT's ``xmui``, and it is the
  negative of the volume form because
  ``B_tv^2 - B_t^2 = (F_b-F)(F_b+F)/R^2 ~ -2*F_b*(F-F_b)/R^2``.

The rule, with its one exception:

* every **producer** whose name carries ``hat`` or ``from_phi`` yields the flux
  convention; every other producer yields the volume convention;
* every **consumer** takes the volume convention, except
  ``virial_beta_pd_from_S_mu_rt``, which takes the flux one and now says so in
  its Convention section.

Nothing said any of this before, so the pipeline fed the flux quantity to
closures that need the volume one, putting a systematic ``2*mu_i`` into every
identity residual and turning every closure's internal inductance negative on
real reconstructions.

Where a test can be, it is written against a quantity computed some other way
-- a value the wrapper derives independently, or the volume integral taken by a
separate route -- rather than against a restatement of the formula under test.
Re-typing a formula only proves the code matches the typing, which is how the
sign survived. `test_the_closures_take_the_volume_convention` is the exception
and is honest about it: its inputs are manufactured from E1/E2/E3, so it is
pure algebra. The physics is carried by
`test_the_closures_agree_with_the_volume_integrals_on_the_sample`.
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
    """The rule, pinned against a volume quantity computed a different way.

    Comparing the two hat-named producers to each other is near-vacuous --
    `computed_diamagnetism_from_phi` delegates to
    `virial_muihat_from_Bt_R0_dphi`, so it can only catch an argument-order
    slip. The substantive claim is that both are the *opposite* of the volume
    convention, and `calculate_diamagnetism` computes that independently, by
    integrating (B_tv^2 - B_t^2) over the plasma rather than from a flux.
    """
    args = (0.15, 0.4, -1.44e-3, 0.042, 0.956)   # B_t, R0, dphi, B_pa, Omega
    hat = virial_muihat_from_Bt_R0_dphi(*args)
    from_phi = computed_diamagnetism_from_phi(args[2], args[0], args[1], args[4], args[3])
    assert hat == pytest.approx(from_phi, rel=1e-12), "argument order"

    from vaft.omas.process_wrapper import compute_diamagnetism

    sample = sample_ods()
    independent = compute_diamagnetism(sample, time_index=0)
    row = vaft.omas.compute_virial_equilibrium_quantities_ods(sample, time_slice=0)[0]
    independent = float(np.asarray(independent, float).reshape(-1)[0])
    assert np.sign(row["mui"]) == np.sign(independent), (
        "the wrapper's volume mu_i must agree in sign with the volume integral "
        "computed from the F profile by a separate route"
    )
    assert np.sign(row["mui_hat"]) == -np.sign(independent)


def test_the_wrapper_reports_both_conventions_under_names_that_say_which():
    """`mui` is the volume integral the relations need; `mui_hat` keeps the flux
    quantity it has always carried.

    They are opposite in sign on every slice -- that is exact. Their magnitudes
    are *not* reliably close: the flux form is the volume one only to first
    order in (F-F_b)/F_b, and on this sample the ratio runs from -0.97 at
    slice 0 to -0.59 at slice 7. An earlier version of this test asserted
    `flux/volume ~ -1` at 5% and passed only because it looked at slice 0.
    """
    rows = vaft.omas.compute_virial_equilibrium_quantities_ods(sample_ods())
    ratios = []
    for row in rows.values():
        volume, flux = row["mui"], row["mui_hat"]
        if not (np.isfinite(volume) and np.isfinite(flux)):
            continue
        assert np.sign(volume) == -np.sign(flux), "the conventions are opposites"
        assert row["mui_from_flux"] == pytest.approx(flux, rel=1e-15)
        ratios.append(flux / volume)
    assert len(ratios) >= 8
    assert max(ratios) < 0, "every ratio is negative"
    # The spread is the point: a test pinning one number here would be pinning
    # the slice it happened to choose.
    assert min(ratios) < -0.9 and max(ratios) > -0.7


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

    # ... and the other half: a NaN *inside* the plasma must stay NaN. Dropping
    # every NaN would have returned a plausible-looking biased alpha here, which
    # is worse than the failure it was fixing.
    b_r_holed = np.where(rm == 0.0, np.nan, 0.2 / np.where(rm == 0.0, 1.0, rm))
    b_r_holed[20, 15] = np.nan
    assert not np.isfinite(
        efit_virial_volume_integrals(rm, zm, rb, zb, b_r_holed, b_z)["alpha"]
    ), "a NaN inside the plasma must not be silently dropped"


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


def test_the_measured_mu_i_reaches_the_closures_on_their_own_convention():
    """The measured loop and the reconstruction must be comparable.

    `virial_muihat_from_Bt_R0_dphi` is the flux convention by the rule above,
    and the closures take the volume one, so the wrapper negates it. Leaving it
    on the flux sign made `measured_mu_i_closures` a comparison of two
    different things: it was self-consistently wrong while the closures also
    took the flux sign, and became mixed the moment they stopped.

    Which toroidal field belongs in that conversion -- and so what sign a
    measured flux carries relative to the stored F -- is a separate, open
    question (#691). This pins only the flux/volume half of it.
    """
    rows = vaft.omas.compute_virial_equilibrium_quantities_ods(sample_ods())
    checked = 0
    for row in rows.values():
        measured = row["mu_i_sources"]["measured"]
        if not np.isfinite(measured):
            continue
        # The packaged loop reads diamagnetic, which is positive in the volume
        # convention; the reconstruction is paramagnetic. They disagree, and
        # the disagreement is only readable because both are on one convention.
        assert measured > 0 > row["mui"]
        # The closure on the measured mu_i must be the same function of it that
        # the derived one is of the derived mu_i -- not off by 2*mu_i.
        from vaft.formula.equilibrium import virial_pair_13_from_S_alpha_mu

        expected, _ = virial_pair_13_from_S_alpha_mu(
            row["s_1"], row["s_2"], row["s_3"], row["alpha"], measured
        )
        assert row["measured_mu_i_closures"]["pair_13"]["beta_p"] == pytest.approx(
            expected, rel=1e-12
        )
        checked += 1
    assert checked >= 8


def test_beta_pd_is_fed_the_flux_quantity_it_is_written_for():
    """`virial_beta_pd_from_S_mu_rt` is the one consumer on the flux side, so
    the wrapper must hand it `mui_from_flux` and not the volume `mui`.

    Given the flux quantity it estimates the same beta_p the volume-side Lao
    closure does, differing only by the second-order gap between the two
    conventions. Given the volume one it would be off by 2*mu_i, which on this
    sample is larger than beta_p itself.
    """
    from vaft.formula.equilibrium import virial_beta_pd_from_S_mu_rt

    row = vaft.omas.compute_virial_equilibrium_quantities_ods(sample_ods(), time_slice=0)[0]
    # beta_p_lao = S1/2 + S2/2*(1 - RT/R0) + mu_i, so RT/R0 inverts out of the
    # row's own numbers -- no need to re-derive the geometry here.
    rt_over_r0 = 1.0 - (row["beta_p"] - 0.5 * row["s_1"] - row["mui"]) / (0.5 * row["s_2"])
    expected = virial_beta_pd_from_S_mu_rt(
        row["s_1"], row["s_2"], row["mui_from_flux"], rt_over_r0
    )
    assert row["beta_pd_vir"] == pytest.approx(expected, rel=1e-9)
    # And it is close to the Lao beta_p, which the volume-fed version would not be.
    assert abs(row["beta_pd_vir"] - row["beta_p"]) < abs(2.0 * row["mui"])
