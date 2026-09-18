"""Which mu_i convention each virial function takes.

There are two in this codebase, and they are negatives of each other:

* **volume** -- ``mu_i = <B_tv^2 - B_t^2> / B_pa^2``, positive for a diamagnetic
  plasma. This is what the three virial relations require.
* **flux** (written ``mu_i_hat``) -- ``4*pi*B_t*R_0*dphi / (B_pa^2 * V)`` with
  ``dphi = integral (B_t - B_tv) dA``. This is EFIT's ``xmui``, and it is the
  negative of the volume form because
  ``B_tv^2 - B_t^2 = (F_b-F)(F_b+F)/R^2 ~ -2*F_b*(F-F_b)/R^2``.

There is no reliable rule in the *names*. ``hat`` and ``from_phi`` usually mean
the flux convention, but the module has counterexamples in both directions:
``approximated_diamagnetism_from_B_pa_B_tv_R0_delta_phi`` carries neither and
yields flux, while ``virial_bp_li_lihat_from_S123`` carries ``hat`` and yields
volume. What is reliable is the map, which this file pins:

* **flux producers** -- ``virial_muihat_from_Bt_R0_dphi``,
  ``computed_diamagnetism_from_phi``,
  ``approximated_diamagnetism_from_B_pa_B_tv_R0_delta_phi``;
* **volume producers** -- ``calculate_diamagnetism``, the wrapper's ``mui``,
  and the third return of ``virial_bp_li_lihat_from_S123`` (the three relations
  solve for it, so it is volume by construction);
* **every consumer takes volume**, except ``virial_beta_pd_from_S_mu_rt``,
  which takes flux and says so in its Convention section.

An earlier version of this header stated the naming as a rule. It was false in
both directions on the day it was written.

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
from types import SimpleNamespace

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

    They are opposite in sign on every slice -- that is exact -- and their
    magnitudes agree to about 2%, which is the genuine first-order error in
    (F-F_b)/F_b.

    This assertion has been wrong in both directions. It first read
    `flux/volume ~ -1` at 5% and passed only because it looked at slice 0. It
    was then loosened to *require* a spread running out to -0.59, which pinned
    a bug as though it were physics: the conversion was pairing b0 with this
    slice's geometric axis instead of taking F at the boundary, and that axis
    moves 0.3989 -> 0.2426 across the shot. b0 is read once per shot
    (`.flat[0]`), so it cannot be the source of a per-slice drift. With F at
    the boundary the ratio holds at -0.977 .. -0.982 on every slice.
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
    # Tight, and on every slice -- not a property of the one that was looked at.
    # A regression to the b0 * geometric-axis pairing reopens this to -0.59.
    assert min(ratios) > -1.05 and max(ratios) < -0.95


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
    #
    # Both functions, not just one. shafranov_integrals returned its 0.0
    # sentinel for an undetermined denominator, and 0.0 is pair_23's exact
    # singular point -- a plausible-looking alpha that passes the wrapper's
    # `isfinite` abstain guard, which is how the first version of this fix made
    # things worse rather than better.
    b_r_holed = np.where(rm == 0.0, np.nan, 0.2 / np.where(rm == 0.0, 1.0, rm))
    b_r_holed[20, 15] = np.nan
    assert not np.isfinite(
        efit_virial_volume_integrals(rm, zm, rb, zb, b_r_holed, b_z)["alpha"]
    ), "a NaN inside the plasma must not be silently dropped"
    *_, alpha_holed = shafranov_integrals(
        rb, zb, np.full_like(rb, 0.2), rm, zm, b_r_holed, b_z, B_ref=0.2, volume=1.0
    )
    assert not np.isfinite(alpha_holed), (
        "an undetermined alpha must abstain, not return the 0.0 sentinel"
    )


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
        # The two closures must differ by exactly what their two mu_i differ by,
        # and by nothing else. Re-deriving the measured one from `measured`
        # cannot see a 2*mu_i error, because both sides would carry it; the
        # derived closure is the independent anchor.
        #
        # pair_13: beta_p = ((alpha-1)(S1+S2) + S3 + alpha*mu) / (3*alpha-2),
        # so a change of mu moves it by alpha*dmu/(3*alpha-2).
        derived_bp = row["pair_13"]["beta_p"]
        measured_bp = row["measured_mu_i_closures"]["pair_13"]["beta_p"]
        alpha = row["alpha"]
        expected_gap = alpha * (measured - row["mui"]) / (3.0 * alpha - 2.0)
        assert measured_bp - derived_bp == pytest.approx(expected_gap, rel=1e-9)
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
    # And it estimates the same quantity as the Lao beta_p, so the two differ
    # only by the second-order gap between the conventions -- far less than the
    # 2*mu_i a volume-fed version would be out by. An earlier version of this
    # assertion used `< 2*|mu_i|`, which the volume-fed version also satisfies
    # (its difference is exactly 2*|mu_i|, landing a few ulp under).
    assert abs(row["beta_pd_vir"] - row["beta_p"]) < 0.5 * abs(2.0 * row["mui"])


def test_one_report_publishes_one_mu_i_convention():
    """Two signs for one quantity in a single report is how this went wrong.

    `_diamagnetic_energy` computes its mu_i on the flux convention because
    `virial_beta_pd_from_S_mu_rt` takes that -- correctly. But it published the
    flux value under `mui_measured` while the neighbouring check published the
    volume one under `mu_i_measured`, so the same slice carried +0.63 and -0.63
    for the same physical quantity.
    """
    from vaft.validation import validate_equilibrium

    report = validate_equilibrium(sample_ods(), time_slice=0)

    def field(category, check, name):
        entry = next(
            e for e in report[category][check]["slices"] if e["time_slice"] == 0
        )
        return entry.get(name)

    energy = field("independent_validation", "diamagnetic_energy", "mui_measured")
    measured = field("independent_validation", "virial_measured_mu_i", "mu_i_measured")
    equilibrium = field("physical_validity", "virial_parameter_plausibility", "mui")

    assert energy is not None and measured is not None
    assert energy == pytest.approx(measured, rel=1e-12), (
        "the two published measured mu_i must be the same number"
    )
    # Both are the volume convention, so both are comparable with the
    # equilibrium's own -- which is the point of publishing them.
    assert energy > 0 > equilibrium


def test_the_measured_conversion_takes_its_magnitude_from_F_and_its_sign_from_the_machine():
    """Each stored field contributes only what it is good for.

    `vacuum_toroidal_field.b0` is uniformly positive across the history but its
    magnitude drifts up to 39% within a single shot (#325). `profiles_1d.f` is
    the reverse: a stable magnitude with a sign that differs between sources
    for the same shot. So the conversion takes |F_boundary| for the magnitude
    and `VEST_TOROIDAL_FIELD_SIGN` for the direction, and the measurement's own
    sign then carries the physics.

    Using b0 understated the measured mu_i by up to 39% on this sample, which
    flattened a trend that is real: the plasma becomes more diamagnetic through
    the discharge.
    """
    from vaft.omas.process_wrapper import _vest_toroidal_field_sign

    VEST_TOROIDAL_FIELD_SIGN = _vest_toroidal_field_sign()
    from vaft.formula.equilibrium import virial_muihat_from_Bt_R0_dphi

    assert VEST_TOROIDAL_FIELD_SIGN > 0, "VEST's toroidal field is positive"
    # One record, not two: the machine layer owns this sign and marks it
    # unconfirmed pending #298, so a local copy could not stay in step.
    from vaft.machine_mapping import BT_SIGN_VEST_TO_IMAS
    assert VEST_TOROIDAL_FIELD_SIGN == float(BT_SIGN_VEST_TO_IMAS.sign)

    ods = sample_ods()
    rows = vaft.omas.compute_virial_equilibrium_quantities_ods(ods)
    checked, measured_values = 0, []
    for index, row in rows.items():
        eq = ods["equilibrium.time_slice"][index]
        measured = row["mu_i_sources"].get("measured")
        if measured is None or not np.isfinite(measured):
            continue
        f_edge = abs(float(np.asarray(eq["profiles_1d.f"], float)[-1]))
        r_0 = float(eq["boundary.geometric_axis.r"])
        flux = row["mu_i_sources"].get("measured_diamagnetic_flux")
        if flux is None or not np.isfinite(flux):
            flux = _measured_flux(ods, index)
        expected = -virial_muihat_from_Bt_R0_dphi(
            VEST_TOROIDAL_FIELD_SIGN * f_edge / r_0, r_0,
            flux, row["B_pa"], row["V_p"],
        )
        assert measured == pytest.approx(expected, rel=1e-9), (
            "the conversion must use |F_boundary|, not b0*R_0"
        )
        measured_values.append(measured)
        checked += 1
    assert checked >= 8
    # A diamagnetic loop gives a positive volume mu_i, and the trend b0's drift
    # was hiding is monotone across this discharge.
    assert min(measured_values) > 0
    assert measured_values[-1] > 1.7 * measured_values[0]


def _measured_flux(ods, index):
    return float(np.interp(
        float(ods["equilibrium.time"][index]),
        np.asarray(ods["magnetics.time"], float),
        np.asarray(ods["magnetics.diamagnetic_flux.0.data"], float),
    ))


def test_the_measured_mu_i_does_not_move_when_the_stored_F_sign_flips():
    """The blind spot that hid this twice: the packaged sample stores F > 0, so
    an `abs()` on it is a no-op and every test written against that sample
    passes either way.

    The database stores the same shot with F < 0 and an identical measurement,
    so a conversion that used the stored sign would read the same plasma as
    diamagnetic from one source and paramagnetic from the other. The volume
    mu_i is quadratic in F and cannot notice; this one must not either.
    """
    ods = sample_ods()
    base = vaft.omas.compute_virial_equilibrium_quantities_ods(
        copy.deepcopy(ods), time_slice=0
    )[0]

    flipped_ods = copy.deepcopy(ods)
    f = np.asarray(flipped_ods["equilibrium.time_slice.0.profiles_1d.f"], float)
    assert f[-1] > 0, "the packaged sample stores a positive F; that is the trap"
    flipped_ods["equilibrium.time_slice.0.profiles_1d.f"] = -f
    flipped = vaft.omas.compute_virial_equilibrium_quantities_ods(
        flipped_ods, time_slice=0
    )[0]

    # Quadratic in F, so untouched.
    assert flipped["mui"] == pytest.approx(base["mui"], rel=1e-9)
    # Linear in F, so this is the one that would flip on the stored sign.
    assert flipped["mu_i_sources"]["measured"] == pytest.approx(
        base["mu_i_sources"]["measured"], rel=1e-9
    ), "a COCOS-dependent stored sign must not reach the measured mu_i"
    assert flipped["mu_i_sources"]["measured"] > 0


def _geometric_axis_r(eq):
    """R_0 as the wrapper resolved it, or NaN.

    The packaged sample ships no ``boundary.geometric_axis``; the wrapper falls
    back to the boundary centre and writes that back, so this is readable only
    after a computation has run over the slice.
    """
    try:
        value = eq["boundary.geometric_axis.r"]
    except Exception:
        return float("nan")
    return float(value) if value is not None else float("nan")


def test_the_flux_conversion_uses_F_at_the_boundary_not_b0_times_the_geometric_axis():
    """The structural form of the fix, not just its numeric consequence.

    ``mui_hat`` is ``4*pi*B_t*R_0*dphi / (B_pa^2 * V)``, so the field-times-radius
    it was built with can be read straight back out of the published row:
    ``B_t*R_0 = mui_hat * B_pa^2 * V / (4*pi*dphi)``. That has to be F at the
    boundary.

    b0 is defined at ``vacuum_toroidal_field.r0`` -- a fixed 0.4 m here -- so
    pairing it with ``boundary.geometric_axis.r`` is not F at the boundary once
    the axis has moved, and on the last slice of this sample the two differ by
    41%.
    """
    ods = sample_ods()
    rows = vaft.omas.compute_virial_equilibrium_quantities_ods(ods)
    b0 = float(np.asarray(ods["equilibrium.vacuum_toroidal_field.b0"], float).flat[0])
    checked = 0
    diverged = 0
    for index, row in rows.items():
        eq = ods["equilibrium.time_slice"][index]
        if "profiles_1d.f" not in eq:
            continue
        if not (np.isfinite(row["mui_hat"]) and np.isfinite(row["phi_dia_comp"])):
            continue
        f_edge = float(np.asarray(eq["profiles_1d.f"], float)[-1])
        implied = row["mui_hat"] * row["B_pa"] ** 2 * row["V_p"] / (
            4 * np.pi * row["phi_dia_comp"]
        )
        assert implied == pytest.approx(f_edge, rel=1e-9), (
            "the quantity actually used must be F at the boundary"
        )
        r_geo = _geometric_axis_r(eq)
        if np.isfinite(r_geo) and abs(b0 * r_geo - f_edge) / abs(f_edge) > 0.05:
            # Where the old pairing would have differed, it must not be what
            # was used -- otherwise this test passes on agreement, not on form.
            assert implied != pytest.approx(b0 * r_geo, rel=0.02)
            diverged += 1
        checked += 1
    assert checked >= 8
    assert diverged >= 1, "no slice separated the two pairings; the test proved nothing"


def test_a_slice_whose_flux_the_wrapper_could_not_convert_is_indeterminate():
    """There is no second conversion any more, so this branch has to be real.

    The fallback that used to stand here recomputed mu_i from b0 * major_radius
    -- the same pairing the wrapper was just fixed not to use -- and published it
    under the same field name. Nothing reached it: replacing its body with a
    raise left the validation suite green. Refusing to answer is the correct
    behaviour, and this pins that it is what happens rather than a crash or an
    invented number.
    """
    from vaft.validation.equilibrium import _diamagnetic_energy

    ods = sample_ods()
    virial = dict(
        s_1=0.5, s_2=0.5, B_pa=0.1, V_p=0.05, rt=0.3, W_kin=100.0,
        mu_i_sources={"measured": float("nan")},
    )
    # R_0 comes from the descriptors rather than from `virial`; every other
    # input is finite, so the only thing left undecided is the conversion.
    descriptors = {"major_radius": SimpleNamespace(value=0.3)}
    out = _diamagnetic_energy(
        ods, 0, virial, {"measured": _measured_flux(ods, 0)}, descriptors
    )
    assert out["status"] == "indeterminate"
    assert "could not convert" in out["reason"]
    assert "mui_measured" not in out, "it must not publish a mu_i it does not have"


def test_the_computed_conversion_holds_under_either_stored_F_sign():
    """`mui_hat` converts `phi_dia_comp` -- a flux computed from the same F grid
    the volume mu_i is integrated over -- so converting it back must return that
    mu_i, negated, to the first-order (F - F_b)/F_b term and nothing more.

    That has to hold whichever sign F is stored with. Replicas of one shot do
    not agree on it: 39915 stores F_b = +0.0598 in the packaged sample and in
    `main`, and -0.0598 in the read-only legacy `public` replica. Flipping F
    flips both `phi_dia_comp` and F at the boundary, so a correct conversion is
    unchanged, and the volume mu_i, quadratic in F, does not move at all.

    What this catches is an `abs()` on the *computed* path: with |F_b| against
    a signed `phi_dia_comp` the ratio inverts under the flip. The measured path
    is a different conversion with its own sign rule, pinned by
    `test_the_measured_mu_i_does_not_move_when_the_stored_F_sign_flips`; this
    test says nothing about it.
    """
    for flip in (False, True):
        ods = copy.deepcopy(sample_ods())
        if flip:
            f = np.asarray(ods["equilibrium.time_slice.0.profiles_1d.f"], float)
            ods["equilibrium.time_slice.0.profiles_1d.f"] = -f
        row = vaft.omas.compute_virial_equilibrium_quantities_ods(ods, time_slice=0)[0]
        assert -row["mui_hat"] / row["mui"] == pytest.approx(1.0, rel=0.05), (
            f"self-consistency must hold with F {'negated' if flip else 'as stored'}"
        )
        assert row["mui"] < 0, "the volume mu_i is quadratic in F and must not move"


def test_the_diamagnetic_beta_p_negation_is_constrained():
    """`virial_beta_pd_from_S_mu_rt` takes the flux convention while the report
    publishes the volume one, so the validation layer negates at that one call.
    Nothing else pins that negation: no other test asserts on
    `beta_p_diamagnetic`, so dropping the minus sign changes the published
    number and leaves the suite green.

    This slice is decidable on the packaged sample, and the test asserts that
    rather than skipping when it is not -- a skip here would turn the guard off
    exactly when the report stops producing the value it guards.
    """
    from vaft.validation import validate_equilibrium
    from vaft.formula.equilibrium import virial_beta_pd_from_S_mu_rt

    report = validate_equilibrium(sample_ods(), time_slice=0)
    entry = next(
        e for e in report["independent_validation"]["diamagnetic_energy"]["slices"]
        if e["time_slice"] == 0
    )
    assert entry["status"] not in {"not_available", "indeterminate"}, entry.get("reason")
    row = vaft.omas.compute_virial_equilibrium_quantities_ods(sample_ods(), time_slice=0)[0]
    expected = virial_beta_pd_from_S_mu_rt(
        row["s_1"], row["s_2"], -entry["mui_measured"], row["rt"] / entry["R_0"]
    )
    # This is the assertion that catches a dropped sign. On this slice the
    # published value is +1.440; without the negation it is +0.174 -- still
    # positive, so the sign check below cannot tell the two apart.
    assert entry["beta_p_diamagnetic"] == pytest.approx(expected, rel=1e-9)
    # A physical floor only: a diamagnetic measurement gives a positive beta_p
    # through this closure.
    assert entry["beta_p_diamagnetic"] > 0
