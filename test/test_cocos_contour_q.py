"""The contour-q flux-normalisation discriminator, and cross-family transform parity.

``identify_flux_exponent`` decides weber against weber-per-radian from Ampere's
law on the LCFS, so it needs ``ip`` and a boundary outline.
``identify_flux_exponent_from_q`` decides the same question by rebuilding ``q``
on interior contours, so it needs neither.  What is tested here is that the two
agree where both can speak, that the second one speaks where the first cannot,
and that neither of them answers when the equilibrium contradicts itself.

Ported from hsyun_GPEC ``library/geqdsk_cocos.py`` on branch
``codex/gpec-flare-cocos-handshake`` (issue #11).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from vaft.data.cocos import cocos_spec
from vaft.process.cocos import (
    CONTOUR_Q_LOG_TOLERANCE,
    cocos_field_scales,
    identify_convention,
    identify_flux_exponent,
    identify_flux_exponent_from_q,
)

#: The legacy's relative criterion, reproduced here so the tests can show both
#: what it would have accepted and that it is now unreachable.
LEGACY_SEPARATION = 3.0

TWO_PI = 2.0 * math.pi


@pytest.fixture(scope="module")
def equilibrium():
    """The packaged VEST g-file, which is stored in weber per radian."""
    from vaft.data.resources import sample_geqdsk
    from vaft.process.equilibrium import as_equilibrium

    return as_equilibrium(sample_geqdsk(), convention=1)


def _rescale(equilibrium, factor: float):
    """The same equilibrium with every poloidal flux multiplied by ``factor``."""
    return dataclasses.replace(
        equilibrium,
        psi=equilibrium.psi * factor,
        psi_axis=equilibrium.psi_axis * factor,
        psi_boundary=equilibrium.psi_boundary * factor,
        psi_1d=equilibrium.psi_1d * factor,
    )


# --------------------------------------------------------------------------
# Transform parity with the legacy cocos_transform
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source, target, legacy_psi, legacy_bt",
    [
        # geqdsk_cocos.cocos_transform(source, target)["PSI"] and ["BT"].
        (5, 3, -1.0, +1.0),
        (2, 3, +1.0, -1.0),
        # The cross-family case, where the legacy folds a factor of 2*pi into
        # the same number VAFT keeps as a pure sign.
        (15, 3, -1.0 / TWO_PI, +1.0),
        (5, 13, -1.0 * TWO_PI, +1.0),
    ],
)
def test_the_legacy_transform_is_the_sign_times_the_flux_exponent_factor(
    source, target, legacy_psi, legacy_bt
):
    """``PSI = sigma_ip sigma_bp (2 pi)^(e_target - e_source)``.

    VAFT splits that deliberately: :func:`cocos_field_scales` returns the sign
    and nothing else, because the ``2*pi`` is a unit change and applying it
    silently is how a flux ends up off by 6.28.  The product is the legacy
    number, which is what this pins.
    """
    psi_scale, field_scale = cocos_field_scales(source, target)
    exponent = cocos_spec(target).exp_bp - cocos_spec(source).exp_bp
    assert psi_scale * TWO_PI**exponent == pytest.approx(legacy_psi)
    assert field_scale == pytest.approx(legacy_bt)


def test_the_sign_alone_is_not_the_legacy_number_across_families():
    """The reason the previous test multiplies: within a family the two agree,
    across families they differ by 2*pi and a caller that forgets is wrong."""
    assert cocos_field_scales(15, 3)[0] == -1
    assert cocos_field_scales(15, 3)[0] != pytest.approx(-1.0 / TWO_PI)


# --------------------------------------------------------------------------
# The discriminator on a real equilibrium
# --------------------------------------------------------------------------


def test_the_two_discriminators_agree_on_the_packaged_equilibrium(equilibrium):
    result = identify_flux_exponent_from_q(equilibrium)
    assert result.exponent == 0
    assert result.reason == ""
    assert identify_flux_exponent(equilibrium)[0] == 0


def test_the_losing_hypothesis_lands_at_log_two_pi(equilibrium):
    """What makes the test decisive rather than a threshold on a continuum: the
    wrong hypothesis is the right one times 2*pi, so in a log-ratio it sits
    ln(2*pi) = 1.8379 away whatever the equilibrium -- and, unlike a plain
    relative error, by the same amount in either direction."""
    result = identify_flux_exponent_from_q(equilibrium)
    assert result.log_ratio_per_radian < 1e-3
    assert result.log_ratio_weber == pytest.approx(math.log(TWO_PI), rel=1e-3)
    assert result.separation > 1e3
    assert result.surface_count >= 10


def test_a_weber_equilibrium_is_identified_as_weber(equilibrium):
    """The same file with psi multiplied by 2*pi is the other family, and the
    two log-ratios swap places exactly -- which a relative error would not do:
    it gave 5.283 one way round and 0.841 the other for the same 2*pi."""
    per_radian = identify_flux_exponent_from_q(equilibrium)
    weber = identify_flux_exponent_from_q(_rescale(equilibrium, TWO_PI))
    assert weber.exponent == 1
    assert weber.log_ratio_weber < 1e-3
    assert weber.log_ratio_per_radian == pytest.approx(math.log(TWO_PI), rel=1e-3)
    assert weber.log_ratio_per_radian == pytest.approx(
        per_radian.log_ratio_weber, rel=1e-3
    )


# --------------------------------------------------------------------------
# Where it answers and Ampere's law cannot
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field, value", [("ip", 0.0), ("ip", None), ("lcfs", None)])
def test_it_answers_without_the_current_or_the_boundary(equilibrium, field, value):
    """The whole reason to carry this method: it reads the flux map in its
    interior, so neither ``ip`` nor the boundary outline is an input."""
    crippled = dataclasses.replace(equilibrium, **{field: value})
    assert identify_flux_exponent(crippled) == (None, None), "Ampere's law needs both"
    assert identify_flux_exponent_from_q(crippled).exponent == 0


def test_identify_convention_narrows_to_one_family_without_a_boundary(equilibrium):
    """Before the fallback existed, a g-file with no outline spanned both storage
    families and the FLARE handshake refused it."""
    per_radian = dataclasses.replace(equilibrium, lcfs=None)
    weber = dataclasses.replace(_rescale(equilibrium, TWO_PI), lcfs=None)
    assert identify_convention(per_radian, clockwise_phi=True) == (2,)
    assert identify_convention(weber, clockwise_phi=True) == (12,)


# --------------------------------------------------------------------------
# Abstention, and the legacy defect it fixes (D-04)
# --------------------------------------------------------------------------


def _legacy_relative_errors(result):
    """The legacy's two relative errors, recovered from the measured log-ratios.

    It compared ``|q_model - q_stored| / q_stored`` rather than a log-ratio, so
    asserting what it would have done means recomputing in its own measure --
    applying its threefold rule to a log-ratio is a different criterion and
    answers differently.  The two hypotheses differ by exactly ``2*pi``, which
    is what pins down the sign ``|ln|`` dropped.
    """
    for sign in (1.0, -1.0):
        ratio = math.exp(sign * result.log_ratio_per_radian)
        if math.isclose(
            abs(math.log(TWO_PI * ratio)), result.log_ratio_weber, rel_tol=1e-9
        ):
            return abs(ratio - 1.0), abs(TWO_PI * ratio - 1.0)
    raise AssertionError("the two log-ratios are not ln(2*pi) apart")


@pytest.mark.parametrize("factor", [0.5, 1.0 / 3.0])
def test_a_mis_scaled_psi_is_an_abstention_not_a_confident_answer(equilibrium, factor):
    """The legacy accepted a winner purely for beating the loser threefold, with
    no bar on its own agreement.  On psi scaled by one half its two relative
    errors are 1.00 and 11.57 -- a separation of 11.6 -- so it returns "weber
    per radian, decisive" for a hypothesis its own measurement misses by a
    factor of two.
    """
    result = identify_flux_exponent_from_q(_rescale(equilibrium, factor))
    winner = min(result.log_ratio_per_radian, result.log_ratio_weber)

    # The legacy criterion, in the legacy's own measure, would have accepted it.
    legacy_winner, legacy_loser = sorted(_legacy_relative_errors(result))
    assert legacy_loser / legacy_winner > LEGACY_SEPARATION
    assert legacy_winner > 0.5, "and its own residual was never checked"
    # The bar on the winner's own agreement is what rejects it.
    assert winner > CONTOUR_Q_LOG_TOLERANCE
    assert result.exponent is None
    assert "off by a factor" in result.reason


def test_the_legacy_separation_rule_can_no_longer_fail(equilibrium):
    """Why there is one criterion and not two: the hypotheses sit exactly
    ln(2*pi) apart, so a winner inside the band leaves a ratio of at least
    ln(2*pi)/0.1398 - 1 = 12.  The separation is reported, never applied."""
    result = identify_flux_exponent_from_q(equilibrium)
    assert result.exponent is not None
    assert min(result.log_ratio_per_radian, result.log_ratio_weber) <= CONTOUR_Q_LOG_TOLERANCE
    assert result.separation > 12.0


@pytest.mark.parametrize("factor", [1.16, 1.0 / 1.16])
def test_the_acceptance_band_is_symmetric_in_the_ratio(equilibrium, factor):
    """A relative error is not: it accepted a model q 1.16 times too small and
    rejected one 1.16 times too large, for the same factor of disagreement."""
    scaled = dataclasses.replace(equilibrium, q=equilibrium.q * factor)
    result = identify_flux_exponent_from_q(scaled)
    winner = min(result.log_ratio_per_radian, result.log_ratio_weber)
    assert winner == pytest.approx(math.log(1.16), rel=0.05)
    assert result.exponent is None


def test_a_psi_between_the_two_families_is_rejected(equilibrium):
    """psi scaled by 3 puts the measurement between the hypotheses; the winner
    is off by a factor of 2.09 and is refused.  This is the one the legacy also
    refused, in its own measure."""
    result = identify_flux_exponent_from_q(_rescale(equilibrium, 3.0))
    legacy_winner, legacy_loser = sorted(_legacy_relative_errors(result))
    assert legacy_loser / legacy_winner < LEGACY_SEPARATION
    assert result.exponent is None


def test_a_detected_ip_contradiction_is_not_overruled_by_the_second_opinion(equilibrium):
    """An ``ip`` five times too small is a real defect of the file.  Ampere's law
    measures it and rejects both families; the contour-q rung reads neither
    ``ip`` nor the boundary, so letting it answer here would hide the defect.
    """
    wrong = dataclasses.replace(equilibrium, ip=equilibrium.ip * 0.2)
    exponent, ratio = identify_flux_exponent(wrong)
    assert exponent is None and ratio is not None, "the rung ran and rejected both"
    # On its own the second rung is perfectly happy: ip is not one of its inputs.
    assert identify_flux_exponent_from_q(wrong).exponent == 0
    # identify_convention must still decline to name one index.
    assert len(identify_convention(wrong, clockwise_phi=True)) > 1


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["psi", "q", "f", "psi_1d", "r", "z"])
def test_a_missing_input_abstains_and_says_which(equilibrium, field):
    result = identify_flux_exponent_from_q(dataclasses.replace(equilibrium, **{field: None}))
    assert result.exponent is None
    assert field in result.reason


def test_a_q_profile_of_zeros_leaves_no_usable_surface(equilibrium):
    result = identify_flux_exponent_from_q(
        dataclasses.replace(equilibrium, q=np.zeros_like(equilibrium.q))
    )
    assert result.exponent is None
    assert "usable closed contours" in result.reason


def test_a_profile_with_no_finite_values_says_which_one(equilibrium):
    """Distinguished from "no usable contours": an all-NaN q and a plasma the
    contours miss are different defects and used to give the same message."""
    for name in ("q", "f"):
        blank = np.full_like(getattr(equilibrium, name), np.nan)
        result = identify_flux_exponent_from_q(
            dataclasses.replace(equilibrium, **{name: blank})
        )
        assert result.exponent is None
        assert "has no finite values" in result.reason


def test_an_axis_outside_the_grid_says_so(equilibrium):
    result = identify_flux_exponent_from_q(
        dataclasses.replace(equilibrium, magnetic_axis=(99.0, 99.0))
    )
    assert result.exponent is None
    assert "encloses the magnetic axis" in result.reason


def test_a_degenerate_flux_window_abstains(equilibrium):
    result = identify_flux_exponent_from_q(
        dataclasses.replace(equilibrium, psi_boundary=equilibrium.psi_axis)
    )
    assert result.exponent is None
    assert "psi_boundary equals psi_axis" in result.reason


def test_too_few_levels_abstains_rather_than_answering_from_one_surface(equilibrium):
    result = identify_flux_exponent_from_q(equilibrium, levels=[0.5, 0.6])
    assert result.exponent is None
    assert result.surface_count == 2
    assert "fewer than 3" in result.reason
