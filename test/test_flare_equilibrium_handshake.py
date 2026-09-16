"""Handing a GEQDSK to FLARE in the convention FLARE actually traces in.

FLARE takes two manual sign multipliers and derives its field directions from
whatever they produce, so getting them wrong does not shift the answer -- it
changes the traced topology. The legacy inferred them from g-file metadata and
fell back to no conversion when the parse failed; these tests pin the two
properties that replaces: the multipliers follow from an identified COCOS, and
an unidentifiable one is refused.
"""

from __future__ import annotations

import dataclasses

import pytest

from vaft.data.cocos import convention_for, cocos_spec
from vaft.code.flare import FlareEquilibriumScales, flare_equilibrium_scales
from vaft.process.cocos import cocos_field_scales


def _equilibrium(**fields):
    """A real EquilibriumData, populated only where a test needs it.

    Not a stub: the identification reads more of an equilibrium than the four
    sign-bearing fields, and a hand-built object that satisfied it would be
    asserting my idea of the reader rather than the reader.
    """
    from vaft.data.equilibrium import EquilibriumData

    return EquilibriumData(**fields)


def _cocos_5_like():
    """An equilibrium with the signs the DIII-D reference g-file carries.

    Measured on that file: ``bt0`` negative, ``ip`` positive, ``q`` positive
    and ``psi_1d`` rising from -0.2996 to -0.0373 -- which identifies as COCOS
    5 or 6, the pair that differs by the machine's toroidal handedness.
    """
    import numpy as np

    return _equilibrium(
        bt0=-1.7228, ip=1.18663e6,
        q=np.linspace(1.062, 8.0323, 32),
        psi_1d=np.linspace(-0.299591, -0.0373057, 32),
        psi_axis=-0.299591, psi_boundary=-0.0373057,
    )


# --------------------------------------------------------------------------
# The registered convention
# --------------------------------------------------------------------------


def test_flare_is_registered_as_cocos_3():
    """Read off what equi2d.f90 does, not off a declaration -- FLARE's source
    does not contain the word COCOS."""
    flare = convention_for("flare")
    assert flare.cocos == 3
    assert flare.psi_unit == "Wb/rad"
    assert flare.confirmed is True
    assert not flare.identifies_per_file


# --------------------------------------------------------------------------
# The sign derivation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source, expected",
    [
        # Measured on real runs: a COCOS 5 g-file into FLARE, and a COCOS 2
        # CHEASE equilibrium. Both are reproduced by the sign relations.
        (5, (-1, +1)),
        (2, (+1, -1)),
    ],
)
def test_the_measured_conversions_follow_from_the_sign_relations(source, expected):
    assert cocos_field_scales(source, 3) == expected


def test_a_conversion_to_its_own_convention_changes_nothing():
    for index in INDICES:
        assert cocos_field_scales(index, index) == (1, 1)


#: The eighteen COCOS indices: 1-8 and 11-18. There is no 9 or 10.
INDICES = tuple(range(1, 9)) + tuple(range(11, 19))


def test_the_flux_multiplier_follows_the_product_of_the_two_sigmas():
    """Sauter Eq. 12: B_pol goes as sigma_Bp * sigma_RphiZ grad(psi) x grad(phi),
    so psi's sign relative to the physical field is the product -- not
    sigma_Bp alone, which is the easy mistake."""
    for source in INDICES:
        for target in INDICES:
            first, second = cocos_spec(source), cocos_spec(target)
            psi_scale, field_scale = cocos_field_scales(source, target)
            assert psi_scale == (
                first.sigma_bp * first.sigma_rpz * second.sigma_bp * second.sigma_rpz
            )
            assert field_scale == first.sigma_rpz * second.sigma_rpz


def test_the_conversion_is_its_own_inverse():
    for source in INDICES:
        for target in INDICES:
            forward = cocos_field_scales(source, target)
            assert cocos_field_scales(target, source) == forward


def test_a_convention_pair_differing_only_in_the_poloidal_angle_is_not_distinguished():
    """A documented limitation, pinned so it is not mistaken for completeness:
    sigma_rhothetaphi sets theta's direction and these two multipliers do not
    carry it."""
    pairs = [
        (a, b)
        for a in INDICES
        for b in INDICES
        if a != b
        and cocos_spec(a).sigma_bp == cocos_spec(b).sigma_bp
        and cocos_spec(a).sigma_rpz == cocos_spec(b).sigma_rpz
        and cocos_spec(a).sigma_rhotp != cocos_spec(b).sigma_rhotp
    ]
    assert pairs, "the fixture assumes such a pair exists"
    for a, b in pairs:
        assert cocos_field_scales(a, b) == (1, 1)


@pytest.mark.parametrize("bad", [0, 9, 10, 19, -3])
def test_an_index_outside_the_eighteen_is_refused(bad):
    with pytest.raises(ValueError):
        cocos_field_scales(bad, 3)


# --------------------------------------------------------------------------
# The handshake
# --------------------------------------------------------------------------


def test_a_stated_source_convention_skips_the_identification():
    """Provenance settles some files that the signs alone cannot."""
    scales = flare_equilibrium_scales(_equilibrium(),
                                      source_cocos=5)
    assert isinstance(scales, FlareEquilibriumScales)
    assert (scales.scale_ip, scales.scale_bt) == (-1, +1)
    assert (scales.source_cocos, scales.target_cocos) == (5, 3)


def test_an_unidentifiable_equilibrium_is_refused_not_defaulted():
    """The legacy fell back to 1.0 on a parse failure, which is
    indistinguishable from "no conversion needed"."""
    with pytest.raises(ValueError, match="cannot be identified"):
        flare_equilibrium_scales(_equilibrium())


def test_an_ambiguous_pair_is_refused_rather_than_halved():
    """A g-file identifies as an odd/even pair differing by the machine's
    toroidal handedness; and when the flux normalization cannot be measured
    either, by the weber / weber-per-radian family as well. The message has to
    say which, because the two need different facts to resolve and naming the
    commoner one would be wrong half the time."""
    equilibrium = _cocos_5_like()
    with pytest.raises(ValueError, match="weber or weber per radian"):
        flare_equilibrium_scales(equilibrium)
    # Provenance settles it where the data cannot.
    resolved = flare_equilibrium_scales(equilibrium, source_cocos=5)
    assert (resolved.scale_ip, resolved.scale_bt) == (-1, +1)
    assert resolved.psi_per_radian is True


def test_the_refusal_names_the_fact_it_is_missing():
    """Two ambiguities reach the same refusal and they are not resolved the
    same way."""
    import numpy as np

    # Signs alone, no LCFS: both the handedness and the flux family are open.
    both = _cocos_5_like()
    with pytest.raises(ValueError) as open_both:
        flare_equilibrium_scales(both)
    assert "clockwise_phi" in str(open_both.value)
    assert "weber or weber per radian" in str(open_both.value)

    # With the handedness supplied, only the family is left.
    with pytest.raises(ValueError) as family_only:
        flare_equilibrium_scales(both, clockwise_phi=False)
    assert "weber or weber per radian" in str(family_only.value)
    assert "clockwise_phi" not in str(family_only.value)


def test_the_result_says_which_flux_family_the_source_was_in():
    """These multipliers carry signs only; a source in 11-18 also needs its
    flux divided by 2*pi, and the caller is told rather than silently left
    with a factor missing."""
    assert flare_equilibrium_scales(
        _equilibrium(), source_cocos=5
    ).psi_per_radian is True
    assert flare_equilibrium_scales(
        _equilibrium(), source_cocos=15
    ).psi_per_radian is False


def test_the_scales_are_a_frozen_record_of_where_they_came_from():
    scales = flare_equilibrium_scales(
        _equilibrium(), source_cocos=2
    )
    assert dataclasses.asdict(scales) == {
        "scale_ip": +1, "scale_bt": -1,
        "source_cocos": 2, "target_cocos": 3, "psi_per_radian": True,
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        scales.scale_ip = -1
