"""Tests for VAFT's COCOS convention model and per-code declarations.

The reference is Sauter & Medvedev, *Tokamak Equilibrium Coordinate Conventions*
(2013): Table I for the sixteen indices, Eq. 20 for the poloidal field, and
Eq. 23 for the sign consistency relations.
"""

from __future__ import annotations

import math

import pytest


def test_only_the_sixteen_defined_indices_are_accepted():
    """9 and 10 do not exist; the old range check accepted the whole 1..18 span."""
    from vaft.data.cocos import COCOS_INDICES, cocos_spec

    assert COCOS_INDICES == (1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18)
    for index in (0, 9, 10, 19, -1, 12.5):
        with pytest.raises(ValueError, match="not defined"):
            cocos_spec(index)


def test_spec_reproduces_sauter_table_one():
    """Spot-check the rows that pin the four independent sign choices."""
    from vaft.data.cocos import cocos_spec

    expected = {
        # index: (sigma_bp, sigma_rpz, sigma_rhotp, exp_bp)
        1: (+1, +1, +1, 0),
        2: (+1, -1, +1, 0),
        3: (-1, +1, -1, 0),
        5: (+1, +1, -1, 0),
        7: (-1, +1, +1, 0),
        11: (+1, +1, +1, 1),
        13: (-1, +1, -1, 1),
        17: (-1, +1, +1, 1),
    }
    for index, (sigma_bp, sigma_rpz, sigma_rhotp, exp_bp) in expected.items():
        spec = cocos_spec(index)
        assert (spec.sigma_bp, spec.sigma_rpz, spec.sigma_rhotp, spec.exp_bp) == (
            sigma_bp, sigma_rpz, sigma_rhotp, exp_bp
        ), index


def test_index_and_index_plus_ten_differ_only_by_the_two_pi_exponent():
    """Table I: COCOS i and i+10 share every orientation, differing only in e_Bp."""
    from vaft.data.cocos import cocos_spec

    for index in range(1, 9):
        low, high = cocos_spec(index), cocos_spec(index + 10)
        assert (low.sigma_bp, low.sigma_rpz, low.sigma_rhotp) == (
            high.sigma_bp, high.sigma_rpz, high.sigma_rhotp
        )
        assert (low.exp_bp, high.exp_bp) == (0, 1)
        assert low.psi_per_radian and not high.psi_per_radian
        assert high.bp_factor == pytest.approx(low.bp_factor / (2 * math.pi))


def test_bp_factor_carries_both_the_orientation_sign_and_the_two_pi():
    """Eq. 20's coefficient is sigma_RphiZ * sigma_Bp / (2*pi)**e_Bp.

    Applying only the 2*pi part leaves the field inverted for half the indices,
    which is what the pre-existing `_grid_fields` did.
    """
    from vaft.data.cocos import COCOS_INDICES, cocos_spec

    for index in COCOS_INDICES:
        spec = cocos_spec(index)
        assert spec.bp_factor == pytest.approx(
            spec.sigma_rpz * spec.sigma_bp / (2 * math.pi) ** spec.exp_bp
        )
    # The form VAFT hardcoded, B_R = -(1/R) dpsi/dZ, is only correct for these.
    matching = [i for i in COCOS_INDICES if cocos_spec(i).bp_factor == pytest.approx(-1.0)]
    assert matching == [2, 3, 6, 7]
    # And it is sign-inverted for the internal convention.
    assert cocos_spec(11).bp_factor > 0


def test_equation_23_agrees_with_the_tabulated_q_and_pprime_signs():
    """Eq. 23 must reproduce Table I's sign(q) and sign(dp/dpsi) for Ip, B0 > 0.

    This is an independent cross-check: the expected signs come from VAFT's own
    Eq. 23 relations, the tabulated ones straight from `omas.define_cocos`.
    """
    from vaft.data.cocos import COCOS_INDICES, cocos_spec

    for index in COCOS_INDICES:
        spec = cocos_spec(index)
        assert spec.expected_sign("q", sigma_ip=1, sigma_b0=1) == spec.sign_q_pos, index
        assert spec.expected_sign("pprime", sigma_ip=1, sigma_b0=1) == spec.sign_pprime_pos, index


def test_equation_23_relations_track_the_current_and_field_signs():
    """Sauter Table III: flipping Ip flips psi, dp/dpsi, j_phi and q; flipping B0 flips F and q."""
    from vaft.data.cocos import cocos_spec

    spec = cocos_spec(11)
    base = {name: spec.expected_sign(name, sigma_ip=1, sigma_b0=1)
            for name in ("f", "phi_tor", "dpsi", "pprime", "j_phi", "q")}
    flipped_ip = {name: spec.expected_sign(name, sigma_ip=-1, sigma_b0=1) for name in base}
    flipped_b0 = {name: spec.expected_sign(name, sigma_ip=1, sigma_b0=-1) for name in base}

    for name in ("dpsi", "pprime", "j_phi", "q"):
        assert flipped_ip[name] == -base[name], name
    for name in ("f", "phi_tor"):
        assert flipped_ip[name] == base[name], name
    for name in ("f", "phi_tor", "q"):
        assert flipped_b0[name] == -base[name], name
    for name in ("dpsi", "pprime", "j_phi"):
        assert flipped_b0[name] == base[name], name


def test_unknown_quantity_names_are_rejected_with_the_valid_list():
    from vaft.data.cocos import cocos_spec

    with pytest.raises(ValueError, match="expected one of"):
        cocos_spec(11).expected_sign("beta_p", sigma_ip=1, sigma_b0=1)


def test_internal_convention_is_the_imas_data_dictionary_one():
    from vaft.data.cocos import VAFT_INTERNAL_COCOS, convention_for

    assert VAFT_INTERNAL_COCOS == 11
    for name in ("imas", "omas"):
        assert convention_for(name).cocos == VAFT_INTERNAL_COCOS
        assert convention_for(name).psi_unit == "Wb"


def test_every_external_code_declares_a_convention():
    """Each adapter must declare what its code expects rather than assuming."""
    from vaft.data.cocos import COCOS_INDICES, convention_for, known_codes

    assert {"chease", "efit", "geqdsk", "gpec", "imas", "omas", "tes", "vfit"} <= set(known_codes())
    assert convention_for("chease").cocos == 2  # Sauter Sect. IX
    for name in known_codes():
        convention = convention_for(name)
        assert convention.reference, name
        assert convention.cocos is None or convention.cocos in COCOS_INDICES, name
        # A g-file carries no convention field, so those must identify per file.
        if name in {"geqdsk", "efit", "tes", "gpec"}:
            assert convention.identifies_per_file, name


def test_unconfirmed_conventions_are_marked_as_assumptions():
    """VFIT's COCOS 11 claim has never been checked against Eq. 23."""
    from vaft.data.cocos import convention_for

    assert convention_for("vfit").confirmed is False
    assert convention_for("chease").confirmed is True


def test_unknown_code_error_lists_the_declared_ones():
    from vaft.data.cocos import convention_for

    with pytest.raises(KeyError, match="chease"):
        convention_for("freegs")


def test_registry_refuses_to_silently_replace_an_entry():
    from vaft.data.cocos import CodeConvention, convention_for, register_convention

    # Re-registering an identical entry is a no-op, so import order cannot break.
    register_convention(convention_for("chease"))
    with pytest.raises(ValueError, match="already registered"):
        register_convention(CodeConvention(
            name="chease", cocos=13, psi_unit="Wb", reference="wrong",
        ))
    assert convention_for("chease").cocos == 2


def test_registry_rejects_an_undefined_index():
    from vaft.data.cocos import CodeConvention, register_convention

    with pytest.raises(ValueError, match="not defined"):
        register_convention(CodeConvention(
            name="fictional", cocos=9, psi_unit="Wb", reference="none",
        ))


def test_cocos_types_are_exported_from_the_data_package():
    import vaft.data

    assert vaft.data.VAFT_INTERNAL_COCOS == 11
    assert vaft.data.cocos_spec(2).sigma_rpz == -1
    assert vaft.data.convention_for("chease").cocos == 2
    for name in ("CocosSpec", "CodeConvention", "cocos_spec", "convention_for", "VAFT_INTERNAL_COCOS"):
        assert name in vaft.data.__all__, name
