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


# --- Sauter Eq. 23 consistency checking -----------------------------------


def _consistent_equilibrium(cocos: int = 11, *, ip: float = 1.0e6, bt0: float = 2.0):
    """An equilibrium built to satisfy Eq. 23 for ``cocos`` by construction."""
    import numpy as np

    from vaft.data.cocos import cocos_spec
    from vaft.data.equilibrium import EquilibriumConvention, EquilibriumData

    spec = cocos_spec(cocos)
    sigma_ip, sigma_b0 = (1 if ip > 0 else -1), (1 if bt0 > 0 else -1)
    psi_n = np.linspace(0.0, 1.0, 65)
    # sign(psi_edge - psi_axis) = sigma_Ip * sigma_Bp
    delta = spec.expected_sign("dpsi", sigma_ip=sigma_ip, sigma_b0=sigma_b0)
    psi_1d = delta * psi_n
    return EquilibriumData(
        psi_axis=0.0, psi_boundary=float(delta),
        psi_1d=psi_1d,
        # p falls from axis to edge, so sign(dp/dpsi) follows -sigma_Ip*sigma_Bp
        pressure=1.0e4 * (1.0 - psi_n**2),
        f=np.full(psi_n.size, spec.expected_sign("f", sigma_ip=sigma_ip, sigma_b0=sigma_b0) * 1.0),
        q=np.linspace(1.0, 3.0, psi_n.size) * spec.expected_sign("q", sigma_ip=sigma_ip, sigma_b0=sigma_b0),
        ip=ip, bt0=bt0, r0=1.0,
        convention=EquilibriumConvention(cocos, (cocos,), spec.psi_per_radian, None,
                                         sigma_ip, sigma_b0, None, "test"),
    )


@pytest.mark.parametrize("cocos", [1, 2, 3, 5, 7, 11, 13, 17])
@pytest.mark.parametrize("ip,bt0", [(1e6, 2.0), (-1e6, 2.0), (1e6, -2.0), (-1e6, -2.0)])
def test_an_equilibrium_built_to_equation_23_validates_in_its_own_convention(cocos, ip, bt0):
    """Eq. 23 must hold for every index and every combination of Ip and B0 signs."""
    from vaft.process.cocos import validate_cocos

    report = validate_cocos(_consistent_equilibrium(cocos, ip=ip, bt0=bt0), cocos)
    assert report.valid, [item.message for item in report.issues]


def test_the_wrong_index_is_rejected_with_the_relation_that_failed():
    """COCOS 1 and 3 differ in sigma_Bp, so psi and dp/dpsi must both disagree."""
    from vaft.process.cocos import validate_cocos

    report = validate_cocos(_consistent_equilibrium(1), 3)
    assert not report.valid
    codes = {item.code for item in report.issues}
    assert {"cocos_sign_dpsi", "cocos_sign_pprime"} <= codes
    message = next(item.message for item in report.issues if item.code == "cocos_sign_dpsi")
    assert "COCOS 3 requires" in message and "but it is" in message


def test_a_q_sign_mismatch_is_a_warning_because_codes_commonly_emit_abs_q():
    """Sauter Sect. IV: warn on a q mismatch, do not reject the equilibrium."""
    import numpy as np

    from vaft.data.equilibrium import EquilibriumData
    from vaft.process.cocos import validate_cocos

    eq = _consistent_equilibrium(3)
    flipped = EquilibriumData(**{**eq.__dict__, "q": np.abs(eq.q)})
    report = validate_cocos(flipped, 3)
    issues = {item.code: item.severity for item in report.issues}
    assert issues["cocos_sign_q"] == "warning"
    assert report.valid, "a q mismatch alone must not invalidate the equilibrium"


def test_an_undeclared_convention_is_an_error_rather_than_a_silent_pass():
    from vaft.data.equilibrium import EquilibriumData
    from vaft.process.cocos import validate_cocos

    report = validate_cocos(EquilibriumData())
    assert not report.valid
    assert {item.code for item in report.issues} == {"cocos_undeclared"}


def test_unavailable_inputs_produce_one_summary_warning_not_one_per_relation():
    """j_phi and the toroidal flux are not carried on the model, so they are skipped."""
    from vaft.process.cocos import validate_cocos

    report = validate_cocos(_consistent_equilibrium(11), 11)
    unverifiable = [item for item in report.issues if item.code == "cocos_unverifiable"]
    assert len(unverifiable) == 1
    assert "toroidal current density" in unverifiable[0].message
    assert "toroidal flux" in unverifiable[0].message


def test_missing_current_or_field_sign_stops_the_check_without_claiming_success():
    from vaft.data.equilibrium import EquilibriumData
    from vaft.process.cocos import validate_cocos

    eq = _consistent_equilibrium(11)
    report = validate_cocos(EquilibriumData(**{**eq.__dict__, "bt0": None}), 11)
    assert [item.code for item in report.issues] == ["cocos_unverifiable"]
    assert "bt0" in report.issues[0].message


def test_bulk_pressure_slope_is_used_rather_than_a_pointwise_derivative():
    """Sauter: dp/dpsi's 'main' sign, so a non-monotonic edge must not flip it."""
    import numpy as np

    from vaft.data.equilibrium import EquilibriumData
    from vaft.process.cocos import cocos_consistency_signs

    eq = _consistent_equilibrium(11)
    pressure = np.asarray(eq.pressure).copy()
    pressure[-3:] = pressure[-3:] + 5.0  # a small non-monotonic edge feature
    bumped = EquilibriumData(**{**eq.__dict__, "pressure": pressure})
    assert cocos_consistency_signs(bumped)["pprime"] == cocos_consistency_signs(eq)["pprime"]


def test_a_stale_case_header_declares_a_convention_the_signs_contradict():
    """Regression: two packaged g-files carry `COCOS=02` in CASE but are not COCOS 2.

    They are VAFT's own CHEASE outputs, re-signed back to the input pattern by
    `output_cocos="input"` without the header being rewritten. `as_equilibrium`
    promotes the CASE token straight to an explicit index, so the contradiction
    is currently accepted in silence; Eq. 23 catches it.
    """
    from vaft.data.eqdsk import read_geqdsk
    from vaft.data.resources import data_path, require_repository_sample
    from vaft.process.equilibrium import as_equilibrium
    from vaft.process.cocos import validate_cocos

    for name in ("efit/g040330.00320", "kineticEfit/g048224.00300.chease"):
        geqdsk = read_geqdsk(require_repository_sample(data_path(name)))
        assert "COCOS=02" in str(geqdsk.mapping["CASE"]), name
        equilibrium = as_equilibrium(geqdsk)
        assert equilibrium.convention.cocos == 2, name
        assert equilibrium.convention.source == "GEQDSK header", name

        declared = validate_cocos(equilibrium, 2)
        assert not declared.valid, f"{name}: the declared index should not validate"
        assert {"cocos_sign_dpsi", "cocos_sign_pprime"} <= {i.code for i in declared.issues}
        # The index its observable signs actually support.
        assert validate_cocos(equilibrium, 7).valid, name


def test_the_packaged_efit_sample_validates_in_the_family_its_signs_identify():
    """g039915 identifies as {1,2,11,12}; Eq. 23 must accept those and reject others."""
    from vaft.data.resources import sample_geqdsk
    from vaft.process.equilibrium import as_equilibrium
    from vaft.process.cocos import validate_cocos

    equilibrium = as_equilibrium(sample_geqdsk())
    assert equilibrium.convention.candidates == (1, 2, 11, 12)
    for cocos in (1, 2, 11, 12):
        assert validate_cocos(equilibrium, cocos).valid, cocos
    for cocos in (3, 4, 7, 8):
        assert not validate_cocos(equilibrium, cocos).valid, cocos
