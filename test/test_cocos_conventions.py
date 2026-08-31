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
    """g039915's signs support the 1/2/11/12 family; Eq. 23 must reject the rest.

    Identification narrows further, to (1, 2), because Ampere's law settles the
    flux exponent -- but Eq. 23 tests only the sign relations, which cannot
    distinguish an index from its +10 counterpart.
    """
    from vaft.data.resources import sample_geqdsk
    from vaft.process.equilibrium import as_equilibrium
    from vaft.process.cocos import validate_cocos

    equilibrium = as_equilibrium(sample_geqdsk())
    assert equilibrium.convention.candidates == (1, 2)
    for cocos in (1, 2, 11, 12):
        assert validate_cocos(equilibrium, cocos).valid, cocos
    for cocos in (3, 4, 7, 8):
        assert not validate_cocos(equilibrium, cocos).valid, cocos


# --- Sauter Eq. 20: the poloidal field ------------------------------------


def test_poloidal_field_factor_falls_back_to_the_historical_form():
    """cocos=None must keep B_R = -(1/R) dpsi/dZ so untouched callers are unchanged."""
    from vaft.formula.equilibrium import poloidal_field_factor

    assert poloidal_field_factor(None) == -1.0
    # which is the COCOS 2/3/6/7 form, the one the codebase always assumed
    for index in (2, 3, 6, 7):
        assert poloidal_field_factor(index) == -1.0


def test_poloidal_field_has_opposite_sign_in_cocos_1_and_cocos_2():
    """The sign the old code could not distinguish.

    COCOS 1 and 2 store psi with opposite sign, so a formula that hardcodes
    B_R = -(1/R) dpsi/dZ returns opposite fields for the same physical
    equilibrium.  The packaged VEST g-file is COCOS 1 or 2 depending only on
    `clockwise_phi`, so this was a coin flip on real data.
    """
    from vaft.formula.equilibrium import poloidal_field_factor

    assert poloidal_field_factor(1) == -poloidal_field_factor(2)
    assert poloidal_field_factor(11) == -poloidal_field_factor(12)


def test_grid_fields_agree_across_conventions_including_sign():
    """The same physical equilibrium must give the same B in every convention.

    Before Eq. 20 was applied, only |B| agreed: the components flipped between
    the COCOS 1 and COCOS 2 representations because psi flipped but the field
    formula did not.
    """
    import numpy as np

    from vaft.data.resources import sample_geqdsk
    from vaft.process.equilibrium import as_equilibrium, convert_cocos
    from vaft.process._equilibrium_parametric import _grid_fields

    base = as_equilibrium(sample_geqdsk(), convention=1)
    _, _, br_ref, bz_ref, bp_ref = _grid_fields(base)
    assert np.any(np.abs(br_ref) > 1e-6), "the fixture must carry a real field"
    for target in (2, 3, 5, 8, 11, 12, 13, 18):
        _, _, br, bz, bp = _grid_fields(convert_cocos(base, target))
        scale = float(np.nanmax(np.abs(br_ref)))
        assert np.nanmax(np.abs(br - br_ref)) < 1e-9 * scale, target
        assert np.nanmax(np.abs(bz - bz_ref)) < 1e-9 * float(np.nanmax(np.abs(bz_ref))), target
        assert np.nanmax(np.abs(bp - bp_ref)) < 1e-9 * float(np.nanmax(np.abs(bp_ref))), target


def test_dimensionless_descriptors_stay_invariant_across_conventions():
    """beta_p and li are dimensionless, so no convention may move them."""
    from vaft.data.resources import sample_geqdsk
    from vaft.process.equilibrium import as_equilibrium, convert_cocos, derive_global_descriptors

    base = as_equilibrium(sample_geqdsk(), convention=1)
    reference = derive_global_descriptors(base)
    for target in (2, 3, 5, 8, 11, 12, 13, 18):
        other = derive_global_descriptors(convert_cocos(base, target))
        for name in ("beta_p_boundary_average", "li_virial", "alpha", "beta_t", "s1", "s2", "s3"):
            assert other[name].value == pytest.approx(reference[name].value, rel=1e-9), (name, target)


def test_field_magnitude_is_invariant_while_components_carry_the_orientation():
    """|B_pol| never depends on the convention; the components do."""
    import numpy as np

    from vaft.formula.equilibrium import poloidal_field_factor

    dpsi_dr, dpsi_dz, r = 0.37, -0.11, 0.62
    magnitudes = set()
    for index in (1, 2, 3, 11, 12, 13):
        k = poloidal_field_factor(index)
        b_r, b_z = k * dpsi_dz / r, -k * dpsi_dr / r
        magnitudes.add(round(float(np.hypot(b_r, b_z)) * abs(1.0 / k), 9))
    # After removing each convention's own scale, one physical magnitude remains.
    assert len(magnitudes) == 1


# --- Recording the convention on an ODS -----------------------------------


def test_an_unlabelled_ods_reports_no_convention_rather_than_guessing():
    from omas import ODS

    from vaft.omas.general import ods_cocos

    assert ods_cocos(ODS()) is None
    assert ods_cocos(ODS(), default=11) == 11


def test_the_convention_is_recorded_where_data_dictionary_3_can_hold_it():
    """DD 3.x has no ids_properties.cocos, which is why nothing ever wrote it.

    The field arrives in DD 4; until then the convention lives on
    equilibrium.code.parameters, alongside CHEASE's metrics and EFIT's
    auxiliary quantities.
    """
    import pytest as _pytest
    from omas import ODS

    from vaft.omas.general import COCOS_PARAMETER_PATH, ods_cocos, set_ods_cocos

    ods = ODS()
    with _pytest.raises(Exception):
        ods["equilibrium.ids_properties.cocos"] = 11

    set_ods_cocos(ods, 11, source="test")
    assert ods[COCOS_PARAMETER_PATH] == 11
    assert ods_cocos(ods) == 11


def test_a_recorded_convention_survives_a_save_and_load():
    import pathlib
    import tempfile

    from omas import ODS

    from vaft.omas.general import ods_cocos, set_ods_cocos

    ods = ODS()
    set_ods_cocos(ods, 2)
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "labelled.json"
        ods.save(str(path))
        reloaded = ODS()
        reloaded.load(str(path))
    assert ods_cocos(reloaded) == 2


def test_recording_an_undefined_convention_is_refused():
    from omas import ODS

    from vaft.omas.general import set_ods_cocos

    with pytest.raises(ValueError, match="not defined"):
        set_ods_cocos(ODS(), 9)


# --- Identification -------------------------------------------------------


def test_the_flux_exponent_is_decided_by_amperes_law():
    """loop(B_p dl) = mu0*|Ip| separates a weber psi from a weber/radian one.

    Computing B_p as if psi were weber/radian gives a ratio of 1 when that
    holds and 2*pi when it does not, so the two outcomes are a factor 2*pi
    apart rather than a marginal comparison.
    """
    import numpy as np

    from vaft.data.resources import sample_geqdsk
    from vaft.process.cocos import identify_flux_exponent
    from vaft.process.equilibrium import as_equilibrium, convert_cocos

    per_radian = as_equilibrium(sample_geqdsk(), convention=1)
    exponent, ratio = identify_flux_exponent(per_radian)
    assert exponent == 0
    assert ratio == pytest.approx(1.0, abs=0.05)

    weber = convert_cocos(per_radian, 11)
    exponent, ratio = identify_flux_exponent(weber)
    assert exponent == 1
    assert ratio == pytest.approx(2 * np.pi, rel=0.05)


def test_the_flux_exponent_is_unavailable_rather_than_guessed():
    from vaft.data.equilibrium import EquilibriumData
    from vaft.process.cocos import identify_flux_exponent

    assert identify_flux_exponent(EquilibriumData()) == (None, None)


def test_identification_reaches_a_single_index_once_the_machine_phi_is_known():
    """The sign family comes from the data; clockwise_phi is a machine fact."""
    from vaft.data.resources import sample_geqdsk
    from vaft.process.cocos import identify_convention
    from vaft.process.equilibrium import as_equilibrium

    equilibrium = as_equilibrium(sample_geqdsk())
    assert identify_convention(equilibrium) == (1, 2)
    assert identify_convention(equilibrium, clockwise_phi=True) == (2,)
    assert identify_convention(equilibrium, clockwise_phi=False) == (1,)


def test_identification_recognises_a_weber_psi_after_conversion():
    """The same equilibrium in COCOS 11 must identify in the 11-18 family."""
    from vaft.data.resources import sample_geqdsk
    from vaft.process.cocos import identify_convention
    from vaft.process.equilibrium import as_equilibrium, convert_cocos

    weber = convert_cocos(as_equilibrium(sample_geqdsk(), convention=1), 11)
    assert identify_convention(weber) == (11, 12)


def test_identification_is_independent_of_psi_profile_storage_order():
    """omas.identify_cocos reads sign(gradient(psi))[0], so order matters to it.

    A boundary-to-axis profile -- which ODS data can legitimately be -- would
    otherwise invert sigma_Bp and select the wrong family.
    """
    from vaft.data.equilibrium import EquilibriumData
    from vaft.data.resources import sample_geqdsk
    from vaft.process.cocos import identify_convention
    from vaft.process.equilibrium import as_equilibrium

    equilibrium = as_equilibrium(sample_geqdsk())
    reversed_storage = EquilibriumData(**{
        **equilibrium.__dict__,
        "psi_1d": equilibrium.psi_1d[::-1],
        "q": equilibrium.q[::-1],
    })
    assert identify_convention(reversed_storage) == identify_convention(equilibrium) == (1, 2)


def test_the_stale_case_files_identify_as_the_family_equation_23_accepts():
    """The CASE header says COCOS 2; the signs and Eq. 23 both say 7."""
    from vaft.data.eqdsk import read_geqdsk
    from vaft.data.resources import data_path, require_repository_sample
    from vaft.process.cocos import identify_convention, validate_cocos
    from vaft.process.equilibrium import as_equilibrium

    for name in ("efit/g040330.00320", "kineticEfit/g048224.00300.chease"):
        equilibrium = as_equilibrium(read_geqdsk(require_repository_sample(data_path(name))))
        candidates = identify_convention(equilibrium)
        assert candidates == (7, 8), name
        assert 2 not in candidates, f"{name}: the declared index is not supported by the data"
        for candidate in candidates:
            assert validate_cocos(equilibrium, candidate).valid, (name, candidate)


def test_identification_returns_nothing_when_the_inputs_are_missing():
    from vaft.data.equilibrium import EquilibriumData
    from vaft.process.cocos import identify_convention

    assert identify_convention(EquilibriumData()) == ()


def test_a_declared_convention_the_data_contradicts_is_reported_not_trusted_silently():
    """Regression for the stale `COCOS=02` CASE token.

    `_from_geqdsk` promoted the header straight to an explicit index and
    stopped there.  Identification now always runs, so the declaration still
    wins -- a caller asserting a convention is taken at their word -- but the
    disagreement is recorded and surfaces as a validation warning.
    """
    from vaft.data.eqdsk import read_geqdsk
    from vaft.data.resources import data_path, require_repository_sample
    from vaft.process.equilibrium import as_equilibrium, validate_equilibrium

    equilibrium = as_equilibrium(read_geqdsk(require_repository_sample(data_path("efit/g040330.00320"))))
    assert equilibrium.convention.cocos == 2
    assert equilibrium.convention.identified == (7, 8)
    assert equilibrium.convention.contradicted

    issues = {item.code: item for item in validate_equilibrium(equilibrium).issues}
    assert "cocos_declared_conflicts_with_signs" in issues
    message = issues["cocos_declared_conflicts_with_signs"].message
    assert "COCOS 2 is declared" in message and "(7, 8)" in message


def test_an_explicit_convention_that_contradicts_the_file_is_also_reported():
    """Asserting COCOS 11 on a weber-per-radian g-file is a real inconsistency."""
    from vaft.data.resources import sample_geqdsk
    from vaft.process.equilibrium import as_equilibrium

    equilibrium = as_equilibrium(sample_geqdsk(), convention=11)
    assert equilibrium.convention.cocos == 11
    assert equilibrium.convention.identified == (1, 2)
    assert equilibrium.convention.contradicted


def test_a_consistent_declaration_is_not_flagged():
    from vaft.data.resources import sample_geqdsk
    from vaft.process.equilibrium import as_equilibrium, validate_equilibrium

    for cocos in (1, 2):
        equilibrium = as_equilibrium(sample_geqdsk(), convention=cocos)
        assert not equilibrium.convention.contradicted, cocos
        codes = {item.code for item in validate_equilibrium(equilibrium).issues}
        assert "cocos_declared_conflicts_with_signs" not in codes, cocos


# --- The d/dpsi profiles --------------------------------------------------


def test_pprime_and_ffprime_are_carried_rather_than_re_derived():
    """They were dropped on import and recomputed by np.gradient on export."""
    import numpy as np

    from vaft.data.eqdsk import from_equilibrium
    from vaft.data.resources import sample_geqdsk
    from vaft.process.equilibrium import as_equilibrium

    geqdsk = sample_geqdsk()
    equilibrium = as_equilibrium(geqdsk, convention=1)
    assert equilibrium.pprime is not None and equilibrium.ffprime is not None

    exported = from_equilibrium(equilibrium)
    for key in ("PPRIME", "FFPRIM", "PRES", "FPOL", "QPSI"):
        np.testing.assert_array_equal(
            np.asarray(exported[key], dtype=float), np.asarray(geqdsk[key], dtype=float), err_msg=key
        )


def test_the_dpsi_profiles_transform_by_the_inverse_of_psi():
    """convert_cocos applied PSI, F, Q, IP and BT but never PPRIME or F_FPRIME.

    dp/dpsi and F dF/dpsi are derivatives with respect to psi, so they scale by
    1/PSI.  Leaving them untransformed silently mixed conventions inside one
    equilibrium.
    """
    import numpy as np

    from vaft.data.resources import sample_geqdsk
    from vaft.process.equilibrium import as_equilibrium, convert_cocos

    equilibrium = as_equilibrium(sample_geqdsk(), convention=1)
    weber = convert_cocos(equilibrium, 11)
    # PSI picks up 2*pi going 1 -> 11, so the derivatives must lose it.
    ratio = weber.psi_1d[1] / equilibrium.psi_1d[1]
    assert ratio == pytest.approx(2 * np.pi, rel=1e-9)
    np.testing.assert_allclose(weber.pprime * ratio, equilibrium.pprime, rtol=1e-9)
    np.testing.assert_allclose(weber.ffprime * ratio, equilibrium.ffprime, rtol=1e-9)


@pytest.mark.parametrize("target", [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18])
def test_a_conversion_round_trip_restores_every_profile(target):
    """11 -> k -> 11 must return the original, derivatives included."""
    import numpy as np

    from vaft.data.resources import sample_geqdsk
    from vaft.process.equilibrium import as_equilibrium, convert_cocos

    start = convert_cocos(as_equilibrium(sample_geqdsk(), convention=1), 11)
    restored = convert_cocos(convert_cocos(start, target), 11)
    for name in ("psi", "psi_1d", "pressure", "f", "q", "pprime", "ffprime"):
        np.testing.assert_allclose(
            getattr(restored, name), getattr(start, name), rtol=1e-12, err_msg=name
        )
    assert restored.ip == pytest.approx(start.ip)
    assert restored.bt0 == pytest.approx(start.bt0)


def test_equation_23_uses_the_carried_pprime_when_it_is_available():
    """With a real dp/dpsi profile the check no longer needs the bulk slope."""
    from vaft.data.resources import sample_geqdsk
    from vaft.process.cocos import cocos_consistency_signs
    from vaft.process.equilibrium import as_equilibrium

    equilibrium = as_equilibrium(sample_geqdsk(), convention=1)
    assert equilibrium.pprime is not None
    assert cocos_consistency_signs(equilibrium)["pprime"] == -1


def test_the_solovev_model_carries_its_convention_and_its_constant_sources():
    """Regression: EquilibriumData was built positionally there.

    Adding pprime/ffprime to the model silently shifted every argument after q,
    so the convention landed in r0 and the equilibrium came back unlabelled.
    """
    import numpy as np

    from scipy.constants import mu_0 as MU0

    from vaft.data.equilibrium import SolovevEquilibrium
    from vaft.process.equilibrium import solovev_to_equilibrium

    pprime = -1.0e5
    c1 = -0.02
    c2 = -4.0 * (-MU0 * pprime / 8.0) - 2.0 * c1
    model = SolovevEquilibrium(np.array([0.0, c1, c2, 0.0, 0.0]), pprime, 0.0, 1.0, psi_boundary=-0.002)
    equilibrium = solovev_to_equilibrium(model, np.linspace(0.5, 1.5, 151), np.linspace(-0.7, 0.7, 151))

    assert equilibrium.convention.cocos == 11
    assert equilibrium.r0 == pytest.approx(1.0)
    assert equilibrium.bt0 == pytest.approx(model.f_boundary / model.rref)
    # solovev_to_equilibrium emits the requested convention, so psi carries the
    # 2*pi for COCOS 11 and the d/dpsi sources carry its inverse.
    np.testing.assert_allclose(equilibrium.pprime, pprime / (2 * np.pi))
    np.testing.assert_allclose(equilibrium.ffprime, 0.0)

    per_radian = solovev_to_equilibrium(
        model, np.linspace(0.5, 1.5, 151), np.linspace(-0.7, 0.7, 151), convention=1
    )
    np.testing.assert_allclose(equilibrium.psi, per_radian.psi * 2 * np.pi)
    np.testing.assert_allclose(per_radian.pprime, pprime)


# --- Adapters declaring their conventions ---------------------------------


def test_the_chease_target_is_derived_from_the_registry_not_hand_maintained():
    """The five signs were a literal dict; they follow from COCOS 2 and Eq. 23."""
    from vaft.code.chease import (
        CHEASE_COCOS02_SIGNS,
        CHEASE_ORIENTATION,
        _desired_signs_for_cocos,
    )
    from vaft.data.cocos import convention_for

    assert convention_for("chease").cocos == 2
    assert CHEASE_ORIENTATION == {"sigma_ip": -1, "sigma_b0": +1}
    assert CHEASE_COCOS02_SIGNS == _desired_signs_for_cocos(2, **CHEASE_ORIENTATION)
    # The values themselves must not have moved: CHEASE's input is byte-checked.
    assert CHEASE_COCOS02_SIGNS == {
        "dpsi": -1, "bcentr": 1, "current": -1, "fpol": 1, "q": -1,
    }


def test_a_different_convention_gives_a_different_sign_pattern():
    """Guards against the derivation collapsing to a constant."""
    from vaft.code.chease import _desired_signs_for_cocos

    assert _desired_signs_for_cocos(2, sigma_ip=-1, sigma_b0=1) != _desired_signs_for_cocos(
        3, sigma_ip=-1, sigma_b0=1
    )
    assert _desired_signs_for_cocos(2, sigma_ip=-1, sigma_b0=1) != _desired_signs_for_cocos(
        2, sigma_ip=+1, sigma_b0=1
    )


def test_the_vfit_psi_factor_is_the_declared_conversion_not_a_bare_two_pi():
    """A bare 2*pi asserts VFIT is COCOS 1: the 2 -> 11 factor is -2*pi."""
    import numpy as np
    from omas import cocos_transform

    from vaft.data.cocos import VAFT_INTERNAL_COCOS, convention_for
    from vaft.data.vfit import _TWO_PI

    convention = convention_for("vfit")
    assert convention.cocos == 1
    assert convention.confirmed is False, "the index has never been checked against Eq. 23"
    assert _TWO_PI == pytest.approx(
        cocos_transform(convention.cocos, VAFT_INTERNAL_COCOS)["PSI"]
    )
    # Unchanged numerically, so VFIT output does not move.
    assert _TWO_PI == pytest.approx(2.0 * np.pi)
    assert cocos_transform(2, VAFT_INTERNAL_COCOS)["PSI"] == pytest.approx(-2.0 * np.pi)


def test_a_geqdsk_conversion_labels_the_ods_when_the_convention_is_certain():
    """An unlabelled ODS is how weber-per-radian psi came to sit in weber slots."""
    from vaft.data.eqdsk import from_equilibrium
    from vaft.data.resources import sample_geqdsk
    from vaft.omas.general import ods_cocos
    from vaft.process.equilibrium import as_equilibrium

    certain = from_equilibrium(as_equilibrium(sample_geqdsk(), convention=2))
    assert ods_cocos(certain.to_omas()) == 2


def test_an_ambiguous_convention_is_left_unlabelled_rather_than_guessed():
    """Without clockwise_phi the VEST sample is COCOS 1 or 2; silence is honest."""
    from vaft.data.resources import sample_geqdsk
    from vaft.omas.general import ods_cocos
    from vaft.process.equilibrium import as_equilibrium

    assert as_equilibrium(sample_geqdsk()).convention.cocos is None
    assert ods_cocos(sample_geqdsk().to_omas()) is None


def test_a_contradicted_declaration_is_not_copied_into_a_new_artifact():
    """g040330 declares COCOS 2 in CASE and its signs support 7.

    Writing the declaration into the ODS would launder the contradiction into an
    artifact that no longer carries the evidence against it.
    """
    from vaft.data.eqdsk import read_geqdsk
    from vaft.data.resources import data_path, require_repository_sample
    from vaft.omas.general import ods_cocos
    from vaft.process.equilibrium import as_equilibrium

    geqdsk = read_geqdsk(require_repository_sample(data_path("efit/g040330.00320")))
    assert as_equilibrium(geqdsk).convention.contradicted
    assert ods_cocos(geqdsk.to_omas()) is None


def test_labelling_never_breaks_a_conversion():
    """Provenance is not a reason to fail; a degenerate g-file still converts."""
    import numpy as np

    from vaft.data.eqdsk import GEQDSK, to_omas

    minimal = GEQDSK({
        "CASE": "degenerate", "NW": 2, "NH": 2,
        "RDIM": 1.0, "ZDIM": 1.0, "RCENTR": 1.0, "RLEFT": 0.5, "ZMID": 0.0,
        "RMAXIS": 1.0, "ZMAXIS": 0.0, "SIMAG": 0.0, "SIBRY": 1.0,
        "BCENTR": 1.0, "CURRENT": 0.0,
        "FPOL": np.ones(2), "PRES": np.zeros(2), "FFPRIM": np.zeros(2),
        "PPRIME": np.zeros(2), "PSIRZ": np.zeros((2, 2)), "QPSI": np.ones(2),
        "NBBBS": 0, "LIMITR": 0,
        "RBBBS": np.array([]), "ZBBBS": np.array([]),
        "RLIM": np.array([]), "ZLIM": np.array([]),
    })
    ods = to_omas(minimal)
    assert "equilibrium.time_slice.0.profiles_1d.psi" in ods


# --- Review findings ------------------------------------------------------


def test_the_ods_virial_path_uses_one_convention_for_both_fields():
    """Regression: the boundary field was converted and the grid field was not.

    `compute_virial_equilibrium_quantities_ods` feeds both to
    `shafranov_integrals`.  With a COCOS 11 label the boundary field carried
    k = +1/(2*pi) while the grid field still carried k = -1, so the two differed
    by -2*pi inside one calculation and li came out about 17% wrong.  li, the
    Shafranov integrals and the diamagnetism are dimensionless, so the same
    equilibrium expressed in either convention must give the same numbers.
    """
    from vaft.data.eqdsk import from_equilibrium
    from vaft.data.resources import sample_geqdsk
    from vaft.omas.general import set_ods_cocos
    from vaft.omas.process_wrapper import compute_virial_equilibrium_quantities_ods
    from vaft.process.equilibrium import as_equilibrium, convert_cocos

    def virial(cocos):
        equilibrium = as_equilibrium(sample_geqdsk(), convention=1)
        if cocos != 1:
            equilibrium = convert_cocos(equilibrium, cocos)
        ods = from_equilibrium(equilibrium).to_omas()
        set_ods_cocos(ods, cocos)
        return compute_virial_equilibrium_quantities_ods(ods)[0]

    per_radian, weber = virial(1), virial(11)
    for name in ("li", "li_vir_lao", "s_1", "s_2", "s_3", "beta_pd_vir", "mui"):
        assert weber[name] == pytest.approx(per_radian[name], rel=1e-6), name


def test_the_field_interpolator_receives_the_ods_convention():
    """Regression: the interpolator gained a cocos parameter its caller ignored."""
    import inspect

    from vaft.omas import process_wrapper

    source = inspect.getsource(process_wrapper)
    index = source.index("make_equilibrium_field_interpolator(\n")
    call = source[index:index + 400]
    assert "cocos=" in call, "the production caller must declare the ODS convention"


def test_a_recorded_ods_convention_is_read_back_by_the_adapter():
    """Regression: the write side used code.parameters, the read side did not.

    `_from_ods` only looked at `ids_properties.cocos`, which cannot exist under
    DD 3.41, so an ODS VAFT had just labelled came back ambiguous and
    `convert_cocos` refused it.
    """
    from vaft.data.resources import sample_geqdsk
    from vaft.omas.general import set_ods_cocos
    from vaft.process.equilibrium import as_equilibrium, convert_cocos

    ods = sample_geqdsk().to_omas()
    set_ods_cocos(ods, 2)
    equilibrium = as_equilibrium(ods)
    assert equilibrium.convention.cocos == 2
    assert not equilibrium.convention.ambiguous
    assert convert_cocos(equilibrium, 11).convention.cocos == 11


@pytest.mark.parametrize("scale", [3.0, 3.7, 10.0])
def test_the_flux_exponent_abstains_when_the_ratio_is_near_neither_answer(scale):
    """The two answers are 2*pi apart, so anything between is a broken input.

    A bare nearest-neighbour would confidently pick one; with psi scaled by
    3, 3.7 or 10 there is no honest answer to give.
    """
    from vaft.data.equilibrium import EquilibriumData
    from vaft.data.resources import sample_geqdsk
    from vaft.process.cocos import identify_flux_exponent
    from vaft.process.equilibrium import as_equilibrium

    base = as_equilibrium(sample_geqdsk(), convention=1)
    scaled = EquilibriumData(**{
        **base.__dict__, "psi": base.psi * scale,
        "psi_axis": base.psi_axis * scale, "psi_boundary": base.psi_boundary * scale,
    })
    exponent, ratio = identify_flux_exponent(scaled)
    assert exponent is None, f"ratio {ratio} should not have produced an answer"
    assert ratio == pytest.approx(scale, rel=0.05)


def test_an_ip_that_disagrees_with_the_psi_map_does_not_produce_a_confident_index():
    """A 5x mismatch between Ip and the flux map is evidence of a broken file."""
    from vaft.data.equilibrium import EquilibriumData
    from vaft.data.resources import sample_geqdsk
    from vaft.process.cocos import identify_convention, identify_flux_exponent
    from vaft.process.equilibrium import as_equilibrium

    base = as_equilibrium(sample_geqdsk(), convention=1)
    wrong = EquilibriumData(**{**base.__dict__, "ip": base.ip * 0.2})
    assert identify_flux_exponent(wrong)[0] is None
    # Falls back to the sign family rather than naming a single index.
    assert len(identify_convention(wrong, clockwise_phi=True)) > 1


def test_a_real_x_point_survives_grid_refinement():
    """Regression: the flux window shrank as h^2, so refining lost X-points.

    psi_boundary is taken from the source and no reconstruction places it at the
    X-point flux to one part in a million, so the window needs a floor tied to
    that precision rather than to the grid alone.
    """
    import numpy as np

    from vaft.data.equilibrium import Contour, EquilibriumConvention, EquilibriumData, Topology
    from vaft.process.equilibrium import derive_boundary_representation, extract_flux_surface_contours

    def single_null(points, psi_boundary_error):
        c = 1.481
        r = np.linspace(0.5, 1.5, points)
        z = np.linspace(-0.8, 0.8, points + 40)
        rm, zm = np.meshgrid(r, z, indexing="ij")
        psi = (rm - 1) ** 2 + zm**2 - c * zm**3
        boundary = 4.0 / (27.0 * c**2) * (1.0 + psi_boundary_error)
        raw = extract_flux_surface_contours(psi, r, z, 0.0, boundary, [1.0])[1.0]
        rb, zb = max(raw, key=lambda pair: pair[0].size)
        theta = np.linspace(0, 2 * np.pi, 481, endpoint=False)
        eq = EquilibriumData(
            r=r, z=z, psi=psi, psi_axis=0.0, psi_boundary=boundary, magnetic_axis=(1.0, 0.0),
            lcfs=Contour(rb, zb),
            limiter=Contour(1.0 + 0.42 * np.cos(theta), 0.62 * np.sin(theta)),
            convention=EquilibriumConvention(1, (1,), True, False, 1, 1, 1, "test"),
        )
        return derive_boundary_representation(eq).topology

    # A boundary flux 0.01% off the separatrix is routine for a reconstruction.
    for points in (161, 321):
        assert single_null(points, -1e-4) is Topology.UPPER_SINGLE_NULL, points
    # And refining must not lose it.
    assert single_null(321, -1e-4) is single_null(161, -1e-4)


def test_the_flux_window_floor_does_not_readmit_real_reconstruction_artifacts():
    """The floor must stay far below the nearest artifact on real VEST data."""
    from vaft.data.resources import sample_geqdsk
    from vaft.process._equilibrium_parametric import BOUNDARY_FLUX_PRECISION
    from vaft.process.equilibrium import as_equilibrium, derive_boundary_representation

    assert BOUNDARY_FLUX_PRECISION < 0.07 / 10, "the nearest VEST artifact sits at psi_n ~ 1.073"
    boundary = derive_boundary_representation(as_equilibrium(sample_geqdsk(), convention=1))
    assert len(boundary.x_points) > 1
    assert not [point for point in boundary.x_points if point.active]


def test_a_contradicted_declaration_stays_visible_through_a_conversion():
    """Regression: convert_cocos rebuilt the convention without `identified`."""
    from vaft.data.eqdsk import read_geqdsk
    from vaft.data.resources import data_path, require_repository_sample
    from vaft.process.equilibrium import as_equilibrium, convert_cocos

    equilibrium = as_equilibrium(read_geqdsk(require_repository_sample(data_path("efit/g040330.00320"))))
    assert equilibrium.convention.contradicted
    converted = convert_cocos(equilibrium, 12)
    assert converted.convention.identified, "the evidence must survive the conversion"
    assert converted.convention.contradicted
