"""Scientific compatibility of declared representations (#2163, Cases A, B, C and E of #2160).

An assessment compares declarations: identity first, then unit, COCOS and
Fourier convention.  It never reads or changes data, and its outcomes are not
validation verdicts.
"""
from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from vaft.data.fourier import COIL_SECTOR_DECOMPOSITION, GPEC_HARMONIC, FourierConvention
from vaft.data.representation import (
    Compatibility,
    EvidenceLevel,
    Representation,
    assess_compatibility,
    canonical_quantity,
    representation_from_convention,
)

C = Compatibility


# --- Case E: identity -------------------------------------------------------


@pytest.mark.parametrize("a, b", [
    ("line_averaged_density", "separatrix_density"),
    ("line_averaged_density", "volume_averaged_density"),
    ("edge_density", "separatrix_density"),
    ("beta_t", "beta_p"),
    ("beta_n", "beta_n_volume"),
    ("vacuum_perturbed_field", "plasma_response_field"),
    ("psi", "psi_n"),
    ("q95", "estimated_q95"),
])
def test_same_unit_different_quantity_is_incompatible(a, b):
    unit = "-" if a.startswith(("beta", "psi_n", "q")) and a != "psi" else "1e19 m^-3"
    unit = "Wb" if a == "psi" else unit
    unit = "T" if a.endswith("field") else unit
    result = assess_compatibility(Representation(a, unit), Representation(b, unit))
    assert result.status is C.INCOMPATIBLE
    assert not result.transformations


def test_family_members_are_reported_as_distinct_not_as_synonyms():
    result = assess_compatibility(Representation("line_averaged_density", "1e19 m^-3"),
                                  Representation("separatrix_density", "1e19 m^-3"))
    assert "density_average" in result.reasons[0] and "not one quantity" in result.reasons[0]


def test_strict_aliases_are_the_same_quantity():
    result = assess_compatibility(Representation("beta_toroidal", "-"), Representation("beta_t", "-"))
    assert result.status is C.COMPATIBLE
    assert canonical_quantity("I_p") == "plasma_current"
    assert canonical_quantity("ne") == "electron_density"


@pytest.mark.parametrize("term", ["beta", "density_average", "equilibrium", "Ip_MA", "pressure", None, ""])
def test_families_non_quantities_and_unknown_terms_never_resolve(term):
    assert canonical_quantity(term) is None
    result = assess_compatibility(Representation(term, "-"), Representation("beta_n", "-"))
    assert result.status is C.UNRESOLVED


def test_the_electron_density_subject_is_not_any_of_the_averaged_densities():
    result = assess_compatibility(Representation("ne", "1e19 m^-3"),
                                  Representation("line_averaged_density", "1e19 m^-3"))
    assert result.status is C.INCOMPATIBLE


# --- Case B: engineering units ----------------------------------------------------


def test_greenwald_current_in_amperes_requires_an_explicit_scale():
    from vaft.formula.catalog import parameter_representation

    expected = parameter_representation("greenwald_density", "I_p")
    assert expected.unit == "MA" and canonical_quantity(expected.quantity) == "plasma_current"
    result = assess_compatibility(Representation("plasma_current", "A"), expected)
    assert result.status is C.REQUIRES_TRANSFORM
    (transform,) = result.transformations
    assert transform.kind == "unit" and transform.provider == "vaft.formula.units.convert_unit"
    assert transform.parameters["factor"] == pytest.approx(1e-6)


def test_a_parameter_whose_name_is_not_vocabulary_has_no_quantity_until_declared():
    from vaft.formula.catalog import parameter_representation

    minor = parameter_representation("greenwald_density", "a")
    assert minor.quantity is None and minor.unit == "m"
    assert assess_compatibility(Representation("minor_radius", "m"), minor).status is C.UNRESOLVED
    declared = parameter_representation("greenwald_density", "a", quantity="minor_radius")
    assert assess_compatibility(Representation("minor_radius", "m"), declared).status is C.COMPATIBLE
    with pytest.raises(KeyError):
        parameter_representation("greenwald_density", "n_e")


def test_beta_fraction_and_percent_are_one_quantity_in_two_scales():
    result = assess_compatibility(Representation("beta_t", "%"), Representation("beta_t", "-"))
    assert result.status is C.REQUIRES_TRANSFORM
    assert result.transformations[0].parameters["factor"] == pytest.approx(0.01)


def test_normalized_beta_units_are_not_beta_units():
    result = assess_compatibility(Representation("beta_n", "% m T/MA"), Representation("beta_n", "-"))
    assert result.status is C.INCOMPATIBLE


@pytest.mark.parametrize("source, target", [(None, "MA"), ("MA", None), ("varies", "MA"), ("furlong", "MA")])
def test_missing_or_unknown_units_are_unresolved(source, target):
    result = assess_compatibility(Representation("plasma_current", source), Representation("plasma_current", target))
    assert result.status is C.UNRESOLVED
    assert result.unresolved


def test_the_formula_kernel_does_not_check_anything():
    from vaft.formula.stability import greenwald_density

    # Passing amperes is still a silent 1e6 error inside the kernel; the assessment
    # is the opt-in guard, the kernel stays a plain float function.
    assert greenwald_density(1.0e5, 0.3) == pytest.approx(1e6 * greenwald_density(0.1, 0.3))


# --- Case A: COCOS ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample():
    from vaft.data.resources import sample_geqdsk
    from vaft.process.equilibrium import as_equilibrium

    return as_equilibrium(sample_geqdsk(), convention=1)


def test_flux_per_radian_to_weber_goes_through_cocos(sample):
    from vaft.process.equilibrium import convert_cocos

    source = representation_from_convention(sample.convention)
    assert (source.cocos, source.unit, source.evidence) == (1, "Wb/rad", EvidenceLevel.INFERRED)
    target = Representation("psi", "Wb", cocos=11)
    result = assess_compatibility(source, target)
    assert result.status is C.REQUIRES_TRANSFORM
    (transform,) = result.transformations
    assert transform.provider == "vaft.process.equilibrium.convert_cocos"
    assert transform.parameters == {"source_cocos": 1, "target_cocos": 11}

    # Executing the named provider is a separate call; it keeps the invariants.
    converted = convert_cocos(sample, transform.parameters["target_cocos"])
    span = sample.psi_boundary - sample.psi_axis
    assert abs(converted.psi_boundary - converted.psi_axis) == pytest.approx(2 * math.pi * abs(span))
    assert abs(converted.ip) == pytest.approx(abs(sample.ip))
    np.testing.assert_allclose(np.abs(converted.q), np.abs(sample.q))
    back = convert_cocos(converted, 1)
    np.testing.assert_allclose(back.psi, sample.psi, rtol=1e-12, atol=0)
    np.testing.assert_allclose(back.q, sample.q, rtol=1e-12, atol=0)
    assert representation_from_convention(converted.convention).cocos == 11


def test_flux_units_without_cocos_are_not_a_scale_conversion():
    result = assess_compatibility(Representation("psi", "Wb/rad"), Representation("psi", "Wb"))
    assert result.status is C.UNRESOLVED
    assert "COCOS" in result.unresolved[0]


def test_a_unit_that_contradicts_its_cocos_is_unresolved():
    result = assess_compatibility(Representation("psi", "Wb", cocos=1), Representation("psi", "Wb", cocos=11))
    assert result.status is C.UNRESOLVED
    assert any("contradicts COCOS 1" in item for item in result.unresolved)


def test_one_sided_cocos_is_unresolved():
    result = assess_compatibility(Representation("psi", "Wb", cocos=11), Representation("psi", "Wb"))
    assert result.status is C.UNRESOLVED


def test_a_contradicted_convention_is_unresolved_and_never_transformed(sample):
    stale = replace(sample.convention, cocos=3, candidates=(3,))
    assert stale.contradicted
    rep = representation_from_convention(stale)
    assert rep.evidence is EvidenceLevel.UNRESOLVED and rep.cocos is None
    result = assess_compatibility(rep, Representation("psi", "Wb", cocos=11))
    assert result.status is C.UNRESOLVED
    assert not any(t.kind == "cocos" for t in result.transformations)


def test_a_declaration_without_identification_stays_declared(sample):
    declared = replace(sample.convention, identified=())
    assert representation_from_convention(declared).evidence is EvidenceLevel.DECLARED


def test_compatible_does_not_mean_verified():
    a = Representation("psi", "Wb", cocos=11, evidence=EvidenceLevel.DECLARED)
    b = Representation("psi", "Wb", cocos=11, evidence=EvidenceLevel.VERIFIED)
    result = assess_compatibility(a, b)
    assert result.status is C.COMPATIBLE
    assert result.evidence is EvidenceLevel.DECLARED


def test_vest_polarity_is_not_resolved_by_a_cocos_assessment():
    from vaft.machine_mapping.conventions import equilibrium_orientation_is_resolved

    # A clean COCOS 11 -> 11 assessment says nothing about #288.
    assert assess_compatibility(Representation("plasma_current", "A", cocos=11),
                                Representation("plasma_current", "A", cocos=11)).status is C.COMPATIBLE
    assert equilibrium_orientation_is_resolved() is False


def test_psi_is_per_radian_agrees_with_the_omas_spec():
    from vaft.data.cocos import COCOS_INDICES, cocos_spec, psi_is_per_radian

    for index in COCOS_INDICES:
        assert psi_is_per_radian(index) == cocos_spec(index).psi_per_radian
    with pytest.raises(ValueError):
        psi_is_per_radian(9)


# --- Case C: toroidal harmonics ------------------------------------------------------------


def test_sector_coefficients_need_conjugation_and_doubling_for_gpec():
    result = assess_compatibility(Representation("vacuum_perturbed_field", "T", fourier=COIL_SECTOR_DECOMPOSITION),
                                  Representation("vacuum_perturbed_field", "G", fourier=GPEC_HARMONIC))
    assert result.status is C.REQUIRES_TRANSFORM
    kinds = {t.kind: t for t in result.transformations}
    assert kinds["unit"].parameters["factor"] == pytest.approx(1e4)
    assert kinds["fourier"].parameters == {"source": "coil_sector_decomposition", "target": "gpec_harmonic",
                                           "conjugate": True, "factor": 2.0}


def test_unknown_phase_origin_or_one_sided_fourier_is_unresolved():
    unknown = FourierConvention("unknown_origin", +1, 2.0, phase_origin=None)
    for target in (Representation("vacuum_perturbed_field", "T", fourier=unknown),
                   Representation("vacuum_perturbed_field", "T")):
        result = assess_compatibility(Representation("vacuum_perturbed_field", "T", fourier=GPEC_HARMONIC), target)
        assert result.status is C.UNRESOLVED


# --- invariants of the assessment itself ------------------------------------------------------


def test_assessment_never_modifies_its_inputs_and_serializes():
    import json

    a = Representation("plasma_current", "A")
    b = Representation("plasma_current", "MA")
    before = (a, b)
    result = assess_compatibility(a, b)
    assert (a, b) == before
    json.dumps(result.as_dict())


def test_assessment_loads_no_solver_layer_or_units_library():
    # The canonical vocabulary lives in vaft.plot.taxonomy, and importing it loads
    # the plotting package; that cost is accepted because the assessment is opt-in
    # and never on a numerical path.  OMAS, Pint and the processing layer stay out.
    code = (
        "import sys\n"
        "from vaft.data.representation import Representation, assess_compatibility\n"
        "assess_compatibility(Representation('plasma_current', 'A'), Representation('ip', 'MA'))\n"
        "heavy = [m for m in ('omas', 'pint', 'vaft.process', 'vaft.formula.stability') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    subprocess.run([sys.executable, "-P", "-c", code], check=True)


# --- the boundary vocabulary links to the taxonomy ----------------------------------------------


def test_every_boundary_concept_is_a_canonical_quantity():
    from vaft.formula import boundaries

    quantities = {}
    for boundary in map(boundaries.get_boundary, boundaries.list_boundaries()):
        for quantity in (boundary.target, *boundary.inputs):
            quantities[(quantity.name, quantity.unit)] = quantity
    linked = [q for q in quantities.values() if q.concept is not None]
    assert linked
    for quantity in linked:
        assert canonical_quantity(quantity.concept) == quantity.concept, quantity


def test_the_two_line_density_units_are_one_quantity_but_not_the_same_axis():
    from vaft.formula import boundaries

    every = [q for b in map(boundaries.get_boundary, boundaries.list_boundaries()) for q in (b.target, *b.inputs)]
    quantities = {}
    for quantity in every:
        if quantity.name == "line_average_density":
            quantities[quantity.unit] = quantity
    e19, e20 = quantities["1e19 m^-3"], quantities["1e20 m^-3"]
    assert not boundaries.same_quantity(e19, e20)
    assert boundaries.quantity_conversion_factor(e20, e19) == pytest.approx(10.0)
    edge = next(q for q in every if q.name == "edge_density")
    with pytest.raises(ValueError, match="different quantities"):
        boundaries.quantity_conversion_factor(edge, e20)
