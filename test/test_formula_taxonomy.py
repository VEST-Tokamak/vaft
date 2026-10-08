"""Reduced-representation taxonomy (#1626): vocabulary, parsing, catalog exposure and filters."""

import pytest

from vaft.formula import catalog
from vaft.formula._docstring import parse_docstring
from vaft.formula._taxonomy import (
    FIELDS,
    LOCALITIES,
    PHYSICAL_ROLES,
    REDUCTION_FAMILIES,
    REDUCTION_KINDS,
    QUANTITIES,
    SPATIAL_REPRESENTATIONS,
    parse_reduction,
)

GOOD = "input: profile_1d\noutput: scalar_0d\nkind: feature_extraction\nlocality: global\nrole: profile_descriptor"


def test_a_valid_section_parses():
    reduction, errors = parse_reduction(GOOD)
    assert errors == ()
    assert reduction.input == ("profile_1d",) and reduction.output == "scalar_0d"
    assert reduction.mapping == "profile_1d -> scalar_0d"
    assert parse_reduction(None) == (None, ())
    multi, errors = parse_reduction(GOOD.replace("input: profile_1d", "input: field_2d, scalar_0d"))
    assert not errors and multi.input == ("field_2d", "scalar_0d")


@pytest.mark.parametrize("text, fragment", [
    (GOOD.replace("kind: feature_extraction", "kind: moments"), "kind 'moments'"),
    (GOOD.replace("locality: global", ""), "missing 'locality'"),
    (GOOD + "\nrole: global_descriptor", "given twice"),
    (GOOD + "\nscope: global", "key 'scope'"),
    (GOOD.replace("input: profile_1d", "input: profile"), "input 'profile'"),
    (GOOD.replace("output: scalar_0d", "output scalar_0d"), "not 'key: value'"),
])
def test_a_bad_section_yields_no_reduction_and_says_why(text, fragment):
    reduction, errors = parse_reduction(text)
    assert reduction is None
    assert any(fragment in e for e in errors), errors


def test_vocabularies_are_disjoint_and_unique():
    for vocab in (SPATIAL_REPRESENTATIONS, REDUCTION_KINDS, LOCALITIES, PHYSICAL_ROLES):
        assert len(set(vocab)) == len(vocab)
    assert FIELDS == ("input", "output", "kind", "locality", "role")


def _docstring(reduction: str, unit: str) -> str:
    body = "\n".join("    " + line for line in reduction.splitlines())
    return f'''Summary.

    Returns
    -------
    float
        A thing [{unit}].

    Reduction
    ---------
{body}
    '''


def test_a_dimensionless_reduction_needs_a_dimensionless_return():
    text = GOOD.replace("kind: feature_extraction", "kind: dimensionless_normalization")
    reduction, errors = catalog._reduction(parse_docstring(_docstring(text, "m")))
    assert reduction is None and any("dimensionless" in e for e in errors)
    reduction, errors = catalog._reduction(parse_docstring(_docstring(text, "-")))
    assert reduction is not None and errors == ()


def test_every_catalogued_reduction_is_valid_and_consistent():
    specs = [s for s in catalog.list_formulas() if s.reduction is not None]
    assert len(specs) >= 20
    for spec in specs:
        assert not [e for e in spec.errors if "Reduction" in e], spec.qualname
        r = spec.reduction
        assert set(r.input) <= set(SPATIAL_REPRESENTATIONS) and r.output in SPATIAL_REPRESENTATIONS
        if r.role == "similarity_coordinate":  # a similarity coordinate is dimensionless
            assert spec.returns[0].unit == "-", spec.qualname
        assert "Reduction" not in [s["title"] for s in spec.as_dict()["sections"]]
        assert spec.as_dict()["reduction"]["kind"] == r.kind


def test_dimensionless_does_not_mean_zero_dimensional():
    profiles = catalog.list_formulas(reduction_kind="dimensionless_normalization", output_representation="profile_1d")
    scalars = catalog.list_formulas(reduction_kind="dimensionless_normalization", output_representation="scalar_0d")
    assert profiles and scalars
    assert {s.reduction.locality for s in profiles} == {"flux_surface_local"}


def test_l_i_is_a_quadratic_integral_of_b_p_not_a_moment_of_j():
    li = catalog.describe("virial.virial_li_from_volume").reduction
    assert li.kind == "quadratic_integral" and li.input == ("field_2d", "scalar_0d")  # B_p samples; B_pa, Omega


def test_filters_and_their_validation():
    everything = catalog.list_formulas()
    kinds = catalog.list_formulas(reduction_kind="differential")
    assert kinds and all(s.reduction.kind == "differential" for s in kinds)
    assert len(kinds) < len(everything)
    assert catalog.list_formulas(locality="global", role="similarity_coordinate")
    with pytest.raises(ValueError):
        catalog.list_formulas(reduction_kind="moments")
    with pytest.raises(ValueError):
        catalog.list_formulas(output_representation="0d")


def test_formulas_without_a_section_report_none():
    spec = catalog.describe("stability.helical_phase")
    assert spec.reduction is None and spec.as_dict()["reduction"] is None


def test_snapshot_carries_the_reduction():
    rows = catalog.documentation_snapshot()["formulas"]
    tagged = [r for r in rows if r["reduction"]]
    assert tagged and all(set(r["reduction"]) == {"input", "output", "kind", "locality", "role", "mapping"}
                          for r in tagged)


def test_family_edges_name_catalogued_reductions_or_state_a_kind():
    for family, relations in REDUCTION_FAMILIES.items():
        for rel in relations:
            assert rel.target in QUANTITIES and all(s in QUANTITIES for s in rel.sources), family
            if rel.formula is None:
                assert rel.kind in REDUCTION_KINDS, (family, rel.target)
            else:
                assert rel.kind is None, (family, rel.target)  # the catalog supplies it
                spec = catalog.describe(rel.formula)
                assert spec.reduction is not None, rel.formula
                # the graph's quantities agree with the formula's declared output representation
                assert QUANTITIES[rel.target].representation == spec.reduction.output, (rel.formula, rel.target)


def test_quantity_dimensionless_flags_agree_with_the_formulas_that_produce_them():
    for relations in REDUCTION_FAMILIES.values():
        for rel in relations:
            if rel.formula is None:
                continue
            unit = catalog.describe(rel.formula).returns[0].unit
            assert QUANTITIES[rel.target].dimensionless == (unit == "-"), (rel.formula, unit)


def test_empirical_scaling_is_exactly_an_empirical_formula():
    for spec in catalog.list_formulas():
        if spec.reduction is not None:
            assert (spec.reduction.kind == "empirical_scaling") == spec.empirical, spec.qualname
    text = GOOD.replace("kind: feature_extraction", "kind: empirical_scaling")
    reduction, errors = catalog._reduction(parse_docstring(_docstring(text, "-")))
    assert reduction is None and any("empirical" in e for e in errors)


def test_show_renders_the_reduction_as_a_list():
    card = catalog.describe("virial.virial_li_from_volume").to_markdown()
    assert "- kind: quadratic_integral" in card and "input: field_2d" not in card
