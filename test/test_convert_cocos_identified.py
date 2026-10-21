"""convert_cocos carries the identified candidates through the transform (#601).

The identified list used to be shifted by ten across storage families. That
ignored the sign changes of a same-family conversion and turned a candidate in
the other family into an index that does not exist (11 -> 1 with candidate 3
gave -7).
"""
from __future__ import annotations

from dataclasses import replace

import pytest
from omas import cocos_transform

from vaft.data.cocos import COCOS_INDICES
from vaft.process._equilibrium_parametric import _carry_identified


@pytest.fixture(scope="module")
def sample():
    from vaft.data.resources import sample_geqdsk
    from vaft.process.equilibrium import as_equilibrium

    return as_equilibrium(sample_geqdsk(), convention=1)


@pytest.mark.parametrize("source", COCOS_INDICES)
@pytest.mark.parametrize("target", COCOS_INDICES)
def test_every_carried_candidate_takes_the_same_transform(source, target):
    carried, dropped = _carry_identified(COCOS_INDICES, source, target)
    assert len(carried) + len(dropped) == len(COCOS_INDICES)
    # Only a conversion that crosses families moves the 2*pi, and then only the
    # candidates in the other family lose their image.
    crosses = (source > 10) != (target > 10)
    assert set(dropped) == ({c for c in COCOS_INDICES if (c > 10) != (source > 10)} if crosses else set())
    assert set(carried) <= set(COCOS_INDICES)
    expected = cocos_transform(source, target)
    survivors = [c for c in COCOS_INDICES if c not in dropped]
    for before, after in zip(survivors, carried):
        factors = cocos_transform(before, after)
        assert all(factors[k] == pytest.approx(expected[k]) for k in expected)


def test_the_source_maps_to_the_target():
    for source in COCOS_INDICES:
        for target in COCOS_INDICES:
            assert _carry_identified((source,), source, target) == ((target,), ())


def test_a_cross_family_candidate_is_dropped_not_shifted_out_of_range(sample):
    from vaft.process.equilibrium import convert_cocos

    declared_11 = replace(sample, convention=replace(
        convert_cocos(sample, 11).convention, identified=(3, 11)))
    converted = convert_cocos(replace(sample, convention=declared_11.convention), 1)

    assert all(index in COCOS_INDICES for index in converted.convention.identified)
    assert converted.convention.identified == (1,)
    assert "(3,)" in converted.convention.source and "dropped" in converted.convention.source


@pytest.mark.parametrize("target", [2, 3, 5, 7, 11, 13, 16, 18])
def test_the_carried_set_is_what_identification_finds_on_the_converted_data(sample, target):
    from vaft.process.cocos import identify_convention
    from vaft.process.equilibrium import convert_cocos

    assert sample.convention.identified == (1, 2)
    converted = convert_cocos(sample, target)
    assert set(converted.convention.identified) == set(identify_convention(converted))
    assert target in converted.convention.identified


def test_a_contradicted_declaration_stays_contradicted(sample):
    from vaft.process.equilibrium import convert_cocos

    # Declared 3, but the data supports (1, 2): converting must not launder it.
    stale = replace(sample, convention=replace(sample.convention, cocos=3, candidates=(3,)))
    assert stale.convention.contradicted
    converted = convert_cocos(stale, 13)
    assert converted.convention.contradicted
    assert 13 not in converted.convention.identified


def test_a_contradiction_survives_when_no_candidate_can_be_carried(sample):
    # Declared 1 but the flux is 2*pi-scaled: the data identifies (11, 12). A
    # conversion to 11 drops both, and must not leave a clean-looking record.
    from vaft.process.equilibrium import check_equilibrium_requirements, convert_cocos

    mislabelled = replace(sample, convention=replace(sample.convention, identified=(11, 12)))
    assert mislabelled.convention.contradicted
    converted = convert_cocos(mislabelled, 11)

    assert converted.convention.identified == ()
    assert converted.convention.uncarried == (11, 12)
    assert converted.convention.contradicted
    codes = {issue.code for issue in check_equilibrium_requirements(converted).issues}
    assert "cocos_declared_conflicts_with_signs" in codes


def test_a_partial_drop_does_not_mark_a_supported_declaration_contradicted(sample):
    # Flux exponent abstained: both families survive identification. Declared 1
    # is supported, and the conversion to 11 keeps that.
    from vaft.process.equilibrium import convert_cocos

    open_ended = replace(sample, convention=replace(sample.convention, identified=(1, 2, 11, 12)))
    converted = convert_cocos(open_ended, 11)
    assert converted.convention.identified == (11, 12)
    assert converted.convention.uncarried == ()
    assert not converted.convention.contradicted
