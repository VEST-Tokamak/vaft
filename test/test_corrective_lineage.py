"""The corrective pipeline's stage list and scope: one definition for the Snakefile and the worker."""

from __future__ import annotations

import pytest

from vaft.database.corrective_lineage import CORRECTIVE_LINEAGE, DEFAULT_STAGES, UPSTREAM, scoped, stage_scope
from vaft.database.sources import STAGE_REPLICATION, replicable_stages


def test_every_stage_is_a_registered_corrective_product():
    corrective = set(replicable_stages(produced_by="corrective"))
    assert set(CORRECTIVE_LINEAGE) <= corrective
    assert all(STAGE_REPLICATION[stage].optional for stage in CORRECTIVE_LINEAGE)


def test_the_lineage_is_in_a_dependency_respecting_order():
    position = {stage: index for index, stage in enumerate(CORRECTIVE_LINEAGE)}
    assert set(UPSTREAM) == set(CORRECTIVE_LINEAGE)
    for stage, needs in UPSTREAM.items():
        assert all(position[needed] < position[stage] for needed in needs), stage


def test_the_kinetic_lineages_never_read_the_inferred_temperature():
    """Their pressure would be partitioned back into the T_i it was built from."""
    for stage in ("electron_efit", "kinetic_efit"):
        assert UPSTREAM[stage] == ("composition",)
    readers = {stage for stage, needs in UPSTREAM.items() if "ti_inferred" in needs}
    assert readers == {"resistive_zeff"}, "only the resistive estimate is built on the inferred-T_i product"
    assert UPSTREAM["resistive_zeff"] == ("ti_inferred",)


def test_no_scope_is_the_whole_lineage():
    assert stage_scope(None) == frozenset(DEFAULT_STAGES)
    assert scoped(None) == CORRECTIVE_LINEAGE


def test_a_scope_keeps_lineage_order_and_only_what_it_names():
    scope = stage_scope(["ces", "thomson"])
    assert scope == {"thomson", "ces"}
    assert scoped(scope) == ("thomson", "ces")
    assert scoped(stage_scope("thomson")) == ("thomson",)


def test_the_whole_chain_down_to_the_resistive_estimate_is_a_valid_scope():
    chain = ["thomson", "ces", "core_profiles", "composition", "ti_inferred", "resistive_zeff"]
    assert scoped(stage_scope(chain)) == tuple(chain)


@pytest.mark.parametrize(
    "stages, message",
    [
        (["composition"], "'composition' but not 'core_profiles'"),
        (["thomson", "core_profiles"], "'core_profiles' but not 'ces'"),
        (["thomson", "ces", "core_profiles", "composition", "resistive_zeff"], "'resistive_zeff' but not 'ti_inferred'"),
        (["thomson", "bogus"], "Unknown stage"),
        ([], "empty"),
    ],
)
def test_a_stage_without_its_upstream_is_refused_not_pulled_in(stages, message):
    with pytest.raises(ValueError, match=message):
        stage_scope(stages)
