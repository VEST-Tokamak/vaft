"""Soft X-ray and camera publish on the same contract as every other stage.

Both diagnostics used to have no HSDS publication contract at all: neither was
an `OMASStage`, neither owned a `STAGE_REPLICATION` entry, and their products
sat outside the `FileDBDomain` grammar so `FileDB.resolve` could not address
them (#599).

They own IDS no other stage claims, so both publish into the baseline source.
The high-frame-rate camera set is the exception: it owns the *same* IDS as
routine camera, and two stages cannot own one IDS in one source without the
second replacing the first.
"""

from __future__ import annotations

import pytest

from vaft.database import sources
from vaft.database.filedb import OMAS_PRODUCT_SUFFIXES, FileDB, OMASStage

EXTERNAL_STAGES = ("soft_x_rays", "camera_visible", "camera_visible_fluctuation")


# --------------------------------------------------------------------------- #
# the contract exists at all
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stage", EXTERNAL_STAGES)
def test_each_diagnostic_is_a_real_stage(stage):
    assert stage in {member.value for member in OMASStage}
    entry = sources.replication_for_stage(stage)
    assert entry.source is not None
    assert entry.replicable


@pytest.mark.parametrize("stage", EXTERNAL_STAGES)
def test_each_product_resolves_through_the_canonical_grammar(stage):
    """A hand-built path is one replication cannot find."""
    db = FileDB("/filedb")
    product = db.omas_product(stage, shot=41451)
    assert product.parent == db.omas(stage, shot=41451, artifact="output")
    assert product.name == f"{stage}{OMAS_PRODUCT_SUFFIXES[stage]}"


@pytest.mark.parametrize("stage", EXTERNAL_STAGES)
def test_a_sparse_diagnostic_cannot_fail_a_pipeline(stage):
    """Requirement: a missing or failed optional diagnostic invalidates nothing.

    Most shots have neither acquisition. An ineligible product must be
    recorded as skipped rather than raised, or a shot without a camera would
    fail a run whose required stages all succeeded.
    """
    assert sources.replication_for_stage(stage).optional


@pytest.mark.parametrize("stage", EXTERNAL_STAGES)
def test_these_are_corrective_products_not_routine_ones(stage):
    """Pipeline 1 asserts it has a rule for everything it replicates.

    Without this, adding these stages demanded Snakemake rules in a pipeline
    that does not build them.
    """
    assert sources.replication_for_stage(stage).produced_by == "corrective"
    assert stage not in sources.replicable_stages(produced_by="routine")


# --------------------------------------------------------------------------- #
# one diagnostic's publication cannot disturb another's
# --------------------------------------------------------------------------- #


def test_soft_x_ray_and_camera_own_disjoint_ids():
    """Requirement: publishing one does not alter the other's stored content.

    They are separated at the only level that matters for a write: a stage
    write is projected to the IDS it owns, so a stage that does not own
    `camera_visible` cannot write it.
    """
    sxr = set(sources.replication_for_stage("soft_x_rays").ids)
    camera = set(sources.replication_for_stage("camera_visible").ids)
    assert sxr == {"soft_x_rays"}
    assert camera == {"camera_visible"}
    assert sxr.isdisjoint(camera)


def test_neither_collides_with_an_existing_stage_in_the_baseline_source():
    """Which is why they need no source of their own, unlike IMPA.

    IMPA re-owns `magnetics`, so it was separated to keep an optional
    diagnostic out of the baseline product it would be appended to. These two
    claim IDS nobody else does.
    """
    incumbent = {
        ids
        for stage, entry in sources.STAGE_REPLICATION.items()
        if entry.source == sources.DEFAULT_SOURCE
        and stage not in ("soft_x_rays", "camera_visible")
        for ids in entry.ids
    }
    assert "soft_x_rays" not in incumbent
    assert "camera_visible" not in incumbent


# --------------------------------------------------------------------------- #
# routine and fluctuation camera cannot overwrite one another
# --------------------------------------------------------------------------- #


def test_the_two_camera_lineages_share_an_ids_and_so_must_not_share_a_source():
    routine = sources.replication_for_stage("camera_visible")
    fluctuation = sources.replication_for_stage("camera_visible_fluctuation")

    assert routine.ids == fluctuation.ids == ("camera_visible",)
    assert routine.source != fluctuation.source, (
        "two stages owning one IDS in one source means the second write "
        "replaces the first"
    )


def test_the_split_is_by_source_not_by_occurrence():
    """Lazy HSDS access reads occurrence 0 only.

    Separating the lineages by occurrence would leave the fluctuation product
    unreadable through the path #161's analysis would use.
    """
    for stage in ("camera_visible", "camera_visible_fluctuation"):
        assert sources.replication_for_stage(stage).occurrence == 0


def test_the_fluctuation_source_is_a_known_sparse_lineage():
    entry = sources.replication_for_stage("camera_visible_fluctuation")
    catalog = {source.name: source for source in sources.known_sources()}
    assert entry.source in catalog
    assert catalog[entry.source].sparse
    assert sources.resolve(entry.source, writable=True) == entry.source


def test_their_products_never_share_a_path():
    db = FileDB("/filedb")
    shot = 27134
    assert db.omas_product("camera_visible", shot=shot) != db.omas_product(
        "camera_visible_fluctuation", shot=shot
    )


def test_loading_the_baseline_does_not_reach_the_fluctuation_lineage():
    """`load(source="main")` must return the routine product, never a union."""
    routine = sources.replication_for_stage("camera_visible")
    fluctuation = sources.replication_for_stage("camera_visible_fluctuation")
    assert routine.source == sources.DEFAULT_SOURCE
    assert fluctuation.source != sources.DEFAULT_SOURCE
