"""The named HSDS source registry (issue #56).

Source identity is the one thing that decides which analysis lineage a shot is
written into, so the catalog, the default, the alias rules and the read-only
guarantee for the legacy namespace are all pinned here.
"""

import inspect

import pytest

from vaft.database import sources
from vaft.database.ods import _recorded_source as sources_recorded
from vaft.database.filedb import OMASStage
from vaft.database.sources import (
    DEFAULT_SOURCE,
    LEGACY_SOURCE,
    MissingSourceError,
    ReadOnlySourceError,
    UnknownSourceError,
)


def test_catalog_carries_every_lineage_the_issue_names():
    roots = {
        source.name
        for source in sources.known_sources()
        if "/" not in source.name and not source.projects_to
    }
    assert roots == {
        "public",
        "main",
        "chease-mhd-stability",
        "vfit-element",
        "vfit-gse",
        "electron-efit",
        "kinetic-efit",
        "impa",
        # Same IDS as the routine camera product in `main`, so it needs its own
        # lineage rather than an occurrence: lazy HSDS access reads occurrence
        # 0 only (#599).
        "camera-visible-fluctuation",
    }


def test_every_family_gets_a_refinement_and_one_source_per_stability_product():
    """The hierarchy is generated, so it is checked as a shape, not a list.

    Twelve hand-written entries would drift the moment a product is added;
    asserting the cross-product is what keeps the catalog and the design the
    same statement.
    """
    names = {source.name for source in sources.known_sources()}

    for family, root in sources.FAMILY_SOURCES.items():
        refinement = f"{root}/chease"
        assert refinement in names, family
        for product in ("dcon-peeling", "dcon-kink", "rdcon", "stride"):
            assert f"{refinement}/{product}" in names, (family, product)

    # DCON's two edge treatments are separate sources because a run yields one
    # of them and can never yield both.
    assert "main/chease/dcon-peeling" != "main/chease/dcon-kink"
    assert sources.resolve("main/chease/dcon-kink", writable=True) == (
        "main/chease/dcon-kink"
    )


def test_a_nested_source_names_its_parent_and_every_ancestor():
    """HSDS has no mkdir -p, so provisioning needs them outermost first."""
    entry = sources.CATALOG["main/chease/dcon-peeling"]

    assert entry.parent == "main/chease"
    assert entry.ancestors == ("main", "main/chease")
    assert sources.CATALOG["main"].parent is None
    assert sources.CATALOG["main"].ancestors == ()


def test_the_missing_source_hint_lists_one_command_per_level():
    """An operator should not discover the ordering one failure at a time."""
    message = str(sources.MissingSourceError("main/chease/dcon-peeling"))

    assert "in order" in message
    assert message.index("/main/") < message.index("/main/chease/")
    assert message.index("/main/chease/") < message.index("/main/chease/dcon-peeling/")

    # A flat source still gets the single-command form.
    assert "in order" not in str(sources.MissingSourceError("main"))


def test_the_projecting_alias_reads_through_and_refuses_writes():
    """`magnetic-efit` is a name for what `main` holds, not a second copy."""
    assert sources.resolve("magnetic-efit") == "main"

    with pytest.raises(sources.ReadOnlySourceError, match="read-only projection"):
        sources.resolve("magnetic-efit", writable=True)


def test_a_name_deeper_than_the_grammar_is_refused():
    """family / refinement / product is three; nothing needs a fourth.

    The bound is what stops a typo'd loop from creating an unbounded folder
    tree on the deployment.
    """
    with pytest.raises(sources.HSDSSourceError, match="segments deep"):
        sources.resolve("main/chease/dcon-peeling/extra")


def test_a_shot_inside_a_source_name_is_still_refused():
    """`public/39915` used to fail the grammar; nesting made it well-formed.

    It must still be refused -- the problem with it was never its shape but
    that it names a shot inside a source rather than a source. It now fails
    lookup, which is the more accurate error.
    """
    with pytest.raises(sources.UnknownSourceError):
        sources.resolve("public/39915")


def test_the_read_only_sources_are_each_read_only_for_their_own_reason():
    """Read-only for three different reasons, each deliberate.

    `public` is a legacy reference nothing may rewrite. `magnetic-efit` is a
    *name* for what `main` already holds, so making it writable would give one
    dataset two writable destinations -- the collision named sources exist to
    prevent. `chease-mhd-stability` is superseded: it stays readable so an
    existing deployment keeps resolving, and is deleted by the gated migration
    in #94 rather than being redirected onto the hierarchy, which would be
    implicit composition of several products behind one name.
    """
    unwritable = sorted(s.name for s in sources.known_sources() if not s.writable)
    assert unwritable == ["chease-mhd-stability", "magnetic-efit", LEGACY_SOURCE]

    projecting = [s for s in sources.known_sources() if s.projects_to]
    assert [(s.name, s.projects_to) for s in projecting] == [("magnetic-efit", "main")]


def test_unnamed_source_resolves_to_main():
    assert sources.resolve() == DEFAULT_SOURCE == "main"


def test_legacy_source_still_resolves_for_reads():
    assert sources.resolve(LEGACY_SOURCE) == "public"


def test_writing_to_the_legacy_source_is_refused():
    with pytest.raises(ReadOnlySourceError, match="is read-only"):
        sources.resolve(LEGACY_SOURCE, writable=True)


def test_deprecated_aliases_resolve_and_warn():
    for alias in ("directory", "target"):
        with pytest.warns(DeprecationWarning, match="deprecated alias"):
            assert sources.resolve(**{alias: "vfit-gse"}) == "vfit-gse"


def test_two_names_for_one_source_is_a_type_error():
    with pytest.raises(TypeError, match="only one of source"):
        sources.resolve("main", directory="public")


def test_unknown_source_is_rejected_and_lists_what_is_available():
    with pytest.raises(UnknownSourceError, match="Unknown HSDS source 'maim'"):
        sources.resolve("maim")


def test_experiment_namespaces_are_opt_in_through_the_environment(monkeypatch):
    monkeypatch.setenv(sources.EXTRA_SOURCES_VARIABLE, "scratch-42, private")
    assert sources.resolve("scratch-42") == "scratch-42"
    assert sources.resolve("private", writable=True) == "private"
    assert "scratch-42" in {s.name for s in sources.known_sources()}


def test_opted_in_namespaces_obey_the_same_grammar(monkeypatch):
    monkeypatch.setenv(sources.EXTRA_SOURCES_VARIABLE, "Not A Namespace")
    with pytest.raises(sources.HSDSSourceError, match="not a valid HSDS namespace"):
        sources.resolve("main")


@pytest.mark.parametrize(
    "value",
    [
        "hdf5://public",          # protocol is an internal detail
        "/tmp/data",              # filesystem path
        "Main",                   # uppercase
        "chease.mhd.stability",   # ambiguous with the legacy dotted-domain form
        "39915_test",             # underscore
        "-main",                  # leading hyphen
        "main-",                  # trailing hyphen
        "",
        3,
    ],
)
def test_grammar_rejects_anything_that_is_not_a_bare_namespace(value):
    with pytest.raises(sources.HSDSSourceError, match="must be an HSDS namespace"):
        sources.resolve(value)


def test_none_means_unspecified_rather_than_invalid():
    assert sources.resolve(None) == DEFAULT_SOURCE


def test_hyphenated_catalog_names_are_accepted():
    assert sources.resolve("chease-mhd-stability") == "chease-mhd-stability"


#: What a stage that is one solve's result needs before it has a destination.
#: Supplied by the test rather than defaulted in the code, because guessing it
#: there would send one solver's result to another's source.
_PRODUCT_LINEAGE = {"product": "dcon-peeling"}


def _contract(stage):
    """The replication contract for `stage`, with the lineage it requires."""
    key = stage.value if hasattr(stage, "value") else str(stage)
    needs_product = key in {"mhd_linear", "gpec_ideal"}
    return sources.replication_for_stage(
        stage, **(_PRODUCT_LINEAGE if needs_product else {})
    )


def test_every_filedb_omas_stage_has_an_explicit_replication_contract():
    """A new stage must fail loudly rather than silently replicate nowhere."""
    for stage in OMASStage:
        entry = _contract(stage)
        if entry.source is None:
            # Opting out is allowed, but only with a stated reason.
            assert entry.note, stage
            assert entry.ids == ()
        else:
            assert entry.source in {e.name for e in sources.known_sources()}
            assert entry.ids, stage


def test_static_is_not_shot_replicated_and_says_why():
    entry = sources.replication_for_stage("static")

    assert entry.source is None
    assert entry.replicable is False
    with pytest.raises(sources.HSDSSourceError, match="not replicated to HSDS"):
        sources.source_for_stage("static")


def test_eddy_owns_only_what_it_computes():
    """eddy solves against the diagnostics IDS but owns and stores none of them.

    Its product is exactly this projection, so `replication._project` is an
    identity on it -- what is on disk and what is published cannot disagree
    about which stage is authoritative for an IDS.
    """
    eddy = sources.replication_for_stage("eddy")
    diagnostics = sources.replication_for_stage("diagnostics")

    assert eddy.ids == ("pf_passive",)
    assert set(eddy.ids).isdisjoint(diagnostics.ids)


def test_the_two_equilibrium_owners_are_kept_apart_by_source():
    efit = sources.replication_for_stage("efit")
    chease = sources.replication_for_stage("chease")

    assert efit.ids == chease.ids == ("equilibrium",)
    assert efit.source != chease.source


def test_the_two_mhd_linear_owners_no_longer_need_the_occurrence_to_stay_apart():
    """Per-product sources separate them; the occurrence is now redundant.

    Both write the `mhd_linear` IDS, and they used to share one namespace, so an
    occurrence was the only thing keeping them from overwriting each other. Now
    they resolve to different sources, and would even if the occurrences were
    equal.

    The occurrence is left in place regardless. Removing it is a change to how a
    consumer reads ideal-GPEC's product, and that read contract is #95's open
    question -- this notes that the mechanism became redundant without deciding
    what should replace it.
    """
    stability = _contract("mhd_linear")
    ideal = _contract("gpec_ideal")

    assert "mhd_linear" in stability.ids and "mhd_linear" in ideal.ids
    assert stability.source != ideal.source
    assert stability.occurrence != ideal.occurrence
    assert ideal.deferred_to == "#95"


def test_ideal_gpec_replication_is_still_deferred_to_its_own_issue():
    ideal = sources.replication_for_stage("gpec_ideal")

    assert ideal.deferred_to == "#95"
    assert ideal.replicable is False
    assert "gpec_ideal" not in sources.replicable_stages()


def test_no_replicable_stage_writes_into_a_read_only_source():
    """Three sources are read-only now, and none may be a destination.

    `chease-mhd-stability` joining them is the point of the per-product split:
    a stage still pointing at it would fail here rather than on a deployment.
    """
    for stage in sources.replicable_stages():
        entry = _contract(stage)
        assert entry.source not in {
            s.name for s in sources.known_sources() if not s.writable
        }, stage
        # Every destination must survive a writability check.
        sources.resolve(entry.source, writable=True)


def test_stage_mapping_keeps_the_baseline_and_the_refinement_apart():
    """Each product hangs beneath what it derives from, and none share a source.

    The refinement no longer sits in a namespace of its own beside the
    baseline -- it hangs under it, which is what makes the lineage readable from
    the path. Separate sources still: nothing is unioned on read.
    """
    assert sources.source_for_stage(OMASStage.EFIT) == "main"
    assert sources.source_for_stage("chease") == "main/chease"
    assert (
        sources.source_for_stage("mhd_linear", product="dcon-peeling")
        == "main/chease/dcon-peeling"
    )
    # The two DCON edge treatments are the same module run two ways, so a shared
    # destination would have them overwrite each other.
    assert (
        sources.source_for_stage("mhd_linear", product="dcon-kink")
        == "main/chease/dcon-kink"
    )
    assert len({
        sources.source_for_stage("mhd_linear", product=name)
        for name in ("dcon-peeling", "dcon-kink", "rdcon", "stride")
    }) == 4


def test_a_stability_stage_will_not_guess_which_product_it_is():
    """Five products, and sending one solver's result to another's source is
    exactly the collision the per-product sources exist to prevent."""
    with pytest.raises(sources.HSDSSourceError, match="needs product="):
        sources.source_for_stage("mhd_linear")


def test_another_family_publishes_beneath_its_own_root():
    """The destination is derived from the lineage that named the local product.

    An electron-EFIT campaign gets its own sources by naming its family, not by
    a second table that could disagree with the FileDB grammar.
    """
    assert (
        sources.source_for_stage("mhd_linear", family="electron", product="rdcon")
        == "electron-efit/chease/rdcon"
    )
    assert sources.source_for_stage("chease", family="kinetic") == "kinetic-efit/chease"


def test_unknown_stage_is_reported_with_the_valid_choices():
    with pytest.raises(sources.HSDSSourceError, match="Invalid OMAS stage"):
        sources.source_for_stage("not_a_stage")


def test_missing_source_error_names_the_administrator_fix():
    error = MissingSourceError("main")
    assert "hstouch" in str(error)
    assert "/main/" in str(error)


def test_describe_and_is_writable_agree_with_the_catalog():
    assert sources.describe("main").writable is True
    assert sources.is_writable(LEGACY_SOURCE) is False
    assert "CHEASE" in sources.describe("chease-mhd-stability").purpose


def test_processed_registry_uri_gates_writes_to_the_legacy_source():
    """The corrective registry is opened with raw h5pyd, so it needs its own gate.

    `database.save` refuses `public`, but `processed_shots.h5` never goes
    through it -- an append-mode h5pyd open would sail straight past that
    guarantee and mutate the legacy namespace.
    """
    from vaft.database.utils import processed_registry_uri

    assert processed_registry_uri() == "hdf5://main/processed_shots.h5"
    # Reading the legacy registry is how a fresh source bootstraps its backlog.
    assert (
        processed_registry_uri(LEGACY_SOURCE)
        == "hdf5://public/processed_shots.h5"
    )
    with pytest.raises(ReadOnlySourceError):
        processed_registry_uri(LEGACY_SOURCE, writable=True)


def test_corrective_updater_refuses_a_read_only_source(monkeypatch):
    """The updaters must not be pointable at `public` through the environment."""
    import importlib
    import sys
    from pathlib import Path

    workflow = (
        Path(__file__).parents[1]
        / "workflow"
        / "automatic_pipeline_2_corrective_data_update"
    )
    if not workflow.exists():
        pytest.skip("workflow scripts are not part of the distribution")

    monkeypatch.setenv("VAFT_HSDS_SOURCE", LEGACY_SOURCE)
    monkeypatch.syspath_prepend(str(workflow))
    monkeypatch.delitem(
        sys.modules, "update_thomson_scattering_and_core_profile", raising=False
    )

    with pytest.raises(ReadOnlySourceError):
        importlib.import_module("update_thomson_scattering_and_core_profile")


def test_source_probe_never_asks_the_server_to_create_a_namespace():
    """h5pyd.Folder() defaults to a PUT; a probe must be an explicit read."""
    import inspect

    from vaft.database import utils

    body = inspect.getsource(utils.require_source_exists)
    assert 'mode="r"' in body


def test_source_probe_opens_the_namespace_without_enumerating_its_children(monkeypatch):
    """A publication probe must stay constant-cost as the source grows."""
    from vaft.database import utils

    calls = []

    class Folder:
        def __iter__(self):
            raise AssertionError("the source probe must not enumerate child domains")

    def open_folder(path, *, mode):
        calls.append((path, mode))
        return Folder()

    monkeypatch.setattr(utils.h5pyd, "Folder", open_folder)

    utils.require_source_exists("main")

    assert calls == [("/main/", "r")]


@pytest.mark.parametrize("status", [404, 410])
def test_source_probe_turns_only_missing_namespace_statuses_into_source_errors(
    monkeypatch, status
):
    from vaft.database import utils

    def open_folder(path, *, mode):
        raise OSError(status, "domain not found")

    monkeypatch.setattr(utils.h5pyd, "Folder", open_folder)

    with pytest.raises(MissingSourceError):
        utils.require_source_exists("main")


def test_source_probe_preserves_transient_hsds_errors(monkeypatch):
    from vaft.database import utils

    transient = OSError(503, "server busy")

    def open_folder(path, *, mode):
        raise transient

    monkeypatch.setattr(utils.h5pyd, "Folder", open_folder)

    with pytest.raises(OSError) as caught:
        utils.require_source_exists("main")

    assert caught.value is transient


def test_a_child_source_is_navigable_from_its_parent():
    """A hierarchy nothing can walk is two conventions, not one.

    Read from the catalog rather than from a deployment: the catalog says which
    sources exist, a folder on a server says which have been *provisioned*, and
    conflating the two would make "does this source exist" depend on whether an
    administrator had got to it yet.
    """
    assert [entry.name for entry in sources.children("main")] == ["main/chease"]
    assert [entry.name for entry in sources.children("main/chease")] == [
        "main/chease/dcon-peeling",
        "main/chease/dcon-kink",
        "main/chease/rdcon",
        "main/chease/stride",
    ]
    assert sources.children("impa") == ()

    with pytest.raises(sources.UnknownSourceError):
        sources.children("not-a-source")


def test_knowing_the_children_does_not_compose_them():
    """The no-auto-union rule is what this hierarchy must not quietly relax.

    `load(source="main")` must not acquire the CHEASE refinement because the
    catalog now says where it lives. Composition stays an explicit call.
    """
    assert sources.resolve("main") == "main"
    assert sources.children("main"), "precondition: main has a child"
    # Resolving a parent yields the parent, never a union or a list.
    assert isinstance(sources.resolve("main"), str)


def test_a_child_folder_cannot_be_mistaken_for_a_shot():
    """`/main/chease/` sits beside `/main/39915/` on the deployment.

    Shot listing filters on `isdigit()`, so a child source nested under a parent
    cannot make the parent's shot index wrong -- which is the property that lets
    the logical hierarchy be the physical one.
    """
    from vaft.database import utils

    listed = [
        item
        for item in ["39915", "41524", "chease", "master.h5", "dcon-peeling"]
        if not item.endswith(".h5") and item.isdigit()
    ]
    assert listed == ["39915", "41524"]
    # And the production filter is the one just modelled.
    source = inspect.getsource(utils._get_namespace_folders)
    assert 'item.isdigit()' in source


def test_a_slash_bearing_source_round_trips_through_the_recorded_namespace():
    """`dataset_description.data_entry.user` records where a shot was written.

    A nested name has to survive that round trip, or a product would carry a
    destination it was not written to.
    """
    from omas import ODS

    ods = ODS(consistency_check=False)
    for name in ("main", "main/chease/dcon-peeling"):
        with sources_recorded(ods, name):
            assert ods["dataset_description.data_entry.user"] == name
    assert "dataset_description.data_entry.user" not in ods


def test_a_read_only_refusal_says_why_this_source_is_read_only():
    """Three sources, three reasons; one blanket message sends an operator astray.

    `public` is a legacy reference nothing may rewrite. `chease-mhd-stability`
    is superseded and awaiting migration -- an operator told it is a "legacy
    reference" would look for the wrong thing.
    """
    with pytest.raises(sources.ReadOnlySourceError, match="Legacy source"):
        sources.resolve("public", writable=True)

    with pytest.raises(sources.ReadOnlySourceError, match="Superseded"):
        sources.resolve("chease-mhd-stability", writable=True)

    # The projecting alias has its own refusal, which names the target.
    with pytest.raises(sources.ReadOnlySourceError, match="read-only projection"):
        sources.resolve("magnetic-efit", writable=True)
