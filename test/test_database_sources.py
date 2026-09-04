"""The named HSDS source registry (issue #56).

Source identity is the one thing that decides which analysis lineage a shot is
written into, so the catalog, the default, the alias rules and the read-only
guarantee for the legacy namespace are all pinned here.
"""

import pytest

from vaft.database import sources
from vaft.database.filedb import OMASStage
from vaft.database.sources import (
    DEFAULT_SOURCE,
    LEGACY_SOURCE,
    MissingSourceError,
    ReadOnlySourceError,
    UnknownSourceError,
)


def test_catalog_carries_every_lineage_the_issue_names():
    assert {source.name for source in sources.known_sources()} == {
        "public",
        "main",
        "chease-mhd-stability",
        "vfit-element",
        "vfit-gse",
        "electron-efit",
        "kinetic-efit",
        "impa",
    }


def test_public_is_the_only_read_only_source():
    unwritable = [s.name for s in sources.known_sources() if not s.writable]
    assert unwritable == [LEGACY_SOURCE]


def test_unnamed_source_resolves_to_main():
    assert sources.resolve() == DEFAULT_SOURCE == "main"


def test_legacy_source_still_resolves_for_reads():
    assert sources.resolve(LEGACY_SOURCE) == "public"


def test_writing_to_the_legacy_source_is_refused():
    with pytest.raises(ReadOnlySourceError, match="read-only legacy reference"):
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
    with pytest.raises(sources.HSDSSourceError, match="bare HSDS namespace"):
        sources.resolve("main")


@pytest.mark.parametrize(
    "value",
    [
        "hdf5://public",          # protocol is an internal detail
        "/tmp/data",              # filesystem path
        "public/39915",           # nested path
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
    with pytest.raises(sources.HSDSSourceError, match="bare HSDS namespace"):
        sources.resolve(value)


def test_none_means_unspecified_rather_than_invalid():
    assert sources.resolve(None) == DEFAULT_SOURCE


def test_hyphenated_catalog_names_are_accepted():
    assert sources.resolve("chease-mhd-stability") == "chease-mhd-stability"


def test_every_filedb_omas_stage_has_an_explicit_replication_contract():
    """A new stage must fail loudly rather than silently replicate nowhere."""
    for stage in OMASStage:
        entry = sources.replication_for_stage(stage)
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
    """The eddy product carries the diagnostics IDS through but does not own them.

    `build_eddy_ods` starts from the finalized diagnostics ODS, so replicating
    the whole product would have eddy overwrite what diagnostics wrote.
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


def test_the_two_mhd_linear_owners_are_kept_apart_by_occurrence():
    stability = sources.replication_for_stage("mhd_linear")
    ideal = sources.replication_for_stage("gpec_ideal")

    assert "mhd_linear" in stability.ids and "mhd_linear" in ideal.ids
    assert stability.source == ideal.source
    assert stability.occurrence != ideal.occurrence


def test_ideal_gpec_replication_is_still_deferred_to_its_own_issue():
    ideal = sources.replication_for_stage("gpec_ideal")

    assert ideal.deferred_to == "#95"
    assert ideal.replicable is False
    assert "gpec_ideal" not in sources.replicable_stages()


def test_no_stage_is_replicated_into_the_read_only_legacy_source():
    for stage in sources.replicable_stages():
        assert sources.replication_for_stage(stage).source != LEGACY_SOURCE
        # Every destination must survive a writability check.
        sources.resolve(sources.source_for_stage(stage), writable=True)


def test_stage_mapping_keeps_the_baseline_and_the_refinement_apart():
    assert sources.source_for_stage(OMASStage.EFIT) == "main"
    assert sources.source_for_stage("chease") == "chease-mhd-stability"
    assert sources.source_for_stage("mhd_linear") == "chease-mhd-stability"
    assert sources.source_for_stage("gpec_ideal") == "chease-mhd-stability"


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
