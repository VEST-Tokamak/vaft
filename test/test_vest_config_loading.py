"""Loading and merging the VEST configuration safely (#2159, audit gaps G1-G5).

Synthetic YAML in ``tmp_path`` pins each rule; the last group proves the
packaged vest.yaml satisfies the stricter revision rules at every era bound.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from vaft.machine_mapping import utils
from vaft.machine_mapping.utils import (
    ERA_ONLY_KEYS,
    VestConfigurationError,
    _deep_merge,
    _policy_document,
    _resolve_info_file_path,
    load_yaml,
    package_data_path,
    resolve_diagnostics_time_policies,
    resolve_shot_revisions,
    resolve_shot_revisions_with_provenance,
    resolve_vest_diagnostic,
)


def _write(path: Path, document) -> str:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return str(path)


def _diagnostic_document(**block) -> dict:
    return {0: {"diagnostics": {"probe": block}}}


# --- G1: an explicit path never falls back -------------------------------------------


def test_omitted_info_file_selects_the_packaged_vest_yaml():
    assert _resolve_info_file_path(None) == package_data_path("vest.yaml")


def test_a_bare_packaged_name_still_resolves_to_the_packaged_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _resolve_info_file_path("vest.yaml") == package_data_path("vest.yaml")


def test_an_existing_explicit_path_is_used(tmp_path, monkeypatch):
    path = _write(tmp_path / "campaign.yaml", _diagnostic_document(gain=2.0))
    assert _resolve_info_file_path(path) == path
    assert _resolve_info_file_path(Path(path)) == path
    monkeypatch.chdir(tmp_path)
    assert _resolve_info_file_path("campaign.yaml") == path
    assert resolve_vest_diagnostic(1, "probe", info_file="campaign.yaml") == {"gain": 2.0}


@pytest.mark.parametrize("name", ["missing.yaml", "nested/missing.yaml"])
def test_an_explicit_missing_path_raises_instead_of_falling_back(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    for candidate in (name, str(tmp_path / name)):
        with pytest.raises(VestConfigurationError, match="does not exist"):
            _resolve_info_file_path(candidate)


def test_the_resolvers_refuse_a_missing_explicit_file(tmp_path):
    missing = str(tmp_path / "vest_typo.yaml")
    with pytest.raises(VestConfigurationError, match="vest_typo.yaml"):
        resolve_vest_diagnostic(39915, "plasma_current", info_file=missing)
    with pytest.raises(VestConfigurationError, match="vest_typo.yaml"):
        resolve_diagnostics_time_policies(info_file=missing)


def test_a_directory_is_not_a_configuration_file(tmp_path):
    with pytest.raises(VestConfigurationError, match="not a file"):
        _resolve_info_file_path(str(tmp_path))


# --- malformed files fail where they are read ------------------------------------------


def test_malformed_yaml_names_the_file(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("0:\n  diagnostics: {probe: [1, 2\n", encoding="utf-8")
    with pytest.raises(VestConfigurationError, match="broken.yaml.*not valid YAML"):
        load_yaml(str(path))
    with pytest.raises(VestConfigurationError, match="not valid YAML"):
        resolve_vest_diagnostic(1, "probe", info_file=str(path))


def test_a_top_level_list_is_refused_and_an_empty_file_is_empty(tmp_path):
    listed = tmp_path / "list.yaml"
    listed.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(VestConfigurationError, match="must be a mapping at the top level"):
        load_yaml(str(listed))
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_yaml(str(empty)) == {}


# --- G2: the merge rules ---------------------------------------------------------------


def test_deep_merge_shares_nothing_with_its_arguments():
    base = {"window": {"index": [1, 2], "inner": {"a": 1}}, "untouched": {"b": [3]}}
    override = {"window": {"extra": {"c": [4]}}}
    merged = _deep_merge(base, override)
    merged["untouched"]["b"].append(99)
    merged["window"]["index"].append(99)
    merged["window"]["inner"]["a"] = 99
    merged["window"]["extra"]["c"].append(99)
    assert base == {"window": {"index": [1, 2], "inner": {"a": 1}}, "untouched": {"b": [3]}}
    assert override == {"window": {"extra": {"c": [4]}}}


def test_deep_merge_rules_for_mappings_lists_and_nulls():
    base = {"keep": 1, "nested": {"a": 1, "b": 2}, "items": [1, 2, 3], "placeholder": None, "gone": 5}
    override = {"nested": {"b": 20, "c": 30}, "items": [9], "placeholder": {"x": 1}, "gone": None, "new": [1]}
    assert _deep_merge(base, override) == {
        "keep": 1,
        "nested": {"a": 1, "b": 20, "c": 30},  # recursive
        "items": [9],  # lists replace, never concatenate
        "placeholder": {"x": 1},  # a null base is a placeholder
        "gone": None,  # an explicit null replaces; it is not "omitted"
        "new": [1],
    }


@pytest.mark.parametrize(
    ("base", "override"),
    [
        ({"window": {"index": 1}}, {"window": 5}),
        ({"window": 5}, {"window": {"index": 1}}),
        ({"window": [1, 2]}, {"window": {"index": 1}}),
        ({"a": {"window": {"index": 1}}}, {"a": {"window": "off"}}),
    ],
)
def test_deep_merge_refuses_a_shape_change(base, override):
    with pytest.raises(VestConfigurationError, match="cannot merge"):
        _deep_merge(base, override, context="probe")


def test_shape_conflict_names_the_key_path():
    with pytest.raises(VestConfigurationError, match=r"probe\.a\.window"):
        _deep_merge({"a": {"window": {"index": 1}}}, {"a": {"window": 3}}, context="probe")


def test_time_policy_overrides_follow_the_merge_rules():
    with pytest.raises(VestConfigurationError, match="cannot merge"):
        resolve_diagnostics_time_policies(overrides={"windows": 3})


# --- G3/G4: cached documents are never shared with callers -----------------------------


def test_the_policy_document_cannot_be_mutated_through_a_caller():
    first = _policy_document(None)
    first["diagnostics_time_policies"]["default"] = "mutated"
    first[0]["diagnostics"].clear()
    second = _policy_document(None)
    assert second["diagnostics_time_policies"]["default"] != "mutated"
    assert second[0]["diagnostics"]


def test_a_resolved_diagnostic_cannot_reach_the_cache_or_the_next_shot():
    resolved = resolve_vest_diagnostic(39915, "plasma_current")
    reference = resolved["processing"]["reference"]
    reference["field"] = -1
    resolved["processing"]["baseline"]["revisions"].clear()
    again = resolve_vest_diagnostic(39916, "plasma_current")
    assert again["processing"]["reference"]["field"] != -1
    assert again["processing"]["baseline"]["revisions"]


def test_the_cache_follows_an_edit_of_the_file(tmp_path):
    path = tmp_path / "vest.yaml"
    _write(path, _diagnostic_document(gain=1.0))
    assert resolve_vest_diagnostic(1, "probe", info_file=str(path)) == {"gain": 1.0}
    stat = os.stat(path)
    # Same size and the same mtime: only the content differs ...
    _write(path, _diagnostic_document(gain=2.0))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000))
    assert resolve_vest_diagnostic(1, "probe", info_file=str(path)) == {"gain": 2.0}
    # ... and a rewrite restoring the old mtime but changing the size is seen too.
    _write(path, _diagnostic_document(gain=3.25))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert resolve_vest_diagnostic(1, "probe", info_file=str(path)) == {"gain": 3.25}


def test_the_cache_is_keyed_by_file(tmp_path):
    one = _write(tmp_path / "one.yaml", _diagnostic_document(gain=1.0))
    two = _write(tmp_path / "two.yaml", _diagnostic_document(gain=2.0))
    assert resolve_vest_diagnostic(1, "probe", info_file=one)["gain"] == 1.0
    assert resolve_vest_diagnostic(1, "probe", info_file=two)["gain"] == 2.0
    assert utils._cached_document.cache_info().maxsize >= 2


# --- G5: revision keys -----------------------------------------------------------------


_BASE = {"probe_baseline_end": 10, "index_start": 0}


def test_a_misspelt_revision_key_is_refused():
    revisions = [{"from_shot": 10, "probe_baseline_ned": 20}]
    with pytest.raises(VestConfigurationError, match="unknown key 'probe_baseline_ned'"):
        resolve_shot_revisions(_BASE, revisions, 10, context="window")


def test_revision_keys_are_checked_for_shots_outside_the_bad_era():
    revisions = [{"to_shot": 5, "index_start": 1}, {"from_shot": 10, "probe_baseline_ned": 20}]
    with pytest.raises(VestConfigurationError, match="revision 1: unknown key"):
        resolve_shot_revisions(_BASE, revisions, 1, context="window")


def test_declared_era_only_keys_are_accepted_and_never_resolved():
    base = {**_BASE, ERA_ONLY_KEYS: ["fl10"]}
    revisions = [{"from_shot": 10, "to_shot": 20, "fl10": {"field": 25}}]
    inside, provenance = resolve_shot_revisions_with_provenance(base, revisions, 15, context="ref")
    outside = resolve_shot_revisions(base, revisions, 21, context="ref")
    assert inside == {**_BASE, "fl10": {"field": 25}}
    assert provenance["revision_index"] == 0
    assert outside == _BASE  # absent outside its era, and the declaration is consumed
    assert base[ERA_ONLY_KEYS] == ["fl10"]


@pytest.mark.parametrize(
    ("declared", "message"),
    [
        ("fl10", "must be a list"),
        (["index_start"], "which the base also defines"),
        (["fl10", "fl10"], "twice"),
        (["fl10", "unused"], "which no revision sets"),
        (["from_shot"], "may not name"),
        ([3], "non-empty key names"),
    ],
)
def test_era_only_declarations_are_validated(declared, message):
    base = {**_BASE, ERA_ONLY_KEYS: declared}
    revisions = [{"from_shot": 10, "fl10": 1}]
    with pytest.raises(VestConfigurationError, match=message):
        resolve_shot_revisions(base, revisions, 10, context="ref")


def test_resolve_vest_diagnostic_consumes_a_top_level_declaration(tmp_path):
    path = _write(
        tmp_path / "vest.yaml",
        _diagnostic_document(
            source={"field": 1},
            era_only_keys=["unavailable_reason"],
            revisions=[{"to_shot": 9, "source": None, "unavailable_reason": "absent"}],
        ),
    )
    assert resolve_vest_diagnostic(5, "probe", info_file=path) == {
        "source": None,
        "unavailable_reason": "absent",
    }
    assert resolve_vest_diagnostic(10, "probe", info_file=path) == {"source": {"field": 1}}


def test_a_revision_cannot_change_a_block_shape():
    with pytest.raises(VestConfigurationError, match="revision 0.*cannot merge"):
        resolve_shot_revisions({"window": {"a": 1}}, [{"from_shot": 1, "window": 2}], 1, context="w")


def test_resolved_revisions_share_nothing_with_the_table():
    base = {"window": {"index": [1, 2]}}
    revisions = [{"from_shot": 1, "window": {"extra": [3]}}]
    resolved = resolve_shot_revisions(base, revisions, 1, context="w")
    resolved["window"]["index"].append(9)
    resolved["window"]["extra"].append(9)
    assert base == {"window": {"index": [1, 2]}}
    assert revisions == [{"from_shot": 1, "window": {"extra": [3]}}]


@pytest.mark.parametrize("bad", [True, 41446.5, -1, "abc", None])
def test_shot_numbers_must_be_non_negative_integers(bad):
    revisions = [{"from_shot": 10, "index_start": 1}]
    with pytest.raises(VestConfigurationError, match="shot must be a non-negative integer"):
        resolve_shot_revisions(_BASE, revisions, bad, context="w")


@pytest.mark.parametrize("bad", [True, 41446.5, -1, "abc"])
def test_revision_bounds_must_be_non_negative_integers(bad):
    revisions = [{"from_shot": bad, "index_start": 1}]
    with pytest.raises(VestConfigurationError, match="from_shot must be a non-negative integer"):
        resolve_shot_revisions(_BASE, revisions, 10, context="w")


def test_integral_shot_spellings_still_resolve():
    revisions = [{"from_shot": 10, "to_shot": 20.0, "index_start": 1}]
    for shot in (15, 15.0, "15"):
        assert resolve_shot_revisions(_BASE, revisions, shot, context="w")["index_start"] == 1


# --- the packaged vest.yaml satisfies the stricter rules -------------------------------


def _revision_tables(node, path=()):
    if isinstance(node, dict):
        if isinstance(node.get("revisions"), list):
            yield path, node
        for key, value in node.items():
            yield from _revision_tables(value, (*path, key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _revision_tables(value, (*path, index))


def _bound_shots(revisions):
    shots = {0, 60000}
    for revision in revisions:
        for bound in ("from_shot", "to_shot"):
            if revision.get(bound) is not None:
                shots.update({revision[bound] - 1, revision[bound], revision[bound] + 1})
    return sorted(shot for shot in shots if shot >= 0)


def test_every_packaged_revision_table_resolves_at_every_era_bound():
    document = _policy_document(None)
    tables = list(_revision_tables(document))
    assert len(tables) >= 13
    for path, table in tables:
        base = {key: value for key, value in table.items() if key != "revisions"}
        context = ".".join(map(str, path))
        for shot in _bound_shots(table["revisions"]):
            resolved = resolve_shot_revisions(base, table["revisions"], shot, context=context)
            assert ERA_ONLY_KEYS not in resolved


def test_every_packaged_diagnostic_resolves_at_the_boundary_shots():
    diagnostics = _policy_document(None)[0]["diagnostics"]
    for shot in (39437, 39438, 45895, 45896, 46402, 46403, 46404, 48371, 48372):
        for name in diagnostics:
            assert ERA_ONLY_KEYS not in resolve_vest_diagnostic(shot, name)


def test_packaged_era_only_declarations():
    """The keys vest.yaml sets only inside eras, recorded so a new one is a reviewed change."""
    declared = {
        ".".join(map(str, path)): table[ERA_ONLY_KEYS]
        for path, table in _revision_tables(_policy_document(None))
        if ERA_ONLY_KEYS in table
    }
    assert declared == {
        "0.diagnostics.plasma_current.processing.reference": ["fl10"],
        "0.diagnostics.diamagnetic_flux": ["unavailable_reason"],
        "0.diagnostics.equilibrium_magnetics.processing.known_faults": ["flux_loops"],
        "0.diagnostics.langmuir_probe_mid": ["vd3", "tip_length_mm", "tip_radius_mm"],
        "0.diagnostics.langmuir_probe_upper": ["vd3", "tip_length_mm", "tip_radius_mm"],
        "0.ec_launchers.beams.ech_6kw": [
            "geometry_status",
            "source",
            "launch_plane",
            "launching_position",
            "direction",
            "polarization",
            "stations",
        ],
    }
