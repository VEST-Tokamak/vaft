"""The non-mutating ODS path-access contract (issue #118).

An OMAS ODS creates paths on access, so a *probe* corrupts the object it was
only meant to inspect. The corruption is invisible where anyone would look for
it -- `flat()` never shows it -- and surfaces much later as a consistency-check
failure on load. That has been rediscovered independently in EFIT constraint
generation, in the magnetics signal-quality validator and in the plasma-free
vacuum benchmark.

These tests pin two things: that the shared accessors never mutate, and that
they still tell apart the states `path_exists` has always distinguished. The
last group deliberately reproduces the raw failure, so the reason this module
exists stays explicit and pinned to the OMAS version VAFT supports.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("omas")
from omas import ODS, load_omas_json, save_omas_json

from vaft.machine_mapping.magnetics import POLOIDAL_ANGLE
from vaft.ods_access import get_path, path_count, path_exists, path_value, set_path


def _fingerprint(node, prefix: str = "") -> list[str]:
    """Every nested key, which `flat()` does not report.

    A materialized placeholder is absent from `flat()` but present in `keys()`,
    so `flat()` is not a mutation detector -- checking it was exactly how the
    problem stayed hidden.
    """
    found: list[str] = []
    try:
        keys = list(node.keys())
    except Exception:
        return found
    for key in keys:
        path = f"{prefix}.{key}" if prefix else str(key)
        found.append(path)
        try:
            found.extend(_fingerprint(node[key], path))
        except Exception:
            pass
    return sorted(found)


@pytest.fixture
def ods() -> ODS:
    """One populated probe, one deliberately empty channel, one NaN scalar."""
    source = ODS(consistency_check=False)
    source["magnetics.time"] = np.linspace(0.26, 0.36, 5)
    source["magnetics.b_field_pol_probe.0.field.data"] = np.arange(5.0)
    source["magnetics.b_field_pol_probe.0.position.r"] = float("nan")
    # What the VEST mappers write for a channel that is wired but unmapped.
    source["magnetics.b_field_pol_probe.1.field.data"] = np.array([], dtype=float)
    return source


ABSENT = (
    ("FLT_1D leaf", "magnetics.b_field_pol_probe.0.field.time"),
    ("INT_0D leaf", "magnetics.b_field_pol_probe.0.field.validity"),
    ("leaf under an absent AoS", "magnetics.ip.0.data"),
    ("AoS container", "magnetics.flux_loop"),
    ("nested STRUCTURE", "magnetics.b_field_pol_probe.1.position.r"),
    ("whole IDS", "equilibrium.time"),
)


# ---------------------------------------------------------------------------
# Nothing is created
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,path", ABSENT, ids=[label for label, _ in ABSENT])
def test_probing_an_absent_path_creates_nothing(ods, label, path):
    before = _fingerprint(ods)

    assert path_exists(ods, path) is False
    assert path_value(ods, path) is None
    assert path_value(ods, path, "DEFAULT") == "DEFAULT"
    assert path_count(ods, path) == 0
    with pytest.raises(KeyError):
        get_path(ods, path)

    assert _fingerprint(ods) == before


def test_a_raw_read_does_auto_vivify_under_the_supported_omas(ods):
    """The reason the accessors exist, pinned rather than assumed.

    If a future OMAS stops creating paths on read this fails, and the helper can
    be reconsidered instead of being carried forever on folklore.
    """
    path = "magnetics.b_field_pol_probe.0.field.validity"
    before = _fingerprint(ods)

    ods[path]  # noqa: B018 -- the point is the side effect

    after = _fingerprint(ods)
    assert path in after and path not in before, (
        "a bare subscript no longer auto-vivifies; revisit vaft.ods_access"
    )
    # And the damage is invisible to the obvious check.
    assert set(ODS(consistency_check=False).flat()) == set()
    assert path not in set(ods.flat())


def test_repeated_probing_is_stable(ods):
    before = _fingerprint(ods)
    for _ in range(3):
        for _label, path in ABSENT:
            path_exists(ods, path)
            path_value(ods, path)
            path_count(ods, path)
    assert _fingerprint(ods) == before


def test_probing_leaves_the_ods_serializable_and_consistency_clean(ods, tmp_path):
    """The failure mode that reached real data, from both ends.

    A materialized `FLT_1D` fails as a shape/coordinate mismatch and a
    materialized `INT_0D` as a NaN-to-integer conversion. Neither may survive a
    probe.
    """
    for _label, path in ABSENT:
        path_exists(ods, path)
        path_value(ods, path)

    target = tmp_path / "probed.json"
    save_omas_json(ods, str(target))
    reloaded = load_omas_json(str(target), consistency_check=True)
    assert set(reloaded.flat()) == set(ods.flat())


def test_a_probed_array_leaf_corrupts_the_saved_ods(tmp_path):
    """#118's demonstrated case, reproduced.

    An auto-vivified `FLT_1D` is stored with no shape, and fails against the
    coordinate it is declared on. This is what reached a real EFIT constraints
    ODS and was caught only by a regression test.
    """
    corrupt = ODS(consistency_check=False)
    corrupt["magnetics.time"] = np.linspace(0.26, 0.36, 5)
    corrupt["magnetics.b_field_pol_probe.0.field.time"] = np.linspace(0.26, 0.36, 5)

    corrupt["magnetics.b_field_pol_probe.0.field.data"]  # noqa: B018

    target = tmp_path / "corrupt_array.json"
    save_omas_json(corrupt, str(target))
    with pytest.raises(ValueError, match="inconsistent with coordinates"):
        load_omas_json(str(target), consistency_check=True)


def test_a_probed_integer_leaf_corrupts_the_saved_ods(tmp_path):
    """The same mechanism on an `INT_0D`, which is how it resurfaced.

    A materialized `validity` holds NaN, and an integer node cannot. This broke
    seven EFIT tests during the magnetics signal-quality work and sat silently
    in the vacuum benchmark, where a vacuum shot has no `magnetics.ip` at all.
    """
    corrupt = ODS(consistency_check=False)
    corrupt["magnetics.time"] = np.linspace(0.26, 0.36, 5)
    corrupt["magnetics.b_field_pol_probe.0.field.data"] = np.arange(5.0)

    corrupt["magnetics.b_field_pol_probe.0.field.validity"]  # noqa: B018

    target = tmp_path / "corrupt_int.json"
    save_omas_json(corrupt, str(target))
    with pytest.raises(ValueError, match="cannot convert float NaN"):
        load_omas_json(str(target), consistency_check=True)


def test_the_same_two_probes_through_the_accessors_are_harmless(tmp_path):
    """The point of the module, stated against the two failures above."""
    safe = ODS(consistency_check=False)
    safe["magnetics.time"] = np.linspace(0.26, 0.36, 5)
    safe["magnetics.b_field_pol_probe.0.field.data"] = np.arange(5.0)

    assert path_value(safe, "magnetics.b_field_pol_probe.0.field.validity") is None
    assert path_value(safe, "magnetics.b_field_pol_probe.1.field.data") is None

    target = tmp_path / "safe.json"
    save_omas_json(safe, str(target))
    load_omas_json(str(target), consistency_check=True)


# ---------------------------------------------------------------------------
# The states stay distinct
# ---------------------------------------------------------------------------

def test_present_values_come_back(ods):
    assert path_exists(ods, "magnetics.b_field_pol_probe.0.field.data")
    assert np.array_equal(
        path_value(ods, "magnetics.b_field_pol_probe.0.field.data"), np.arange(5.0)
    )
    assert np.array_equal(
        get_path(ods, "magnetics.b_field_pol_probe.0.field.data"), np.arange(5.0)
    )
    assert path_count(ods, "magnetics.b_field_pol_probe") == 2


def test_a_deliberately_empty_array_is_present_not_absent(ods):
    """The mapper writes `np.array([])` for an unwired channel on purpose.

    Collapsing that into "absent" would erase the difference between a channel
    that reported nothing and one that was never asked.
    """
    path = "magnetics.b_field_pol_probe.1.field.data"

    assert path_exists(ods, path) is True
    assert np.asarray(path_value(ods, path)).size == 0
    assert path_count(ods, path) == 0


def test_a_present_nan_scalar_is_present(ods):
    path = "magnetics.b_field_pol_probe.0.position.r"

    assert path_exists(ods, path) is True
    assert np.isnan(path_value(ods, path))


def test_an_empty_structure_reads_as_absent(ods):
    """The convention `path_exists` has always had, and which the DD shares.

    Materializing a branch and then finding it empty is how absence presents on
    a dynamic ODS, so an empty branch must not read as content -- that is what
    let dead b-probe channels through the EFIT constraints filter.
    """
    ods["magnetics.flux_loop"]  # noqa: B018 -- materialize an empty container

    assert path_exists(ods, "magnetics.flux_loop") is False
    assert path_value(ods, "magnetics.flux_loop", "DEFAULT") == "DEFAULT"
    assert path_count(ods, "magnetics.flux_loop") == 0


def test_a_sub_ods_accepts_relative_paths(ods):
    magnetics = ods["magnetics"]

    assert path_exists(magnetics, "b_field_pol_probe.0.field.data")
    assert not path_exists(magnetics, "b_field_pol_probe.0.field.validity")
    assert path_count(magnetics, "b_field_pol_probe") == 2


# ---------------------------------------------------------------------------
# CodeParameters: where membership does not work, and why that is safe
# ---------------------------------------------------------------------------

@pytest.fixture
def code_parameters() -> ODS:
    source = ODS(consistency_check=False)
    source["equilibrium.code.parameters.time_slice.0.aeqdsk.terror"] = 2.5e-6
    return source


def test_code_parameters_values_are_still_reachable(code_parameters):
    """A subtree OMAS membership cannot see into.

    `in` and `.get` both reach for `omas_data` on a `CodeParameters`, so they
    raise -- for present paths as much as absent ones. Treating that as absence
    would silently drop EFIT's whole convergence block.
    """
    path = "equilibrium.code.parameters.time_slice.0.aeqdsk.terror"

    with pytest.raises(AttributeError):
        path in code_parameters  # noqa: B015 -- documenting why the fallback exists

    assert path_exists(code_parameters, path) is True
    assert path_value(code_parameters, path) == pytest.approx(2.5e-6)
    assert get_path(code_parameters, path) == pytest.approx(2.5e-6)


def test_code_parameters_is_safe_to_probe_by_subscript(code_parameters):
    """Why the fallback cannot reintroduce the bug.

    The fallback is reached only where membership is unsupported, and every
    node kind in that situation is a static container that raises rather than
    vivifying. If OMAS ever makes `CodeParameters` dynamic, this fails.
    """
    before = _fingerprint(code_parameters)

    with pytest.raises(KeyError):
        code_parameters["equilibrium.code.parameters.time_slice.0.aeqdsk.nope"]

    assert _fingerprint(code_parameters) == before
    assert path_exists(code_parameters, "equilibrium.code.parameters.time_slice.0.aeqdsk.nope") is False
    assert _fingerprint(code_parameters) == before


def test_descending_through_a_scalar_leaf_is_absence(code_parameters):
    import json

    source = ODS(consistency_check=False)
    source["equilibrium.code.parameters"] = json.dumps({"a": 1})

    assert path_exists(source, "equilibrium.code.parameters.a") is False
    assert path_value(source, "equilibrium.code.parameters.a", "D") == "D"


# ---------------------------------------------------------------------------
# Required reads, writes, and plain mappings
# ---------------------------------------------------------------------------

def test_get_path_raises_rather_than_planting_a_placeholder(ods):
    before = _fingerprint(ods)

    with pytest.raises(KeyError):
        get_path(ods, "magnetics.b_field_pol_probe.0.field.validity")

    assert _fingerprint(ods) == before


def test_set_path_is_the_one_primitive_that_creates(ods):
    set_path(ods, "magnetics.b_field_pol_probe.0.field.validity", 0)

    assert path_exists(ods, "magnetics.b_field_pol_probe.0.field.validity")
    assert path_value(ods, "magnetics.b_field_pol_probe.0.field.validity") == 0


def test_plain_mappings_keep_working():
    payload = {"a": {"b": [10, 20]}, "empty": {}}

    assert path_exists(payload, "a.b")
    assert path_value(payload, "a.b.1") == 20
    assert path_count(payload, "a.b") == 2
    assert not path_exists(payload, "a.c")
    assert path_value(payload, "a.b.5", "D") == "D"
    with pytest.raises(KeyError):
        get_path(payload, "a.c")

    set_path(payload, "a.c.d", 3)
    assert payload["a"]["c"]["d"] == 3


# ---------------------------------------------------------------------------
# Consumers that must not mutate what they inspect
# ---------------------------------------------------------------------------

def test_counting_a_string_leaf_is_zero_not_its_length():
    """A string is sized but is not a container of entries.

    `len("constraint equilibrium")` is 22, which reads as a plausible channel
    count and is not the question the caller asked.
    """
    source = ODS(consistency_check=False)
    source["equilibrium.ids_properties.comment"] = "constraint equilibrium"

    assert path_count(source, "equilibrium.ids_properties.comment") == 0
    assert path_count({"a": "hello"}, "a") == 0


def test_selecting_vacuum_channels_never_mutates_the_ods():
    """`vacuum_magnetics` probes two optional nodes per channel.

    `poloidal_angle` has a documented fallback for an ODS that stores none, and
    `position` may be absent on a partially mapped channel. Both were read by
    bare subscript, so merely *listing* the channels planted branches in the
    caller's ODS -- and the position read then died on `float(ODS)` instead of
    skipping the channel.
    """
    from vaft.omas.vacuum_magnetics import _candidates, _poloidal_angle

    source = ODS(consistency_check=False)
    source["magnetics.time"] = np.linspace(0.26, 0.36, 5)
    source["magnetics.b_field_pol_probe.0.field.data"] = np.arange(5.0)
    before = _fingerprint(source)

    # No `poloidal_angle` stored: the fallback answers, and nothing is created.
    assert _poloidal_angle(source, "magnetics.b_field_pol_probe.0") == pytest.approx(
        POLOIDAL_ANGLE
    )
    assert _fingerprint(source) == before

    # No `position` stored: the channel is skipped, not crashed on.
    assert _candidates(source) == []
    assert _fingerprint(source) == before
