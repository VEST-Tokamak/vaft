"""Composing `main` with a sparse optional source is explicit (issue #305).

Reading one source returns one source. That is what keeps a missing entry
meaningful: absence from `impa` says no IMPA product was published, and nothing
about whether the baseline shot exists or succeeded. Analysis that wants both
asks for both here, and the result still says where each channel came from.
"""

from __future__ import annotations

import pytest
from omas import ODS

from vaft import database
from vaft.database.composition import compose


def _baseline() -> ODS:
    """Two equilibrium probes, addressed positionally by everything downstream."""
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 39915
    for index, name in enumerate(("MD_A", "MD_B")):
        ods[f"magnetics.b_field_pol_probe.{index}.name"] = name
        ods[f"magnetics.b_field_pol_probe.{index}.identifier"] = name
    return ods


def _impa_product() -> ODS:
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 39915
    for index in range(2):
        ods[f"magnetics.b_field_tor_probe.{index}.identifier"] = f"impa:IMPA 0{index + 1}"
        ods[f"magnetics.b_field_tor_probe.{index}.name"] = f"IMPA 0{index + 1}"
    ods["magnetics.b_field_pol_probe.0.identifier"] = "impa:IMPA Bz 01"
    ods["magnetics.b_field_pol_probe.0.name"] = "IMPA Bz 01"
    return ods


@pytest.fixture
def two_sources(monkeypatch):
    products = {"main": _baseline(), "impa": _impa_product()}

    def fake_load(shot, *, source=None, paths=None, occurrence=None, **kwargs):
        import copy

        return copy.deepcopy(products[source])

    monkeypatch.setattr(database, "load", fake_load)
    return products


def test_composition_appends_after_the_base_and_never_moves_its_indices(two_sources):
    """k-files and the constraint builder address probes by index."""
    composed, provenance = compose(39915)

    assert composed["magnetics.b_field_pol_probe.0.name"] == "MD_A"
    assert composed["magnetics.b_field_pol_probe.1.name"] == "MD_B"
    assert composed["magnetics.b_field_pol_probe.2.name"] == "IMPA Bz 01"
    assert len(composed["magnetics.b_field_tor_probe"]) == 2
    assert provenance["base"] == "main"
    assert provenance["contributed"] == {"impa": 3}


def test_the_provenance_says_where_every_appended_channel_came_from(two_sources):
    _, provenance = compose(39915)

    appended = {entry["index"]: entry for entry in provenance["appended"]}
    assert all(entry["source"] == "impa" for entry in appended.values())
    vertical = [e for e in provenance["appended"] if e["node"].endswith("pol_probe")]
    assert vertical == [
        {"source": "impa", "node": "magnetics.b_field_pol_probe", "source_index": 0, "index": 2}
    ]


def test_reading_one_source_never_brings_the_other(two_sources):
    """The absence of a silent merge is the whole point of the split."""
    baseline = database.load(39915, source="main")
    identifiers = [
        str(baseline[f"magnetics.b_field_pol_probe.{index}.identifier"])
        for index in range(len(baseline["magnetics.b_field_pol_probe"]))
    ]
    assert not any(name.startswith("impa:") for name in identifiers)
    assert "b_field_tor_probe" not in baseline["magnetics"]


def test_a_single_source_is_not_a_composition(two_sources):
    with pytest.raises(ValueError, match="at least one source"):
        compose(39915, sources=("main",))


def test_an_unknown_source_fails_before_any_read(monkeypatch):
    called = []
    monkeypatch.setattr(database, "load", lambda *a, **k: called.append(a))
    from vaft.database.sources import UnknownSourceError

    with pytest.raises(UnknownSourceError):
        compose(39915, sources=("main", "not-a-source"))
    assert called == []
