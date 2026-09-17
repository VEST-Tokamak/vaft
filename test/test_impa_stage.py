"""The standalone IMPA stage product (issue #305).

IMPA is insertable and campaign-dependent: raw fields can exist while the array
is withdrawn, its geometry is self-calibrated per shot, and its vertical-field
sensors are still being qualified (#154, #304). Those are poor invariants for
the baseline magnetics product, so the array has its own stage whose verdict is
its own -- and whose product is self-contained, owing nothing to `main`.
"""

from __future__ import annotations

import warnings

import pytest
from omas import ODS

from vaft.data import resources
from vaft.omas import vest_upstream
from vaft.omas.vest_upstream import build_impa_ods


PACKAGED_SHOTS = (39915, 41524, 41672)


def _raw(shot: int):
    return resources.data_path(f"samples/{shot}/source/vest_{shot}_daq_raw.json.gz")


@pytest.fixture(scope="module")
def rejected_product():
    """39915's array is not in a usable state; the fit saturates its bound."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_impa_ods(shot=39915, raw_source=_raw(39915))


def test_the_product_is_self_contained(rejected_product):
    ods, _ = rejected_product
    # No wall, no pf_active, no equilibrium: nothing of the baseline shot is
    # duplicated to make the source stand on its own.
    assert sorted(ods.keys()) == ["dataset_description", "magnetics"]


def test_the_array_starts_at_index_zero_in_its_own_product(rejected_product):
    """Nothing precedes it here, so it does not append after someone else."""
    ods, _ = rejected_product
    hall = ods["magnetics.b_field_tor_probe"]
    vertical = ods["magnetics.b_field_pol_probe"]
    assert len(hall) == 8 and len(vertical) == 8
    assert all(
        str(hall[index]["identifier"]).startswith("impa:") for index in range(len(hall))
    )
    assert all(
        str(vertical[index]["identifier"]).startswith("impa:")
        for index in range(len(vertical))
    )


def test_a_rejected_calibration_is_kept_locally_and_not_published(rejected_product):
    """`rejected` is not a replicable status, so the source never sees it.

    The verdict travels with the product rather than being thrown away: the
    reason a shot is absent from `impa` is answerable from the local record.
    """
    _, manifest = rejected_product
    assert manifest["status"] == "rejected"
    assert manifest["calibration"]["status"] == "invalid"
    assert manifest["quality_summary"]["rejected"] == ["impa"]
    assert manifest["calibration"]["geometry_method"]
    assert manifest["calibration"]["r0"] is not None

    from vaft.database.replication import REPLICABLE_STATUSES

    assert manifest["status"] not in REPLICABLE_STATUSES


@pytest.mark.parametrize("shot", PACKAGED_SHOTS)
def test_every_packaged_shot_yields_a_verdict_rather_than_an_exception(shot):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, manifest = build_impa_ods(shot=shot, raw_source=_raw(shot))
    assert manifest["stage"] == "impa"
    assert manifest["status"] in {"success", "partial", "rejected", "unavailable"}


def test_a_product_with_no_waveform_still_declares_its_time_mode(rejected_product):
    """The DD rule: an IDS with no `.time` node is homogeneous_time 2, not unset.

    Left unset, a publishable `partial` product would reach IMAS with no time
    mode to honour.
    """
    ods, _ = rejected_product
    assert "time" not in ods["magnetics"]
    assert ods["magnetics.ids_properties.homogeneous_time"] == 2


def test_asking_where_the_axis_is_does_not_invent_a_probe_array(monkeypatch):
    """Probing an absent node in OMAS materializes it, and this ODS is written."""
    import numpy as np

    monkeypatch.setattr(
        vest_upstream, "_archived_field_codes", lambda path: set(range(114, 122))
    )

    def hall_only(ods, shot, tstart, tend, dt, **kwargs):
        # No `field.time` anywhere, so the search falls through to the
        # vertical-field node -- which this era does not wire.
        ods["magnetics.b_field_tor_probe.0.identifier"] = "impa:IMPA 01"
        return {"status": "invalid", "checks": {}, "reasons": [], "provenance": {}}

    monkeypatch.setattr(vest_upstream, "impa_mapper", hall_only)
    ods, _ = build_impa_ods(shot=39915, raw_source=_raw(39915))

    assert "b_field_pol_probe" not in ods["magnetics"]


def test_the_product_records_the_machine_description_it_was_built_from(rejected_product):
    """No static product travels beside a sparse source, so it says so itself."""
    _, manifest = rejected_product
    assert len(manifest["input"]["impa_configuration_sha256"]) == 64
    assert manifest["configuration"]["expected_fields"] == [114, 115, 116, 117, 118, 119, 120, 121]


def test_a_shot_with_no_archived_impa_channel_is_unavailable(tmp_path, monkeypatch):
    """Normal for an insertable diagnostic, and not a fault in the run."""
    monkeypatch.setattr(vest_upstream, "_archived_field_codes", lambda path: {1, 12})
    called = []
    monkeypatch.setattr(
        vest_upstream, "impa_mapper", lambda *a, **k: called.append(a) or {}
    )

    _, manifest = build_impa_ods(shot=39915, raw_source=_raw(39915))

    assert manifest["status"] == "unavailable"
    assert manifest["quality_summary"]["unavailable"] == ["impa"]
    assert called == [], "the mapper must not run when nothing was archived"


def _mapped(monkeypatch, status: str, *, wire_all: bool = True):
    """Drive the manifest verdict from a mapper whose outcome is chosen here."""
    import numpy as np

    fields = [114, 115, 116, 117, 118, 119, 120, 121]
    monkeypatch.setattr(
        vest_upstream, "_archived_field_codes", lambda path: set(fields if wire_all else fields[:-1])
    )

    def fake_mapper(ods, shot, tstart, tend, dt, **kwargs):
        time = np.arange(tstart, tend, dt)
        ods["magnetics.b_field_tor_probe.0.identifier"] = "impa:IMPA 01"
        ods["magnetics.b_field_tor_probe.0.field.time"] = time
        ods["magnetics.b_field_tor_probe.0.field.data"] = np.zeros_like(time)
        return {"status": status, "checks": {}, "reasons": [], "provenance": {}}

    monkeypatch.setattr(vest_upstream, "impa_mapper", fake_mapper)
    return build_impa_ods(shot=39915, raw_source=_raw(39915))


def test_an_accepted_calibration_publishes_and_carries_its_realized_axis(monkeypatch):
    ods, manifest = _mapped(monkeypatch, "valid")

    assert manifest["status"] == "success"
    assert manifest["time_grid"]["impa"]["sample_count"] > 0
    # The coordinate is the axis the channels realized, not the one requested.
    assert len(ods["magnetics.time"]) == manifest["time_grid"]["impa"]["sample_count"]
    assert ods["magnetics.ids_properties.homogeneous_time"] == 1


def test_a_warning_calibration_publishes_as_partial(monkeypatch):
    _, manifest = _mapped(monkeypatch, "warning")
    assert manifest["status"] == "partial"


def test_an_unwired_channel_makes_the_product_partial_not_failed(monkeypatch):
    _, manifest = _mapped(monkeypatch, "valid", wire_all=False)
    assert manifest["status"] == "partial"
    assert manifest["quality_summary"]["missing"] == ["impa:field-121"]
