"""Time-axis resolution for dynamic IDS nodes.

IMAS lets an IDS declare its time layout with
``ids_properties.homogeneous_time``: 1 means every dynamic node shares
``<ids>.time``, 0 means each node carries its own ``time`` sibling. OMAS'
``ODS.time(ids)`` only answers the homogeneous case and returns ``None``
otherwise, which surfaced as a ``TypeError`` deep inside the onset helpers when
the packaged 39915 sample became heterogeneous for ``magnetics``.
"""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

import vaft
from vaft.omas.general import signal_time


def _heterogeneous_ods():
    ods = ODS()
    ods["magnetics.ids_properties.homogeneous_time"] = 0
    ods["magnetics.time"] = np.linspace(0.0, 1.0, 5)
    ods["magnetics.flux_loop.0.flux.time"] = np.linspace(10.0, 11.0, 5)
    ods["magnetics.flux_loop.0.flux.data"] = np.arange(5, dtype=float)
    return ods


def _homogeneous_ods():
    ods = ODS()
    ods["magnetics.ids_properties.homogeneous_time"] = 1
    ods["magnetics.time"] = np.linspace(0.0, 1.0, 5)
    ods["magnetics.flux_loop.0.flux.data"] = np.arange(5, dtype=float)
    return ods


def test_heterogeneous_node_resolves_to_its_own_axis():
    ods = _heterogeneous_ods()
    # The accessor that used to be called here cannot answer for this IDS.
    assert ods.time("magnetics") is None
    resolved = signal_time(ods, "magnetics.flux_loop.0.flux.data")
    np.testing.assert_array_equal(resolved, ods["magnetics.flux_loop.0.flux.time"])


def test_homogeneous_node_falls_back_to_the_ids_axis():
    ods = _homogeneous_ods()
    resolved = signal_time(ods, "magnetics.flux_loop.0.flux.data")
    np.testing.assert_array_equal(resolved, ods["magnetics.time"])


def test_an_empty_node_axis_does_not_shadow_the_ids_axis():
    """A declared-but-unfilled node time must not win over a populated one."""
    ods = _homogeneous_ods()
    ods["magnetics.flux_loop.0.flux.time"] = np.array([], dtype=float)
    np.testing.assert_array_equal(
        signal_time(ods, "magnetics.flux_loop.0.flux.data"), ods["magnetics.time"]
    )


def test_a_node_with_no_axis_anywhere_reports_both_paths_it_tried():
    ods = ODS()
    ods["magnetics.flux_loop.0.flux.data"] = np.arange(3, dtype=float)
    with pytest.raises(KeyError) as excinfo:
        signal_time(ods, "magnetics.flux_loop.0.flux.data")
    message = str(excinfo.value)
    assert "magnetics.flux_loop.0.flux.time" in message
    assert "magnetics.time" in message


def test_onset_helpers_work_on_the_packaged_heterogeneous_sample():
    """The regression itself: these raised TypeError via ODS.time()."""
    ods = vaft.omas.sample_ods()
    assert ods["magnetics.ids_properties.homogeneous_time"] == 0

    for value in (
        vaft.omas.find_vloop_onset(ods),
        vaft.omas.find_ip_onset(ods),
        vaft.omas.find_breakdown_onset(ods),
    ):
        assert np.isfinite(value)

    # change_time_convention drives all three and was the reported failure.
    vaft.omas.change_time_convention(vaft.omas.sample_ods(), convention="breakdown")


def test_find_pf_active_onset_returns_an_onset_for_every_coil():
    """It iterated ``pf_active.channel``, which the DD does not define.

    OMAS yields an empty list for a missing node instead of raising, so the
    helper silently returned ``[]`` for every caller rather than failing loudly.
    """
    ods = vaft.omas.sample_ods()
    coil_count = len(ods["pf_active.coil"])
    assert coil_count == 10

    onsets = vaft.omas.find_pf_active_onset(ods)

    assert len(onsets) == coil_count
    assert all(np.isfinite(onset) for onset in onsets)
