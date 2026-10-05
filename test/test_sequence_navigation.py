"""One navigation contract for samples, frames and stored slices (issue #1380, phase 2).

``PlotCapability.sequence`` states, for any input, how a reader moves through a
plot's scientific states: the storage ``kind``, the selector ``option``, the
``states`` it accepts, the ``selected`` start and what the states' values mean.
The control layer, the GUI player and ``animation=True`` all read it, so they
agree on which states exist; ``sequence_values`` supplies each state's physical
value, never an index.
"""

from __future__ import annotations

import copy

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from _sample_fixtures import sample_ods
from vaft.omas.entries import normalize_entries
from vaft.plot.backend.discovery import describe_entries, sequence_values
from vaft.plot.controls import controls_for

KINDS = {"time_index": "samples", "frame_index": "frames", "time_slice": "stored"}


@pytest.fixture(scope="module", params=[39915, 40600])
def catalog(request):
    entries = normalize_entries(sample_ods(request.param))
    return entries, [record for record in describe_entries(entries) if record.available]


def _slice_control(record):
    found = [c for c in controls_for(record) if c.group == "slice"]
    assert len(found) <= 1, record.name
    return found[0] if found else None


def test_a_plot_has_a_sequence_exactly_when_it_offers_a_slice_control(catalog):
    _, records = catalog
    with_sequence = 0
    for record in records:
        control = _slice_control(record)
        assert bool(record.sequence) == (control is not None), record.name
        with_sequence += bool(record.sequence)
    assert with_sequence >= 7


def test_the_control_is_the_sequence(catalog):
    _, records = catalog
    for record in filter(lambda r: r.sequence, records):
        sequence, control = record.sequence, _slice_control(record)
        assert control.name == sequence["option"]
        assert sequence["kind"] == KINDS[sequence["option"]]
        assert control.default == sequence["selected"] and sequence["selected"] in sequence["states"]
        if control.kind == "range":
            low, high, step = control.options
            assert tuple(range(low, high + 1, step)) == tuple(sequence["states"])
        else:
            assert tuple(control.options) == tuple(sequence["states"])


def test_every_state_has_a_physical_value_inside_the_stated_span(catalog):
    entries, records = catalog
    for record in filter(lambda r: r.sequence, records):
        sequence = record.sequence
        coordinate, unit, values = sequence_values(record.name, entries, sequence["option"])
        assert (coordinate, unit) == (sequence["coordinate"], sequence["unit"]) == ("time", "s")
        chosen = np.asarray([values[i] for i in sequence["states"]], dtype=float)
        assert np.all(np.isfinite(chosen)), record.name
        assert chosen.min() == pytest.approx(sequence["start"]), record.name
        assert chosen.max() == pytest.approx(sequence["stop"]), record.name


def test_the_three_kinds_all_appear_on_the_packaged_samples():
    seen = set()
    for shot in (39915, 40600):
        entries = normalize_entries(sample_ods(shot))
        seen |= {
            record.sequence["kind"]
            for record in describe_entries(entries)
            if record.available and record.sequence
        }
    assert seen == {"samples", "frames", "stored"}


def test_a_single_state_is_no_sequence():
    single = copy.deepcopy(sample_ods(39915))
    while len(single["equilibrium.time_slice"]) > 1:
        del single["equilibrium.time_slice"][len(single["equilibrium.time_slice"]) - 1]
    single["equilibrium.time"] = np.asarray(single["equilibrium.time"])[:1]
    record = next(
        r for r in describe_entries(normalize_entries(single)) if r.name == "equilibrium_profile_q"
    )
    assert record.sequence == {} and _slice_control(record) is None


# -- pinned controls: what the control layer produced before the sequence record ---------------


def _record(**facts):
    from dataclasses import replace

    base = next(r for r in describe_entries(normalize_entries(sample_ods(39915))) if r.name == "plasma_current_time")
    return replace(base, **{"times": {}, "slices": {}, **facts})


PINNED = [
    pytest.param(
        {"times": {"start": 0.26, "stop": 0.36, "count": 2500, "option": "time_index", "selected": 1158,
                   "kind": "samples", "coordinate": "time", "unit": "s"}},
        ("time_index", "range", "Time sample (260-360 ms)", 1158, (0, 2499, 1), (), "slice"),
        id="dense samples",
    ),
    pytest.param(
        {"times": {"start": 0.30944, "stop": 0.32146, "count": 602, "option": "frame_index", "selected": 0,
                   "kind": "frames", "coordinate": "time", "unit": "s"}},
        ("frame_index", "range", "Camera frame (309-321 ms)", 0, (0, 601, 1), (), "slice"),
        id="camera frames",
    ),
    pytest.param(
        {"slices": {"total": 5, "usable": (0, 2, 3), "times": (0.31, 0.32, 0.33, 0.34, 0.35), "selected": 1,
                    "container": "equilibrium.time_slice", "kind": "stored", "option": "time_slice"}},
        ("time_slice", "choice", "Equilibrium slice", 2, (0, 2, 3), ("0: 310.0 ms", "2: 330.0 ms", "3: 340.0 ms"), "slice"),
        id="stored slices with a gap and an unusable selected",
    ),
    pytest.param(
        {"slices": {"total": 2, "usable": (0, 1), "times": (0.31, 0.32), "selected": 0,
                    "container": "core_profiles.profiles_1d", "kind": "stored", "option": "time_slice"}},
        ("time_slice", "choice", "core_profiles slice", 0, (0, 1), ("0: 310.0 ms", "1: 320.0 ms"), "slice"),
        id="another IDS's slices",
    ),
    pytest.param(
        {"times": {"start": 0.26, "stop": 0.36, "count": 2500, "shared": False,
                   "kind": "samples", "coordinate": "time", "unit": "s"}},
        None,
        id="channels on no shared grid",
    ),
    pytest.param(
        {"slices": {"total": 1, "usable": (0,), "times": (0.31,), "selected": 0,
                    "container": "equilibrium.time_slice", "kind": "stored", "option": "time_slice"}},
        None,
        id="one stored slice",
    ),
]


@pytest.mark.parametrize(("facts", "expected"), PINNED)
def test_the_slice_control_is_pinned(facts, expected):
    record = _record(**facts)
    found = [c for c in controls_for(record) if c.group == "slice"]
    if expected is None:
        assert found == [] and record.sequence == {}
        return
    (control,) = found
    assert (control.name, control.kind, control.label, control.default, tuple(control.options),
            tuple(control.labels), control.group) == expected
    assert isinstance(record.sequence["states"], tuple)


def test_a_vacuum_map_refuses_time_and_time_index_together():
    from vaft.plot.backend.recipes import build_model

    with pytest.raises(ValueError, match="one of time=, time_index="):
        build_model("vacuum_field", normalize_entries(sample_ods(39915)), time=0.3, time_index=10)
