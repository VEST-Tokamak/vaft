"""The default of a multi-channel time plot is the channels that carry a signal.

``selection`` gains the signal presets ``active`` (default: flagged valid and
non-zero), ``valid`` and ``all``; an explicit selection is drawn as named.
A scalar family's ``synthetic="both"`` no longer repeats its own waveform.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import omas
import pytest

import vaft
import vaft.omas
from vaft.omas.entries import normalize_entries
from vaft.plot.backend.recipes import build_model, selection_presets
from vaft.plot.selection import ACTIVE, ALL, PRESETS, SIGNAL_PRESETS, VALID


@pytest.fixture(scope="module")
def sample():
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(str(vaft.data.data_path("samples/39915/omas.json.gz")))


def _three_probes():
    ods = omas.ODS(consistency_check=False)
    time = np.linspace(0.0, 1.0, 5)
    ods["magnetics.time"] = time
    for index, (values, validity) in enumerate((
        (np.sin(time), 0),          # valid, carries a signal
        (np.zeros(5), 0),           # valid, never moved
        (np.cos(time), -1),         # flagged invalid
    )):
        base = f"magnetics.b_field_pol_probe.{index}"
        ods[f"{base}.field.data"] = values
        ods[f"{base}.field.time"] = time
        ods[f"{base}.field.validity"] = validity
        ods[f"{base}.position.r"] = 0.5 + 0.1 * index
        ods[f"{base}.position.z"] = 0.0
        ods[f"{base}.identifier"] = f"P{index}"
    return ods


def test_signal_presets_are_part_of_the_selection_vocabulary():
    assert SIGNAL_PRESETS == (ACTIVE, VALID, ALL)
    assert set(SIGNAL_PRESETS) <= set(PRESETS) and set(SIGNAL_PRESETS) <= set(selection_presets())


@pytest.mark.parametrize("selection, expected", [
    (None, ["P0"]), (ACTIVE, ["P0"]), (VALID, ["P0", "P1"]), (ALL, ["P0", "P1", "P2"]),
])
def test_the_default_keeps_channels_that_carry_a_valid_signal(selection, expected):
    model = build_model("b_field_probe_time_field", normalize_entries(_three_probes()), selection=selection)
    assert [trace.index for trace in model.series] == [int(name[1]) for name in expected]


def test_an_explicit_selection_is_drawn_as_named_invalid_or_flat():
    entries = normalize_entries(_three_probes())
    model = build_model("b_field_probe_time_field", entries, selection=[2])
    assert len(model.series) == 1 and model.series[0].is_invalid_channel
    model = build_model("b_field_probe_time_field", entries, selection=["P1"])
    assert len(model.series) == 1 and not np.any(model.series[0].y)
    with pytest.raises(ValueError, match="supported presets: .*active, valid, all"):
        build_model("b_field_probe_time_field", entries, selection="bogus")


def test_the_sample_probes_and_coils_lose_their_dead_channels_by_default(sample):
    entries = normalize_entries(sample)
    everything = build_model("b_field_probe_time_field", entries, selection="all")
    default = build_model("b_field_probe_time_field", entries)
    assert sum(t.is_invalid_channel for t in everything.series) == 1
    assert len(default.series) == len(everything.series) - 1
    assert not any(t.is_invalid_channel for t in default.series)
    coils = build_model("pf_coil_time_current", entries)
    assert [t.label for t in coils.series] == ["PF1", "PF5", "PF6", "PF9", "PF10"]
    assert len(build_model("pf_coil_time_current", entries, selection="all").series) == 10
    figure, axes = vaft.omas.plot_pf_coil_time_current(sample, layout="subplots")
    assert [a.get_title() for a in axes.ravel() if a.get_visible()] == ["PF1", "PF5", "PF6", "PF9", "PF10"]
    plt.close(figure)


def test_a_scalar_family_does_not_repeat_its_own_waveform_as_a_constraint(sample):
    entries = normalize_entries(sample)
    model = build_model("diamagnetic_flux_time", entries, synthetic="both")
    assert [trace.role for trace in model.series] == [""]
    model = build_model("plasma_current_time", entries, synthetic="both")
    assert "constraint" not in {trace.role for trace in model.series}
