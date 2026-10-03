"""Selection decides which channels; validity decides how their flags are drawn (issue #1380, phase 3).

The signal presets (``active``, ``valid``) used to drop a condemned channel
before the renderer saw it, so ``validity="show"`` could never show it.  A
``validity=`` the caller states now owns the flags: the channel reaches the
renderer, demoted under ``show``, left out under ``mask``, plain under
``ignore``.  Without ``validity=`` nothing changes.  Separately, a state that
is usable for navigation (a stored magnetics sample) may hold flagged data:
stepping through time re-reads the flags of each state.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft
from _sample_fixtures import sample_ods
from vaft.omas.entries import normalize_entries
from vaft.plot.backend.recipes import build_model
from vaft.plot.style import INVALID_COLOR

#: B-probe 25 of shot 39915 carries a channel-level invalid flag.
CONDEMNED = 25
#: Every flux loop of 39915 is flagged valid up to this sample and not after it.
LAST_VALID = 1999


@pytest.fixture(scope="module")
def sample():
    return sample_ods(39915)


@pytest.fixture(scope="module")
def entries(sample):
    return normalize_entries(sample)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _indices(model):
    return [trace.index for trace in model.series]


def _lines(axes):
    return [line for axis in np.atleast_1d(axes).ravel() for line in axis.lines]


# -- selection vs validity ---------------------------------------------------------------


def test_without_validity_the_presets_still_leave_condemned_channels_out(entries):
    for selection in (None, "active", "valid"):
        options = {} if selection is None else {"selection": selection}
        assert CONDEMNED not in _indices(build_model("b_field_probe_time_field", entries, **options))
    assert CONDEMNED in _indices(build_model("b_field_probe_time_field", entries, selection="all"))


@pytest.mark.parametrize("selection", [None, "active", "valid", "all"])
@pytest.mark.parametrize("validity", ["show", "mask", "ignore"])
def test_a_stated_validity_brings_the_condemned_channel_to_the_renderer(entries, selection, validity):
    options = {"validity": validity} | ({} if selection is None else {"selection": selection})
    model = build_model("b_field_probe_time_field", entries, **options)
    trace = next(t for t in model.series if t.index == CONDEMNED)
    assert trace.is_invalid_channel


def test_active_still_drops_a_channel_without_signal_whatever_the_validity(sample):
    """Validity owns the flags, not the signal check: a dead channel stays out of ``active``."""
    import copy

    dead = copy.deepcopy(sample)
    path = "magnetics.b_field_pol_probe.3.field.data"
    dead[path] = np.zeros_like(np.asarray(sample[path], dtype=float))
    dead_entries = normalize_entries(dead)
    for validity in ("show", "mask", "ignore"):
        assert 3 not in _indices(build_model("b_field_probe_time_field", dead_entries, validity=validity))
    assert 3 in _indices(build_model("b_field_probe_time_field", dead_entries, validity="show", selection="all"))


@pytest.mark.parametrize("selection", [None, "valid"])
def test_through_a_preset_show_demotes_mask_removes_and_ignore_draws_plainly(sample, selection):
    """The preset itself passes the condemned channel on; the old code dropped it before drawing."""
    options = {} if selection is None else {"selection": selection}

    def lines(**more):
        figure, axes = vaft.omas.plot_b_field_probe_time_field(sample, **options, **more)
        drawn = _lines(axes)
        return [line for line in drawn if line.get_color() == INVALID_COLOR], len(drawn)

    _, default_total = lines()
    demoted, total_show = lines(validity="show")
    assert demoted and total_show == default_total + 1
    hidden, total_mask = lines(validity="mask")
    assert not hidden and total_mask == default_total
    plain, total_ignore = lines(validity="ignore")
    assert not plain and total_ignore == total_show


def _flagged_thomson():
    import copy

    ods = copy.deepcopy(sample_ods(48224))
    ods["thomson_scattering.channel.0.t_e.validity"] = -2
    return ods


def test_a_channel_profile_point_carries_its_flag_under_a_stated_validity():
    """Thomson/CES profiles: the condemned channel's point is flagged, so mask removes it."""
    ods = _flagged_thomson()
    entries = normalize_entries(ods)
    name = "thomson_scattering_profile_electron_temperature"
    default = build_model(name, entries)
    stated = build_model(name, entries, validity="mask")
    (before,), (after,) = default.series, stated.series
    assert len(after.x) == len(before.x) + 1
    assert after.valid_mask is not None and int((~after.valid_mask).sum()) == 1

    def points(validity):
        figure, axis = vaft.omas.plot_thomson_scattering_profile_electron_temperature(ods, validity=validity)
        return sum(int(np.isfinite(line.get_ydata()).sum()) for line in axis.lines)

    assert points("mask") == len(before.x)
    assert points("ignore") == len(after.x)


def _timed_flag_on_thomson():
    """Channel 0 is valid as a channel; one of its samples is flagged (``validity_timed``)."""
    import copy

    ods = copy.deepcopy(sample_ods(48224))
    data = np.asarray(ods["thomson_scattering.channel.0.t_e.data"])
    timed = np.zeros(data.shape, dtype=int)
    timed[0] = -1  # the sample the profile shows at time_slice=0
    ods["thomson_scattering.channel.0.t_e.validity_timed"] = timed
    return ods


def test_without_validity_a_per_sample_flag_does_not_demote_a_profile_point():
    """Cold review 0.8.0 delta-absorb-13-infra F6: the flags are attached only with a stated validity=.

    The contract of this module is that without ``validity=`` nothing changes
    from 0.7.1, which drew a channel profile without a per-point mask.  The
    #1380 phase-3 builder attached the per-sample flag unconditionally, so a
    code-0 channel whose ``validity_timed`` is negative at the shown sample
    was drawn demoted by default.  A stated ``validity=`` owns the flags: it
    carries them (``show``), removes them (``mask``), or ignores them.
    """
    ods = _timed_flag_on_thomson()
    entries = normalize_entries(ods)
    name = "thomson_scattering_profile_electron_temperature"
    (default,) = build_model(name, entries).series
    assert default.valid_mask is None and default.validity is None
    (shown,) = build_model(name, entries, validity="show").series
    assert shown.valid_mask is not None and int((~shown.valid_mask).sum()) == 1
    assert len(shown.x) == len(default.x)

    def points(**kwargs):
        figure, axis = vaft.omas.plot_thomson_scattering_profile_electron_temperature(ods, **kwargs)
        return sum(int(np.isfinite(line.get_ydata()).sum()) for line in axis.lines)

    assert points() == points(validity="ignore") == len(default.x)
    assert points(validity="mask") == len(default.x) - 1


def test_the_interactive_controls_keep_the_channels_a_stated_validity_brought_in(sample):
    result = vaft.omas.plot_b_field_probe_time_field(sample, interactive=True, interaction_backend="none", validity="show")
    shown = len(_lines(result.axes))
    static_figure, static_axes = vaft.omas.plot_b_field_probe_time_field(sample, validity="show")
    assert shown == len(_lines(static_axes))
    result.state.set("validity", "mask")
    assert len(_lines(result.axes)) == shown - 1  # the condemned channel, now masked
    result.state.set("validity", "show")
    assert len(_lines(result.axes)) == shown


def test_opening_the_controls_without_validity_draws_the_static_default(sample):
    result = vaft.omas.plot_b_field_probe_time_field(sample, interactive=True, interaction_backend="none")
    figure, axes = vaft.omas.plot_b_field_probe_time_field(sample)
    assert len(_lines(result.axes)) == len(_lines(axes))


# -- state usability vs content validity ---------------------------------------------------


def _flags(model):
    flags = []
    for panel in model.models:
        for trace in panel.series:
            mask = trace.valid_mask
            flags.extend([True] * len(trace.x) if mask is None else list(np.asarray(mask, dtype=bool)))
    return flags


def test_stepping_through_time_re_reads_each_states_flags(entries):
    """Sample 2000 onwards is a usable state whose content is flagged: it stays in the sequence."""
    before = build_model("flux_loop_spatial_flux", entries, time_index=LAST_VALID, selection="all")
    after = build_model("flux_loop_spatial_flux", entries, time_index=LAST_VALID + 2, selection="all")
    assert all(_flags(before)) and len(_flags(before)) == 11
    assert not any(_flags(after)) and len(_flags(after)) == 11


@pytest.mark.parametrize(("validity", "drawn_after"), [("show", 11), ("mask", 0), ("ignore", 11)])
def test_the_validity_mode_applies_to_every_navigated_state(sample, validity, drawn_after):
    result = vaft.omas.plot_flux_loop_spatial_flux(
        sample, interactive=True, interaction_backend="none", selection="all", validity=validity,
    )

    def points():
        return sum(int(np.isfinite(line.get_ydata()).sum()) for line in _lines(result.axes))

    result.state.set("time_index", LAST_VALID)
    assert points() == 11
    result.state.set("time_index", LAST_VALID + 2)
    assert points() == drawn_after
    result.state.set("time_index", LAST_VALID)
    assert points() == 11
    assert result.state["validity"] == validity  # stepping time left the mode alone


def test_an_animation_keeps_flagged_states_in_its_sequence(sample):
    movie = vaft.omas.plot_flux_loop_spatial_flux(
        sample, time_index=[LAST_VALID - 1, LAST_VALID + 2], animation=True, validity="mask", dpi=30,
    )
    assert len(list(movie.frames())) == 2
    assert movie.metadata["style"]["validity"] == "mask"
