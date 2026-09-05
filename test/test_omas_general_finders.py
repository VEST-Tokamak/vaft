"""The general onset finders on the shared timings, and the time-convention memo (issue #409, PR-v).

``find_breakdown_onset``, ``find_pulse_duration``, ``find_ip_onset``,
``find_vloop_onset`` and ``find_pf_active_onset`` used to carry their own
rules (a 5 % threshold on H-alpha by index, the argmax of a flux record);
they now answer from ``vaft.omas.plasma_timing`` and
``vaft.omas.discharge_timing``.  ``change_time_convention`` derives the
convention origins once and never recomputes what it stored.
"""
from __future__ import annotations

import inspect
import logging

import numpy as np
import pytest
from omas import ODS, CodeParameters

import vaft.omas
import vaft.omas.plasma_timing
from vaft.omas.discharge_timing import discharge_timing
from vaft.omas.general import (
    ONSET_METHOD,
    ONSET_METHOD_LEGACY,
    change_time_convention,
    find_bt,
    find_breakdown_onset,
    find_ip_onset,
    find_pf_active_onset,
    find_pulse_duration,
    find_vloop_onset,
)
from vaft.omas.plasma_timing import PlasmaTimingError, plasma_timing

from _plasma_timing_fixtures import current, grid, light, pickup_only, pipeline_ods, synthetic_ods


def test_the_submodule_is_not_shadowed_by_the_star_export():
    assert inspect.ismodule(vaft.omas.plasma_timing)
    assert callable(vaft.omas.find_breakdown_onset)


@pytest.mark.parametrize("loader", [lambda: pipeline_ods(39915), vaft.omas.sample_ods])
def test_the_finders_answer_from_the_shared_timings(loader):
    ods = loader()
    timing = plasma_timing(ods)
    events = discharge_timing(ods)

    assert find_breakdown_onset(ods) == timing.onset
    assert find_pulse_duration(ods) == pytest.approx(timing.offset - timing.onset)
    assert find_ip_onset(ods) == timing.ip.start
    assert find_vloop_onset(ods) == events.vloop.zero_crossing
    assert find_vloop_onset(ods) == pytest.approx(0.3022, abs=5e-4)
    assert find_breakdown_onset(ods) == pytest.approx(0.3063, abs=5e-4)
    assert np.isfinite(find_bt(ods))

    onsets = find_pf_active_onset(ods)
    assert len(onsets) == len(events.pf_onsets) == 10
    for value, coil in zip(onsets, events.pf_onsets):
        assert (value == coil.time) if coil.found else np.isnan(value)


def test_no_plasma_is_an_error_that_names_the_reason():
    t = grid()
    ods = synthetic_ods(slow=0.002 * np.random.default_rng(1).standard_normal(t.size), ip=pickup_only(t), t=t)

    with pytest.raises(ValueError, match="no plasma timing: .*ip_principal"):
        find_breakdown_onset(ods)
    with pytest.raises(ValueError, match="no plasma-current pulse"):
        find_ip_onset(ods)
    with pytest.raises(ValueError, match="no loop-voltage zero crossing: .*loop_not_found"):
        find_vloop_onset(ods)
    assert find_pf_active_onset(ods) == []

    no_ip = synthetic_ods(slow=light(t), t=t)
    with pytest.raises(PlasmaTimingError):
        find_breakdown_onset(no_ip)


def _memo(ods):
    return dict(ods["summary.code.parameters"])


def test_the_convention_round_trip_is_exact_and_recorded(caplog, capsys):
    ods = vaft.omas.sample_ods()
    before = {tuple(path): np.array(ods[path], copy=True) for path in ods.paths()
              if path and path[-1] in ("time", "onset", "offset")
              and isinstance(ods[path], (np.ndarray, float, int))}
    assert before

    with caplog.at_level(logging.INFO, logger="vaft.omas.general"):
        change_time_convention(ods, convention="breakdown")
        memo = _memo(ods)
        assert memo["time_convention"] == "breakdown"
        assert memo["onset_method"] == ONSET_METHOD
        assert memo["breakdown_onset_source"] == "h_alpha_primary"
        assert memo["ip_onset_source"] == "ip_principal"
        assert memo["vloop_onset_source"].startswith("magnetics.flux_loop.5 dflux_dt")
        assert memo["onset_flags"] == ""
        assert ods["magnetics.time"][0] == pytest.approx(before[("magnetics", "time")][0] - memo["breakdown_onset"])
        change_time_convention(ods, convention="daq")

    assert "shift" in caplog.text and "daq -> breakdown" in caplog.text
    assert capsys.readouterr().out == ""
    for path, values in before.items():
        np.testing.assert_allclose(np.asarray(ods[list(path)]), values, rtol=0, atol=1e-12, err_msg=str(path))
    assert _memo(ods)["time_convention"] == "daq"


def test_a_stored_memo_is_never_recomputed(monkeypatch):
    ods = vaft.omas.sample_ods()
    change_time_convention(ods, convention="vloop")
    stored = _memo(ods)

    def boom(*_args, **_kwargs):
        raise AssertionError("the origins must not be recomputed")

    monkeypatch.setattr("vaft.omas.general._derive_onsets", boom)
    change_time_convention(ods, convention="ip")
    change_time_convention(ods, convention="daq")
    memo = _memo(ods)
    assert memo["time_convention"] == "daq"
    assert {k: memo[k] for k in stored if k != "time_convention"} == {
        k: stored[k] for k in stored if k != "time_convention"
    }


def test_a_legacy_memo_on_a_shifted_product_keeps_its_origins():
    ods = vaft.omas.sample_ods()
    params = ods.setdefault("summary.code.parameters", CodeParameters())
    params["time_convention"] = "vloop"
    params["vloop_onset"] = 0.30388     # the argmax-of-flux value the old finder gave
    params["ip_onset"] = 0.3
    params["breakdown_onset"] = 0.3307

    change_time_convention(ods, convention="daq")

    memo = _memo(ods)
    assert memo["vloop_onset"] == 0.30388 and memo["breakdown_onset"] == 0.3307
    assert memo["onset_method"] == ONSET_METHOD_LEGACY
    assert "legacy_origins_retained" in memo["onset_flags"].split(";")
    assert memo["time_convention"] == "daq"


def test_a_legacy_memo_on_an_unshifted_product_is_rederived():
    ods = vaft.omas.sample_ods()
    params = ods.setdefault("summary.code.parameters", CodeParameters())
    params["time_convention"] = "daq"
    params["vloop_onset"] = 0.30388
    params["ip_onset"] = 0.3
    params["breakdown_onset"] = 0.3307

    change_time_convention(ods, convention="vloop")

    memo = _memo(ods)
    assert memo["onset_method"] == ONSET_METHOD
    assert memo["vloop_onset"] == pytest.approx(0.3022, abs=5e-4)
    assert "legacy_memo_rederived" in memo["onset_flags"].split(";")


def test_a_missing_origin_refuses_its_convention_with_the_reason():
    t = grid()
    ods = synthetic_ods(slow=light(t), ip=current(t), t=t)   # no flux loops: no vloop origin

    change_time_convention(ods, convention="breakdown")
    memo = _memo(ods)
    assert "vloop_onset" not in memo
    assert "vloop_onset:not_found:" in memo["onset_flags"]
    with pytest.raises(ValueError, match=r"no 'vloop' origin: vloop_onset:not_found:.*loop_not_found"):
        change_time_convention(ods, convention="vloop")
    with pytest.raises(ValueError, match="Unknown convention"):
        change_time_convention(ods, convention="trigger")


def test_the_approach_flag_reaches_the_memo():
    ods = pipeline_ods(41524)
    change_time_convention(ods, convention="vloop")
    memo = _memo(ods)
    assert "vloop:approached_without_crossing" in memo["onset_flags"].split(";")
    assert 0.310 <= memo["vloop_onset"] <= 0.320
