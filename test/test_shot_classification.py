"""Shot classification from pressure, H-alpha light and plasma current.

`classify_shot` called `is_signal_active(data, threshold=...)` against a
signature that takes `var_ratio_thresh`/`change_ratio_thresh`. Every call raised
TypeError, a bare `except` swallowed it, and the function answered 'Vacuum' for
every shot -- including packaged 39915, an 80 kA discharge.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import omas
import pytest

import vaft
import vaft.omas


PACKAGED_PLASMA_SHOTS = (39915, 41524, 41672)


def _load(shot: int):
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(vaft.data.sample(shot, representation="omas"))


@pytest.mark.parametrize("shot", PACKAGED_PLASMA_SHOTS)
def test_a_packaged_discharge_is_not_reported_as_a_vacuum_shot(shot):
    ods = _load(shot)
    assert float(np.max(ods["magnetics.ip.0.data"])) > 1.0e4  # it is a discharge
    assert vaft.omas.classify_shot(ods) == "Plasma"


@pytest.mark.parametrize("shot", PACKAGED_PLASMA_SHOTS)
def test_the_two_classifiers_agree(shot):
    """They used to disagree: 'Vacuum' from one, 'Plasma' from the other."""
    ods = _load(shot)
    assert vaft.omas.find_shotclass(ods) == vaft.omas.classify_shot(ods)


def _shot(pressure, halpha, ip=None):
    ods = omas.ODS(consistency_check=False)
    ods["barometry.gauge.0.pressure.data"] = np.asarray(pressure, dtype=float)
    ods["spectrometer_uv.channel.0.processed_line.0.intensity.data"] = np.asarray(
        halpha, dtype=float
    )
    if ip is not None:
        ods["magnetics.ip.0.data"] = np.asarray(ip, dtype=float)
    return ods


def test_each_class_is_reachable():
    flat = np.full(64, 1.0)
    varying = np.linspace(0.0, 1.0, 64) + np.tile([0.0, 0.3], 32)
    assert vaft.omas.classify_shot(_shot(flat, varying)) == "Vacuum"
    assert vaft.omas.classify_shot(_shot(varying, flat)) == "BD failure"
    assert vaft.omas.classify_shot(_shot(varying, varying)) == "Plasma"
    # Gas and light, but the current never rose: the breakdown failed.
    assert vaft.omas.classify_shot(_shot(varying, varying, np.zeros(64))) == "BD failure"
    # No current trace at all is not evidence against the light.
    assert vaft.omas.classify_shot(_shot(varying, varying)) == "Plasma"


def test_a_missing_signal_is_reported_not_called_vacuum():
    """An unrecorded gauge is not a vacuum shot, and asking must not create one."""
    empty = omas.ODS(consistency_check=False)
    with pytest.raises(KeyError, match="barometry.gauge.0.pressure.data"):
        vaft.omas.classify_shot(empty)
    # ODS.__getitem__ materializes a missing path; the check must not.
    assert list(empty.keys()) == []

    assert vaft.omas.find_shotclass(empty) is None  # the lenient form still answers


def test_the_threshold_reaches_the_detector():
    """Passing it under the wrong keyword is what broke this in the first place."""
    barely_moving = np.full(64, 1.0)
    barely_moving[::2] += 1.0e-4
    assert vaft.omas.classify_shot(_shot(barely_moving, barely_moving)) == "Plasma"
    assert (
        vaft.omas.classify_shot(
            _shot(barely_moving, barely_moving), pressure_threshold=1.0
        )
        == "Vacuum"
    )
