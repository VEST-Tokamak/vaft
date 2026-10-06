"""The main-chamber gauge changed from a PKR 251 to an IKR 251 at 46993 (#1543)."""

from __future__ import annotations

import numpy as np
import pytest

from vaft.machine_mapping.utils import calibrate_vest_signal, resolve_vest_diagnostic


def _pressure(shot: int, volts: float) -> float:
    config = resolve_vest_diagnostic(shot, "barometry_main")
    return float(calibrate_vest_signal(np.array([volts]), config["calibration"])[0])


@pytest.mark.parametrize(("shot", "name"), [(46992, "PKR-251 Main Gauge"), (46993, "IKR-251 Main Gauge")])
def test_the_gauge_identity_follows_the_shot(shot, name):
    config = resolve_vest_diagnostic(shot, "barometry_main")
    assert config["gauge"]["name"] == name
    assert config["source"]["field"] == 12


def test_pkr_formula_up_to_46992():
    assert _pressure(46992, 3.25) == pytest.approx(2.4 * 10 ** (1.667 * 3.25 - 11.46))


def test_ikr_formula_from_46993():
    assert _pressure(46993, 2.17) == pytest.approx(2.4 * 10 ** (2.0 * 2.17 - 10.625))


def test_the_base_and_fill_pressures_stay_continuous_across_the_change():
    """Era medians of the raw gauge output (every 10th shot 44000-48960).

    Base 3.24 V -> 2.17 V and fill 3.95 V -> 2.70 V: with the IKR formula on
    the new gauge both stay within a factor of 3 of the old era, where the PKR
    formula alone would put them 40-120x low.
    """
    for before, after in ((3.24, 2.17), (3.95, 2.70)):
        ratio = _pressure(46993, after) / _pressure(46992, before)
        assert 1 / 3 < ratio < 3
        assert _pressure(46992, after) / _pressure(46992, before) < 1 / 30
