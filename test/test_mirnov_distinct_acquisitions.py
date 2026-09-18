"""One acquired channel is one measurement (issues #724, #825).

DAQ field 171 used to be published twice in ``magnetics.b_field_pol_probe``: as
equilibrium probe 36 (``MagneticFieldProbe_C2-05_Bz``, outboard family, 5:30)
and as a ``:phase_reference`` twin at 9:30, with byte-identical samples. The
toroidal mode fit then compared one waveform with itself, measured a phase
difference of exactly zero, and reported an ``n`` that was an identity.

Two guards, each on the thing itself:

* the mapper publishes a phase-reference channel only when the shot's
  equilibrium wiring does not already read its DAQ field;
* the plot asks the data, not the metadata: entries whose samples are
  identical count as one channel, so they can never supply a second position.
"""

from __future__ import annotations

import numpy as np
import pytest

import vaft
from vaft.machine_mapping.magnetics import (
    TOROIDAL_MIRNOV_REFERENCE_CHANNELS,
    _equilibrium_probe_field_codes,
    magnetics_wiring_for_shot,
    toroidal_array_for_shot,
    toroidal_mirnov_reference_channels,
    vfit_magnetics_static,
)
from vaft.machine_mapping.utils import get_path
from vaft.plot.backend.recipes import (
    _distinct_acquisitions,
    _mirnov_phase_available,
    _toroidal_phase_channels,
    _toroidal_phase_group,
)

NAME = "mirnov_spatial_phase"
SAMPLE_RATE = 100_000.0
TIME = np.arange(4096, dtype=float) / SAMPLE_RATE


def _probe(ods, index, angle_deg, data, *, r=0.796, z=0.02, name=None):
    base = f"magnetics.b_field_pol_probe.{index}"
    ods[f"{base}.name"] = name or f"P{index}"
    ods[f"{base}.position.r"] = r
    ods[f"{base}.position.z"] = z
    ods[f"{base}.position.phi"] = float(np.deg2rad(angle_deg))
    ods[f"{base}.voltage.time"] = TIME
    ods[f"{base}.voltage.data"] = np.asarray(data, dtype=float)


def _mode(angle_deg, n=1):
    return np.sin(2.0 * np.pi * 8_000.0 * TIME - n * np.deg2rad(angle_deg))


def _ods():
    from omas import ODS

    ods = ODS()
    ods["dataset_description.data_entry.pulse"] = 99999
    return ods


# ---------------------------------------------------------------------------
# the plot's guard
# ---------------------------------------------------------------------------

def test_one_waveform_under_two_angles_is_not_an_array():
    """The #724 shape: the same samples at 195 and 75 degrees."""
    ods = _ods()
    waveform = _mode(195.0)
    _probe(ods, 0, 195.0, waveform, name="MagneticFieldProbe_C2-05_Bz")
    _probe(ods, 1, 75.0, waveform.copy(), name="MagneticFieldProbe_C2-05_Bz")
    indices, angles = _toroidal_phase_channels(ods)
    assert _distinct_acquisitions(ods, indices, angles)[2] == [0, 1]
    reason = _mirnov_phase_available(ods)
    assert reason is not None and "own waveform" in reason
    assert NAME not in {record.name for record in vaft.omas.available_plots(ods)}


def test_naming_the_copies_explicitly_is_refused():
    from vaft.omas.entries import normalize_entries
    from vaft.plot.backend.recipes import build_model

    ods = _ods()
    waveform = _mode(0.0)
    _probe(ods, 0, 0.0, waveform)
    _probe(ods, 1, 120.0, _mode(120.0))
    _probe(ods, 2, 240.0, waveform.copy())
    with pytest.raises(ValueError, match="repeat another named channel"):
        build_model(
            NAME, normalize_entries(ods), channels=[0, 1, 2],
            frequencies=[8_000.0], window_size=512, preprocess=False, time=0.02,
        )


def test_a_copy_does_not_count_but_the_rest_of_the_array_still_does():
    """Three genuine positions plus a contested copy: the three remain."""
    ods = _ods()
    for index, angle in enumerate((0.0, 120.0, 240.0)):
        _probe(ods, index, angle, _mode(angle))
    _probe(ods, 3, 300.0, _mode(0.0))  # probe 0's samples, claimed at 300 deg
    indices, degrees = _toroidal_phase_group(ods)
    assert indices == [1, 2]  # 0 and 3 contradict each other and both go
    assert _mirnov_phase_available(ods) is None


def test_copies_that_agree_on_their_angle_collapse_to_one():
    ods = _ods()
    for index, angle in enumerate((0.0, 120.0, 240.0)):
        _probe(ods, index, angle, _mode(angle))
    _probe(ods, 3, 120.0, _mode(120.0))
    indices, _ = _toroidal_phase_group(ods)
    assert indices == [0, 1, 2]


def test_a_genuine_three_angle_array_is_offered():
    ods = _ods()
    for index, angle in enumerate((135.0, 225.0, 315.0)):
        _probe(ods, index, angle, _mode(angle, n=1))
    assert _mirnov_phase_available(ods) is None
    assert NAME in {record.name for record in vaft.omas.available_plots(ods)}


# ---------------------------------------------------------------------------
# the mapper's guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shot", [0, 30000, 35520, 39915, 45531])
def test_no_phase_reference_repeats_an_equilibrium_field(shot):
    published = {int(c["field_code"]) for c in toroidal_mirnov_reference_channels(shot)}
    assert not published & _equilibrium_probe_field_codes(shot)


def test_field_171_is_still_recorded_as_the_disputed_claim():
    """The table keeps the 9:30 assignment as a record; it is just not published."""
    disputed = [c for c in TOROIDAL_MIRNOV_REFERENCE_CHANNELS if int(c["field_code"]) == 171]
    assert len(disputed) == 1 and "#825" in disputed[0]["disputed"]
    assert 171 in _equilibrium_probe_field_codes(0)
    assert int(magnetics_wiring_for_shot(39915).channels[36]["field_code"]) == 171


@pytest.mark.parametrize("shot,count", [(0, 67), (30000, 67), (39915, 64)])
def test_the_static_layout_publishes_each_identifier_once(shot, count):
    payload = {}
    vfit_magnetics_static(payload, shot)
    probes = get_path(payload, "magnetics.b_field_pol_probe")
    names = [str(get_path(payload, f"magnetics.b_field_pol_probe.{i}.name")) for i in range(len(probes))]
    assert len(names) == count
    assert len(names) == len(set(names))
    assert names[36] == "MagneticFieldProbe_C2-05_Bz"


def test_the_reference_era_array_is_the_three_named_positions():
    era = toroidal_array_for_shot(30000)
    assert era["name"] == "phase_reference"
    assert era["clocks"] == pytest.approx((1.5, 5.5, 7.5))


# ---------------------------------------------------------------------------
# the packaged samples
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shot", [39915, 41524, 41672])
def test_the_packaged_equilibrium_samples_offer_no_toroidal_fit(shot):
    """They sit in the 35521-44155 gap, and now say so (#724, #825)."""
    ods = vaft.omas.sample_ods(shot)
    count = len(ods["magnetics.b_field_pol_probe"])
    assert count == 64
    identifiers = [ods[f"magnetics.b_field_pol_probe.{i}.identifier"] for i in range(count)]
    assert not [name for name in identifiers if str(name).endswith(":phase_reference")]
    assert toroidal_array_for_shot(shot)["name"] is None
    assert _mirnov_phase_available(ods) is not None
