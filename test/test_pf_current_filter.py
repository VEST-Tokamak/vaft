"""The PF coil-current filter as a named composition (#2162 B3.5).

``vaft.machine_mapping.pf_active._legacy_pf_filter`` designed a SciPy
``firwin`` low-pass (default **Hamming** window, tap count and cutoff from
vest.yaml), applied it forward-backward -- forward only on a record too short
for ``filtfilt`` -- and then smoothed with a 10-sample
``scipy.ndimage.uniform_filter1d``.  The frozen copy below is that function
verbatim from develop 75b154e31 (after #2135); the composition that replaces
it is held to it bit for bit, and its phase behaviour is pinned against
closed-form expectations rather than against either implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage, signal

from vaft.machine_mapping.utils import resolve_vest_diagnostic
from vaft.process.signal_processing import apply_fir_filter


def _frozen_legacy_pf_filter(values, processing):
    filter_config = processing["filter"]
    taps = signal.firwin(
        int(filter_config["taps"]),
        float(filter_config["cutoff_frequency"]),
        pass_zero="lowpass",
        fs=float(filter_config["sample_rate"]),
    )
    filtered = apply_fir_filter(taps, values, short_record="causal")
    return ndimage.uniform_filter1d(filtered, size=min(int(filter_config["smoothing_window"]), values.size))


def _processing(shot=39915):
    return resolve_vest_diagnostic(shot, "pf_active")["processing"]


def _filter(values, processing):
    from vaft.machine_mapping import pf_active

    return pf_active._legacy_pf_filter(values, processing)


def test_vest_yaml_pf_filter_values_are_the_ones_this_file_assumes():
    config = _processing()["filter"]
    assert config == {
        "sample_rate": 25000.0,
        "cutoff_frequency": 2500.0,
        "taps": 251,
        "smoothing_window": 10,
    }


@pytest.mark.parametrize("length", [20_000, 754, 753, 200, 5])
def test_pf_filter_matches_frozen_reference(length):
    """754 = 3 * 251 + 1 is the shortest record filtfilt takes; 753 and below
    take the causal fallback; 5 is shorter than the smoothing window."""
    rng = np.random.default_rng(length)
    time = np.arange(length) / 25_000.0
    values = 1e3 * np.sin(2 * np.pi * 50.0 * time) + 30.0 * rng.standard_normal(length)
    processing = _processing()
    np.testing.assert_array_equal(_filter(values, processing), _frozen_legacy_pf_filter(values, processing))
