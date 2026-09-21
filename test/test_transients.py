"""Current-quench, current-spike and vertical-position measurements (#1005).

Synthetic waveforms whose answers are known by construction: a linear quench
from 80 kA to zero over 5 ms has its 80 % and 20 % crossings 3 ms apart and a
slope of -16 MA/s; a Gaussian bump of 4 kA before it is the spike.  Nothing
here is named an IRE or a disruption -- the functions return numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.process.transients import (
    CurrentQuench,
    CurrentSpike,
    current_quench,
    current_spike,
)

FS = 250_000.0
IP0 = 80_000.0
T_QUENCH = 0.0102
QUENCH_S = 5e-3


def _ip(*, spike=0.0, noise=0.0, sign=1.0, seed=1, with_quench=True):
    rng = np.random.default_rng(seed)
    time = np.arange(int(0.02 * FS)) / FS
    ip = np.full(time.size, IP0)
    if with_quench:
        ramp = np.clip((time - T_QUENCH) / QUENCH_S, 0.0, 1.0)
        ip = IP0 * (1.0 - ramp)
    ip = ip + spike * np.exp(-0.5 * ((time - 0.0100) / 30e-6) ** 2)
    ip = ip + noise * rng.standard_normal(time.size)
    return time, sign * ip


class TestCurrentQuench:
    def test_a_linear_quench_is_measured(self):
        time, ip = _ip()
        result = current_quench(time, ip)
        assert isinstance(result, CurrentQuench)
        assert result.found and result.reason is None
        assert result.reference_current == pytest.approx(IP0)
        assert result.time_80 == pytest.approx(T_QUENCH + 0.2 * QUENCH_S, abs=1 / FS)
        assert result.time_20 == pytest.approx(T_QUENCH + 0.8 * QUENCH_S, abs=1 / FS)
        assert result.duration_80_20 == pytest.approx(0.6 * QUENCH_S, abs=2 / FS)
        assert result.extrapolated_quench_time == pytest.approx(QUENCH_S, abs=4 / FS)
        assert result.didt_min == pytest.approx(-IP0 / QUENCH_S, rel=1e-6)
        assert T_QUENCH <= result.time_didt_min <= T_QUENCH + QUENCH_S

    def test_a_negative_current_is_measured_on_its_magnitude(self):
        time, ip = _ip(sign=-1.0)
        result = current_quench(time, ip)
        assert result.polarity == -1
        assert result.reference_current == pytest.approx(IP0)
        assert result.didt_min == pytest.approx(-IP0 / QUENCH_S, rel=1e-6)

    def test_no_quench_inside_the_window_is_none_with_a_reason(self):
        time, ip = _ip(with_quench=False)
        result = current_quench(time, ip)
        assert not result.found
        assert result.time_20 is None and result.didt_min is None
        assert "20 %" in result.reason

    def test_the_window_restricts_the_search(self):
        time, ip = _ip()
        result = current_quench(time, ip, window=(0.0, 0.012))
        assert not result.found
        assert result.reason

    def test_a_zero_record_is_refused_with_a_reason(self):
        time = np.arange(100) / FS
        result = current_quench(time, np.zeros(100))
        assert not result.found and "no current" in result.reason

    def test_smoothing_tames_the_derivative_of_a_noisy_record(self):
        time, ip = _ip(noise=300.0)
        raw = current_quench(time, ip)
        smooth = current_quench(time, ip, smoothing_s=0.5e-3)
        assert smooth.didt_min == pytest.approx(-IP0 / QUENCH_S, rel=0.1)
        assert raw.didt_min < smooth.didt_min  # noise only makes the raw minimum deeper

    def test_an_explicit_reference_moves_the_crossings(self):
        time, ip = _ip()
        result = current_quench(time, ip, reference_current=100_000.0)
        # 80 kA is already the 80 % level of 100 kA: the crossing is the quench start.
        assert result.time_80 == pytest.approx(T_QUENCH, abs=1 / FS)
        assert result.time_20 == pytest.approx(T_QUENCH + 0.75 * QUENCH_S, abs=1 / FS)

    def test_a_reference_whose_80_percent_is_never_reached_is_a_reason(self):
        time, ip = _ip()
        # 80 % of 110 kA is 88 kA, above the 80 kA the record ever holds: no 80 %
        # crossing exists, so no duration may be reported.
        result = current_quench(time, ip, reference_current=110_000.0)
        assert not result.found
        assert result.time_80 is None and result.duration_80_20 is None
        assert "80 %" in result.reason
        assert result.reference_current == pytest.approx(110_000.0)

    def test_a_reference_five_times_the_peak_is_a_reason_not_a_wrapped_index(self):
        time, ip = _ip()
        # The whole record sits below 20 % of the reference: the fall search would
        # start at the peak itself and step to index -1 without a guard.
        result = current_quench(time, ip, reference_current=5.0 * IP0)
        assert not result.found
        assert result.time_20 is None and result.time_80 is None
        assert "80 %" in result.reason


class TestCurrentSpike:
    def test_a_spike_before_the_quench_is_found(self):
        time, ip = _ip(spike=4_000.0, noise=50.0)
        result = current_spike(time, ip)
        assert isinstance(result, CurrentSpike)
        assert result.found and result.reason is None
        assert result.time == pytest.approx(0.0100, abs=2 / FS)
        assert result.amplitude == pytest.approx(4_000.0, rel=0.1)
        assert result.relative_amplitude == pytest.approx(4_000.0 / IP0, rel=0.1)
        assert result.before == pytest.approx(current_quench(time, ip).time_80)

    def test_no_spike_is_none_with_a_reason(self):
        time, ip = _ip(noise=50.0)
        result = current_spike(time, ip)
        assert not result.found
        assert result.amplitude is None and result.time is None
        assert "noise" in result.reason

    def test_no_quench_and_no_before_is_none_with_a_reason(self):
        time, ip = _ip(with_quench=False)
        result = current_spike(time, ip)
        assert not result.found and "before" in result.reason

    def test_an_explicit_before_is_honoured(self):
        time, ip = _ip(spike=4_000.0, noise=50.0)
        assert not current_spike(time, ip, before=0.0095).found
        assert current_spike(time, ip, before=0.0101).found

    def test_a_negative_current_spike_is_positive_in_magnitude(self):
        time, ip = _ip(spike=4_000.0, noise=50.0, sign=-1.0)
        result = current_spike(time, ip)
        assert result.found and result.amplitude == pytest.approx(4_000.0, rel=0.1)


class TestVerticalPositionHistory:
    def _ods(self, times, z, ip=None):
        from omas import ODS

        ods = ODS()
        ods["equilibrium.ids_properties.homogeneous_time"] = 1
        ods["equilibrium.time"] = np.asarray(times, dtype=float)
        for index, (t, value) in enumerate(zip(times, z)):
            ods[f"equilibrium.time_slice.{index}.time"] = float(t)
            ods[f"equilibrium.time_slice.{index}.global_quantities.magnetic_axis.r"] = 0.4
            ods[f"equilibrium.time_slice.{index}.global_quantities.magnetic_axis.z"] = float(value)
            ods[f"equilibrium.time_slice.{index}.global_quantities.ip"] = (
                1e5 if ip is None else float(ip[index])
            )
        return ods

    def test_z_and_its_rate_follow_the_slices(self):
        from vaft.omas import vertical_position_history

        times = np.array([0.30, 0.31, 0.33, 0.34])
        z = 0.01 + 2.0 * (times - 0.30)  # 2 m/s drift
        result = vertical_position_history(self._ods(times, z))
        assert result.found and result.reason is None
        np.testing.assert_allclose(result.time, times)
        np.testing.assert_allclose(result.z_axis, z)
        np.testing.assert_allclose(result.dz_dt, 2.0)

    def test_slices_are_ordered_by_their_own_time(self):
        from vaft.omas import vertical_position_history

        times = np.array([0.31, 0.30, 0.32])
        z = np.array([0.02, 0.00, 0.04])
        result = vertical_position_history(self._ods(times, z))
        np.testing.assert_allclose(result.time, [0.30, 0.31, 0.32])
        np.testing.assert_allclose(result.z_axis, [0.00, 0.02, 0.04])
        np.testing.assert_allclose(result.dz_dt, 2.0)

    def test_a_slice_without_plasma_current_is_nan_not_a_position(self):
        from vaft.omas import vertical_position_history

        times = np.array([0.30, 0.31, 0.32, 0.33])
        z = np.array([0.0, 0.01, 0.02, 0.0])
        result = vertical_position_history(self._ods(times, z, ip=[1e5, 1e5, 1e5, 0.0]))
        assert np.isnan(result.z_axis[-1]) and np.isnan(result.dz_dt[-1])
        np.testing.assert_allclose(result.z_axis[:3], z[:3])
        np.testing.assert_array_equal(result.valid, [True, True, True, False])

    def test_no_equilibrium_is_none_with_a_reason(self):
        from omas import ODS

        from vaft.omas import vertical_position_history

        result = vertical_position_history(ODS())
        assert not result.found
        assert result.time is None and "equilibrium" in result.reason

    def test_a_single_slice_has_no_rate(self):
        from vaft.omas import vertical_position_history

        result = vertical_position_history(self._ods([0.3], [0.01]))
        assert not result.found and "two" in result.reason
        np.testing.assert_allclose(result.z_axis, [0.01])


class TestFluctuationBandwidths:
    def test_rates_come_from_the_stored_time_bases(self):
        from omas import ODS

        from vaft.omas import fluctuation_bandwidths

        ods = ODS()
        slow = np.arange(100) / 250e3
        fast = np.arange(400) / 2e6
        for index, time in enumerate((slow, slow, fast)):
            ods[f"magnetics.b_field_pol_probe.{index}.voltage.time"] = time
            ods[f"magnetics.b_field_pol_probe.{index}.voltage.data"] = np.zeros(time.size)
        sxr = np.arange(200) / 0.98e6
        ods["soft_x_rays.channel.0.brightness.time"] = sxr
        ods["soft_x_rays.channel.0.brightness.data"] = np.zeros((1, sxr.size))  # the DD stores (1, time)
        result = fluctuation_bandwidths(ods)
        assert set(result) == {
            "Mirnov / magnetic probes (250 kHz)",
            "Mirnov / magnetic probes (2000 kHz)",
            "Soft X-ray",
        }
        fs, nyquist = result["Mirnov / magnetic probes (250 kHz)"]
        assert fs == pytest.approx(250e3) and nyquist == pytest.approx(125e3)
        assert result["Soft X-ray"][1] == pytest.approx(0.49e6)

    def test_a_channel_without_data_is_not_counted(self):
        from omas import ODS

        from vaft.omas import fluctuation_bandwidths

        ods = ODS()
        ods["magnetics.b_field_pol_probe.0.voltage.time"] = np.arange(10) / 1e3
        assert fluctuation_bandwidths(ods) == {}

    def test_the_packaged_sample_reports_its_magnetics_rate(self):
        import vaft.omas

        result = vaft.omas.fluctuation_bandwidths(vaft.omas.sample_ods(39915))
        assert result, "the packaged sample carries magnetic probes"
        for fs, nyquist in result.values():
            assert nyquist == pytest.approx(fs / 2)
