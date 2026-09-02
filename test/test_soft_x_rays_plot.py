import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from vaft.machine_mapping.soft_x_rays import soft_x_rays_from_digitizer_csv
from vaft.plot.soft_x_rays import (
    plot_soft_x_ray_los,
    plot_soft_x_ray_pattern,
    plot_soft_x_ray_signal,
    plot_soft_x_ray_spectrogram,
)


def _tiny_sxr_ods(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    time = np.arange(512, dtype=float)
    ch0 = np.sin(2.0 * np.pi * time / 32.0)
    ch1 = np.cos(2.0 * np.pi * time / 64.0)
    np.savetxt(data_root / "digitizer_22577_12345.csv", np.vstack([ch0, ch1]), delimiter=",")
    return soft_x_rays_from_digitizer_csv(
        12345,
        22577,
        data_root=data_root,
        sample_rate=1024.0,
        time_offset=0.0,
    )


def test_soft_x_ray_plot_helpers_return_figures(tmp_path):
    ods = _tiny_sxr_ods(tmp_path)

    fig, ax = plot_soft_x_ray_los(ods, channels=[0, 1], show=False)
    assert fig is ax.figure
    assert len(ax.lines) >= 2
    plt.close(fig)

    fig, ax = plot_soft_x_ray_signal(ods, channels=[0, 1], show=False)
    assert len(ax.lines) == 2
    plt.close(fig)

    fig, ax = plot_soft_x_ray_spectrogram(ods, channel=0, nperseg=128, max_frequency=400.0, show=False)
    assert ax.collections
    plt.close(fig)

    fig, ax = plot_soft_x_ray_pattern(ods, channels=[0, 1], show=False)
    assert ax.collections
    plt.close(fig)


def test_canonical_sxr_renderers_reach_a_real_vest_sxr_ods(tmp_path):
    """The canonical SXR plots must find the data the VEST mapper actually writes.

    Regression: every soft_x_rays spec required ``channel.{i}.power.data``, which
    no VAFT mapper produces -- the mapper writes ``brightness``, as IMAS defines
    for a detector signal.  The time, spectrogram and spectrum renderers were
    therefore unreachable on any VEST SXR ODS.
    """
    from vaft.omas import plotting

    ods = _tiny_sxr_ods(tmp_path)
    supported = {row["name"] for row in plotting.available_plots(ods)}

    assert {
        "soft_x_rays_time_power",
        "soft_x_rays_spectrogram",
        "soft_x_rays_spectrum",
    } <= supported

    fig, ax = plotting.plot_soft_x_rays_time_power(ods, channels=[0, 1], show=False)
    assert len(ax.get_lines()) == 2
    plt.close(fig)

    fig, ax = plotting.plot_soft_x_rays_spectrum(ods, channel=0, nperseg=128, show=False)
    assert ax.get_xscale() == "log"
    plt.close(fig)


def test_power_data_still_works_when_a_source_provides_it(tmp_path):
    # ``power`` stays a supported spelling for externally sourced ODS.
    from vaft.omas import plotting

    ods = _tiny_sxr_ods(tmp_path)
    time = np.asarray(ods["soft_x_rays.channel.0.brightness.time"]).ravel()
    values = np.asarray(ods["soft_x_rays.channel.0.brightness.data"]).ravel()
    del ods["soft_x_rays.channel.0.brightness"]
    del ods["soft_x_rays.channel.1.brightness"]
    # IMAS gives power.data the same (band, time) shape brightness has.
    ods["soft_x_rays.channel.0.power.data"] = values.reshape(1, -1)
    ods["soft_x_rays.channel.0.power.time"] = time

    fig, ax = plotting.plot_soft_x_rays_time_power(ods, channels=[0], show=False)
    assert len(ax.get_lines()) == 1
    plt.close(fig)


def test_multi_band_brightness_still_reaches_the_power_fallback(tmp_path):
    """A brightness trace with several real energy bands must not mask power.data.

    _first_array used to raise on the first candidate's band ambiguity, so the
    declared power.data fallback was unreachable.
    """
    from vaft.omas import plotting

    ods = _tiny_sxr_ods(tmp_path)
    time = np.asarray(ods["soft_x_rays.channel.0.brightness.time"]).ravel()
    values = np.asarray(ods["soft_x_rays.channel.0.brightness.data"]).ravel()
    # Three real energy bands: not reducible to one trace.
    ods["soft_x_rays.channel.0.brightness.data"] = np.tile(values, (3, 1))
    ods["soft_x_rays.channel.0.power.data"] = values.reshape(1, -1)
    ods["soft_x_rays.channel.0.power.time"] = time

    fig, ax = plotting.plot_soft_x_rays_spectrum(ods, channel=0, nperseg=128, show=False)
    assert len(ax.get_lines()) >= 1
    plt.close(fig)


def test_one_multi_band_channel_does_not_abort_a_multi_channel_figure(tmp_path):
    from vaft.omas import plotting

    ods = _tiny_sxr_ods(tmp_path)
    values = np.asarray(ods["soft_x_rays.channel.0.brightness.data"]).ravel()
    ods["soft_x_rays.channel.0.brightness.data"] = np.tile(values, (3, 1))

    fig, ax = plotting.plot_soft_x_rays_time_power(ods, channels=[0, 1], show=False)

    # Channel 0 is skipped, channel 1 still draws.
    assert len(ax.get_lines()) == 1
    plt.close(fig)


def test_multi_band_with_no_fallback_raises_the_informative_error(tmp_path):
    from vaft.omas import plotting

    ods = _tiny_sxr_ods(tmp_path)
    values = np.asarray(ods["soft_x_rays.channel.0.brightness.data"]).ravel()
    ods["soft_x_rays.channel.0.brightness.data"] = np.tile(values, (3, 1))

    with pytest.raises(ValueError, match="energy bands"):
        plotting.plot_soft_x_rays_spectrum(ods, channel=0, nperseg=128, show=False)
