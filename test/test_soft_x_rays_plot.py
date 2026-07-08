import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
